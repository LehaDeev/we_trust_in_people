"""
Handlers for auto/manual trading section:
    - trading menu with mode toggle
    - open positions list with unrealized P&L
    - trade history
    - trading settings (incl. commission and tax)
    - manual buy (ticker select -> balance/lot check -> execute)
    - manual sell (position select -> P&L preview -> confirm -> execute)
"""
import asyncio
from decimal import Decimal

from aiogram import Router
from aiogram.types import CallbackQuery
from sqlalchemy import select

from bot.keyboards import (
    back_to_trading,
    confirm_buy,
    confirm_sell,
    manual_buy_tickers,
    manual_sell_positions,
    trading_menu,
)
from config.settings import data_settings, trading_settings
from db.database import get_session
from db import trade_repo
from db.models import Asset, Trade
from tinkoff.instruments import get_instrument_by_ticker
from tinkoff.market_data import get_last_price, get_last_prices
from tinkoff.portfolio import get_rub_balance
from trading import state
from trading.executor import TradeExecutor
from trading.notifier import notify_close, notify_open
from trading.profitability import (
    adjusted_sl_price,
    adjusted_tp_price,
    breakeven_pct,
    calculate_pnl,
    format_pnl_breakdown,
)
from utils.logger import logger

router = Router(name="trading")

_REASON_LABEL = {
    "STOP_LOSS": "🔴 Стоп-лосс",
    "TAKE_PROFIT": "✅ Тейк-профит",
    "SELL_SIGNAL": "🔵 Сигнал SELL",
    "MANUAL": "🖐 Ручная продажа",
}

_executor = TradeExecutor()


def _mode_text(is_auto: bool) -> str:
    """Строка статуса режима торговли."""
    if is_auto:
        return "🤖 <b>Режим: Авто</b>\nScheduler торгует по ML-сигналам автоматически."
    return "🖐 <b>Режим: Ручной</b>\nScheduler приостановлен. Используй кнопки Купить / Продать."


async def _show_trading_menu(callback: CallbackQuery) -> None:
    """Вспомогательная функция: отрисовать меню торговли с балансом."""
    is_auto = state.is_auto()
    try:
        balance = await get_rub_balance()
        balance_str = f"💵 Свободно: <b>{balance:.2f} ₽</b>"
    except Exception:
        balance_str = "💵 Свободно: <i>н/д</i>"
    text = f"🤖 <b>Торговля</b>\n\n{_mode_text(is_auto)}\n\n{balance_str}"
    await callback.message.edit_text(
        text, reply_markup=trading_menu(is_auto), parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:trading")
async def cb_trading_menu(callback: CallbackQuery) -> None:
    """Показать меню раздела торговли."""
    await _show_trading_menu(callback)


@router.callback_query(lambda c: c.data == "trading:toggle")
async def cb_toggle(callback: CallbackQuery) -> None:
    """Переключить режим авто / ручной."""
    new_value = state.toggle()
    mode = "авто" if new_value else "ручной"
    await callback.answer(f"Режим переключён: {mode}")
    await _show_trading_menu(callback)


# ── Ручная покупка ────────────────────────────────────────────────────────────

@router.callback_query(lambda c: c.data == "trading:buy")
async def cb_buy_select(callback: CallbackQuery) -> None:
    """Показать список тикеров для ручной покупки."""
    tickers = data_settings.tickers
    await callback.message.edit_text(
        "🛒 <b>Ручная покупка</b>\n\nВыбери тикер:",
        reply_markup=manual_buy_tickers(tickers),
        parse_mode="HTML",
    )
    await callback.answer()


@router.callback_query(
    lambda c: c.data
    and c.data.startswith("trading:buy:")
    and not c.data.startswith("trading:buy:confirm:")
)
async def cb_buy_preview(callback: CallbackQuery) -> None:
    """
    Показать превью покупки: количество бумаг, сумму, прогноз SL/TP.

    Не исполняет ордер — только отображает детали и кнопку подтверждения.
    """
    ticker = callback.data.split(":", 2)[2]
    await callback.answer(f"⏳ Загружаю данные {ticker}...")

    try:
        async with get_session() as session:
            # Найти актив в БД
            result = await session.execute(
                select(Asset).where(Asset.ticker == ticker)
            )
            asset = result.scalar_one_or_none()
            if not asset:
                await callback.message.edit_text(
                    f"❌ Актив <b>{ticker}</b> не найден в БД.\n"
                    "Запустите сборщик данных.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            # Получить текущую цену и размер лота параллельно
            current_price, instrument_info = await asyncio.gather(
                get_last_price(asset.figi),
                get_instrument_by_ticker(ticker),
            )

            if current_price <= 0:
                await callback.message.edit_text(
                    f"❌ Не удалось получить цену для <b>{ticker}</b>.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            lot_size = instrument_info.lot if instrument_info and instrument_info.lot > 0 else 1
            lots = trading_settings.lots_per_ticker
            total_qty = lots * lot_size
            needed = current_price * total_qty

            # Прогноз P&L при TP и SL
            sl_pct = Decimal(str(trading_settings.stop_loss_pct))
            tp_pct = Decimal(str(trading_settings.take_profit_pct))
            tp_price = current_price * (Decimal("1") + tp_pct)
            sl_price = current_price * (Decimal("1") - sl_pct)
            tp_breakdown = calculate_pnl(current_price, tp_price, lots, lot_size)
            sl_breakdown = calculate_pnl(current_price, sl_price, lots, lot_size)
            be_pct = breakeven_pct()

            # Проверить баланс
            open_trades = await trade_repo.get_open_trades(session)
            balance = await get_rub_balance()

        if balance < needed:
            await callback.message.edit_text(
                f"⚠️ <b>Недостаточно средств для покупки {ticker}</b>\n\n"
                f"Нужно:    <b>{needed:.2f} ₽</b>\n"
                f"  ({lots} лот × {lot_size} шт × {current_price:.2f} ₽)\n"
                f"Доступно: <b>{balance:.2f} ₽</b>",
                reply_markup=back_to_trading(),
                parse_mode="HTML",
            )
            return

        if len(open_trades) >= trading_settings.max_open_positions:
            await callback.message.edit_text(
                f"⚠️ Достигнут лимит позиций ({trading_settings.max_open_positions}).",
                reply_markup=back_to_trading(),
                parse_mode="HTML",
            )
            return

        text = (
            f"🛒 <b>Покупка: {ticker}</b>\n\n"
            f"Количество:  <b>{lots} лот × {lot_size} шт = {total_qty} бумаг</b>\n"
            f"Цена:        <b>{current_price:.2f} ₽</b> за бумагу\n"
            f"Сумма:       <b>{needed:.2f} ₽</b>\n\n"
            f"SL (−{sl_pct * 100:.1f}%):  {sl_price:.2f} ₽\n"
            f"TP (+{tp_pct * 100:.1f}%): {tp_price:.2f} ₽\n\n"
            f"📊 <b>Прогноз при TP (+{tp_pct * 100:.1f}%):</b>\n"
            f"Чистая прибыль: <b>+{tp_breakdown.net_pnl:.2f} ₽</b>  "
            f"(комиссии −{tp_breakdown.buy_commission + tp_breakdown.sell_commission:.2f} ₽, "
            f"НДФЛ −{tp_breakdown.tax:.2f} ₽)\n"
            f"📊 <b>Прогноз при SL (−{sl_pct * 100:.1f}%):</b>\n"
            f"Чистый убыток: <b>{sl_breakdown.net_pnl:.2f} ₽</b>  "
            f"(комиссии −{sl_breakdown.buy_commission + sl_breakdown.sell_commission:.2f} ₽)\n\n"
            f"💵 Доступно: {balance:.2f} ₽\n"
            f"<i>Безубыточность от +{be_pct * 100:.2f}% роста цены</i>"
        )
        await callback.message.edit_text(
            text, reply_markup=confirm_buy(ticker), parse_mode="HTML"
        )

    except Exception as e:
        logger.error("Ошибка превью покупки", ticker=ticker, error=str(e))
        await callback.message.edit_text(
            f"❌ Ошибка при загрузке данных для <b>{ticker}</b>.",
            reply_markup=back_to_trading(),
            parse_mode="HTML",
        )


@router.callback_query(lambda c: c.data and c.data.startswith("trading:buy:confirm:"))
async def cb_buy_confirm(callback: CallbackQuery) -> None:
    """Выполнить ручную покупку после подтверждения пользователем."""
    ticker = callback.data.split(":", 3)[3]
    await callback.answer(f"⏳ Покупаю {ticker}...")

    try:
        async with get_session() as session:
            # Найти актив в БД
            result = await session.execute(
                select(Asset).where(Asset.ticker == ticker)
            )
            asset = result.scalar_one_or_none()
            if not asset:
                await callback.message.edit_text(
                    f"❌ Актив <b>{ticker}</b> не найден в БД.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            # Получить цену и лот-сайз
            current_price, instrument_info = await asyncio.gather(
                get_last_price(asset.figi),
                get_instrument_by_ticker(ticker),
            )

            if current_price <= 0:
                await callback.message.edit_text(
                    f"❌ Не удалось получить цену для <b>{ticker}</b>.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            lot_size = instrument_info.lot if instrument_info and instrument_info.lot > 0 else 1
            lots = trading_settings.lots_per_ticker
            total_qty = lots * lot_size
            needed = current_price * total_qty

            # Повторная проверка баланса и лимита позиций
            balance = await get_rub_balance()
            if balance < needed:
                await callback.message.edit_text(
                    f"⚠️ <b>Недостаточно средств для покупки {ticker}</b>\n\n"
                    f"Нужно: <b>{needed:.2f} ₽</b>  |  Доступно: <b>{balance:.2f} ₽</b>",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            open_trades = await trade_repo.get_open_trades(session)
            if len(open_trades) >= trading_settings.max_open_positions:
                await callback.message.edit_text(
                    f"⚠️ Достигнут лимит позиций ({trading_settings.max_open_positions}).",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            # Открыть позицию
            trade = await _executor.open_position(
                session=session,
                asset=asset,
                instrument_uid=asset.figi,
                current_price=current_price,
                lot_size=lot_size,
            )

        if trade:
            tp_pct = Decimal(str(trading_settings.take_profit_pct))
            sl_pct = Decimal(str(trading_settings.stop_loss_pct))
            tp_price = current_price * (Decimal("1") + tp_pct)
            sl_price = current_price * (Decimal("1") - sl_pct)
            tp_breakdown = calculate_pnl(current_price, tp_price, lots, lot_size)
            sl_breakdown = calculate_pnl(current_price, sl_price, lots, lot_size)
            be_pct = breakeven_pct()

            await notify_open(
                ticker=ticker,
                price=trade.entry_price,
                lots=trade.lots,
                lot_size=lot_size,
                stop_loss=trade.stop_loss_price,
                take_profit=trade.take_profit_price,
            )
            text = (
                f"✅ <b>Куплено: {ticker}</b>\n\n"
                f"Количество:  <b>{lots} лот × {lot_size} шт = {total_qty} бумаг</b>\n"
                f"Цена:        <b>{trade.entry_price:.2f} ₽</b> за бумагу\n"
                f"Сумма:       <b>{needed:.2f} ₽</b>\n\n"
                f"SL: {trade.stop_loss_price:.2f} ₽  TP: {trade.take_profit_price:.2f} ₽\n\n"
                f"📊 <b>Прогноз при TP (+{tp_pct * 100:.1f}%):</b>\n"
                f"Чистая прибыль: <b>+{tp_breakdown.net_pnl:.2f} ₽</b>  "
                f"(комиссии −{tp_breakdown.buy_commission + tp_breakdown.sell_commission:.2f} ₽, "
                f"НДФЛ −{tp_breakdown.tax:.2f} ₽)\n"
                f"📊 <b>Прогноз при SL (−{sl_pct * 100:.1f}%):</b>\n"
                f"Чистый убыток: <b>{sl_breakdown.net_pnl:.2f} ₽</b>  "
                f"(комиссии −{sl_breakdown.buy_commission + sl_breakdown.sell_commission:.2f} ₽)\n\n"
                f"<i>Безубыточность от +{be_pct * 100:.2f}% роста цены</i>"
            )
        else:
            text = f"❌ Не удалось открыть позицию по <b>{ticker}</b>."

    except Exception as e:
        logger.error("Ошибка ручной покупки", ticker=ticker, error=str(e))
        text = f"❌ Ошибка при покупке <b>{ticker}</b>."

    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )


# ── Ручная продажа: список → превью → подтверждение ──────────────────────────

@router.callback_query(lambda c: c.data == "trading:sell")
async def cb_sell_select(callback: CallbackQuery) -> None:
    """Показать список открытых позиций для ручной продажи."""
    await callback.answer("⏳ Загружаю позиции...")
    try:
        async with get_session() as session:
            open_trades = await trade_repo.get_open_trades(session)
            if not open_trades:
                await callback.message.edit_text(
                    "💸 <b>Ручная продажа</b>\n\nОткрытых позиций нет.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            asset_ids = [t.asset_id for t in open_trades]
            result = await session.execute(
                select(Asset).where(Asset.id.in_(asset_ids))
            )
            assets = {a.id: a for a in result.scalars().all()}

        positions = [
            (t.id, assets[t.asset_id].ticker, f"{t.entry_price:.2f}")
            for t in open_trades
            if t.asset_id in assets
        ]

        await callback.message.edit_text(
            "💸 <b>Ручная продажа</b>\n\n"
            "Выбери позицию для просмотра расчёта P&L:",
            reply_markup=manual_sell_positions(positions),
            parse_mode="HTML",
        )
    except Exception as e:
        logger.error("Ошибка загрузки позиций для продажи", error=str(e))
        await callback.message.edit_text(
            "❌ Не удалось загрузить позиции.",
            reply_markup=back_to_trading(),
            parse_mode="HTML",
        )


@router.callback_query(lambda c: c.data and c.data.startswith("trading:sell:preview:"))
async def cb_sell_preview(callback: CallbackQuery) -> None:
    """
    Показать расчёт P&L перед продажей.

    FIFO: продаётся позиция с самой ранней датой открытия по данному активу.
    Показывает breakdown: gross, комиссии, НДФЛ, net — и кнопку подтверждения.
    """
    trade_id_str = callback.data.split(":", 3)[3]
    try:
        trade_id = int(trade_id_str)
    except ValueError:
        await callback.answer("Некорректный ID сделки")
        return

    await callback.answer("⏳ Рассчитываю P&L...")

    try:
        async with get_session() as session:
            result = await session.execute(
                select(Trade).where(Trade.id == trade_id, Trade.status == "OPEN")
            )
            trade = result.scalar_one_or_none()
            if not trade:
                await callback.message.edit_text(
                    "⚠️ Позиция не найдена или уже закрыта.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            asset_result = await session.execute(
                select(Asset).where(Asset.id == trade.asset_id)
            )
            asset = asset_result.scalar_one_or_none()
            if not asset:
                await callback.message.edit_text(
                    "❌ Актив не найден.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

        # Текущая цена
        current_price = await get_last_price(asset.figi)
        if current_price <= 0:
            current_price = trade.entry_price

        lot_size = getattr(trade, "lot_size", 1) or 1
        breakdown = calculate_pnl(
            entry_price=trade.entry_price,
            exit_price=current_price,
            lots=trade.lots,
            lot_size=lot_size,
        )

        total_qty = trade.lots * lot_size
        entry_total = trade.entry_price * total_qty
        exit_total = current_price * total_qty

        warning = (
            "\n\n⚠️ <b>Продажа убыточна</b> после комиссий и НДФЛ!\n"
            "Рекомендуем дождаться роста или закрыть через тейк-профит."
            if not breakdown.is_profitable
            else ""
        )

        text = (
            f"💸 <b>Продажа: {asset.ticker}</b>\n"
            f"FIFO — позиция от {trade.opened_at.strftime('%d.%m.%Y %H:%M')}\n\n"
            f"Количество:   <b>{trade.lots} лот × {lot_size} шт = {total_qty} бумаг</b>\n"
            f"Куплено:      <b>{trade.entry_price:.2f} ₽</b> × {total_qty} = {entry_total:.2f} ₽\n"
            f"Продажа:      <b>{current_price:.2f} ₽</b> × {total_qty} = {exit_total:.2f} ₽\n\n"
            f"{format_pnl_breakdown(breakdown)}"
            f"{warning}"
        )

        await callback.message.edit_text(
            text,
            reply_markup=confirm_sell(trade_id),
            parse_mode="HTML",
        )

    except Exception as e:
        logger.error("Ошибка расчёта P&L при продаже", trade_id=trade_id, error=str(e))
        await callback.message.edit_text(
            "❌ Не удалось рассчитать P&L.",
            reply_markup=back_to_trading(),
            parse_mode="HTML",
        )


@router.callback_query(lambda c: c.data and c.data.startswith("trading:sell:confirm:"))
async def cb_sell_execute(callback: CallbackQuery) -> None:
    """Выполнить ручную продажу после подтверждения пользователем."""
    trade_id_str = callback.data.split(":", 3)[3]
    try:
        trade_id = int(trade_id_str)
    except ValueError:
        await callback.answer("Некорректный ID сделки")
        return

    await callback.answer("⏳ Продаю...")

    try:
        async with get_session() as session:
            result = await session.execute(
                select(Trade).where(Trade.id == trade_id, Trade.status == "OPEN")
            )
            trade = result.scalar_one_or_none()
            if not trade:
                await callback.message.edit_text(
                    "⚠️ Позиция не найдена или уже закрыта.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            asset_result = await session.execute(
                select(Asset).where(Asset.id == trade.asset_id)
            )
            asset = asset_result.scalar_one_or_none()
            if not asset:
                await callback.message.edit_text(
                    "❌ Актив не найден.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            current_price = await get_last_price(asset.figi)
            if current_price <= 0:
                current_price = trade.entry_price

            lot_size = getattr(trade, "lot_size", 1) or 1
            breakdown = calculate_pnl(
                entry_price=trade.entry_price,
                exit_price=current_price,
                lots=trade.lots,
                lot_size=lot_size,
            )

            closed_trade = await _executor.close_position(
                session=session,
                trade=trade,
                asset=asset,
                instrument_uid=asset.figi,
                current_price=current_price,
                reason="MANUAL",
            )

        net_pnl = closed_trade.pnl or Decimal("0")
        net_sign = "+" if net_pnl >= 0 else ""
        await notify_close(
            ticker=asset.ticker,
            entry_price=trade.entry_price,
            exit_price=closed_trade.exit_price or current_price,
            reason="MANUAL",
            net_pnl=net_pnl,
            gross_pnl=breakdown.gross_pnl,
            commission=breakdown.buy_commission + breakdown.sell_commission,
            tax=breakdown.tax,
        )
        text = (
            f"✅ <b>Продано: {asset.ticker}</b>\n"
            f"Вход: {trade.entry_price:.2f} ₽ → Выход: {closed_trade.exit_price:.2f} ₽\n\n"
            f"{format_pnl_breakdown(breakdown)}\n\n"
            f"<b>Итого: {net_sign}{net_pnl:.2f} ₽</b>"
        )
    except Exception as e:
        logger.error("Ошибка ручной продажи", trade_id=trade_id, error=str(e))
        text = "❌ Ошибка при продаже позиции."

    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )


# ── Позиции / История / Параметры ─────────────────────────────────────────────

@router.callback_query(lambda c: c.data == "trading:positions")
async def cb_positions(callback: CallbackQuery) -> None:
    """Показать все открытые позиции с нереализованным P&L."""
    await callback.answer("⏳ Загружаю позиции...")
    try:
        async with get_session() as session:
            trades = await trade_repo.get_open_trades(session)
            asset_ids = list({t.asset_id for t in trades})
            if asset_ids:
                result = await session.execute(
                    select(Asset).where(Asset.id.in_(asset_ids))
                )
                assets = {a.id: a for a in result.scalars().all()}
            else:
                assets = {}

        if not trades:
            text = "📋 <b>Открытые позиции</b>\n\nПозиций нет."
        else:
            # Получаем текущие цены для всех позиций одним запросом
            figis = [assets[t.asset_id].figi for t in trades if t.asset_id in assets]
            try:
                prices = await get_last_prices(figis) if figis else {}
            except Exception:
                prices = {}

            lines = [f"📋 <b>Открытые позиции ({len(trades)})</b>", ""]
            for trade in trades:
                asset = assets.get(trade.asset_id)
                ticker = asset.ticker if asset else f"asset#{trade.asset_id}"
                lot_size = getattr(trade, "lot_size", 1) or 1

                current_price = prices.get(asset.figi, Decimal("0")) if asset else Decimal("0")
                if current_price > 0:
                    breakdown = calculate_pnl(
                        entry_price=trade.entry_price,
                        exit_price=current_price,
                        lots=trade.lots,
                        lot_size=lot_size,
                    )
                    net_sign = "+" if breakdown.net_pnl >= 0 else ""
                    pnl_icon = "🟢" if breakdown.is_profitable else "🔴"
                    pnl_str = (
                        f"{pnl_icon} чистый P&L: {net_sign}{breakdown.net_pnl:.2f} ₽  "
                        f"(сейчас {current_price:.2f} ₽)"
                    )
                else:
                    pnl_str = "цена н/д"

                lines.append(
                    f"<b>{ticker}</b>  {trade.lots} лот × {lot_size} шт  "
                    f"по {trade.entry_price:.2f} ₽\n"
                    f"  SL: {trade.stop_loss_price:.2f} ₽  "
                    f"TP: {trade.take_profit_price:.2f} ₽\n"
                    f"  {pnl_str}"
                )
            text = "\n".join(lines)
    except Exception as e:
        logger.error("Ошибка загрузки позиций", error=str(e))
        text = "❌ Не удалось загрузить позиции."

    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )


@router.callback_query(lambda c: c.data == "trading:history")
async def cb_history(callback: CallbackQuery) -> None:
    """Показать историю последних 10 закрытых сделок."""
    await callback.answer("⏳ Загружаю историю...")
    try:
        async with get_session() as session:
            trades = await trade_repo.get_trade_history(session, limit=10)
            asset_ids = list({t.asset_id for t in trades})
            if asset_ids:
                result = await session.execute(
                    select(Asset).where(Asset.id.in_(asset_ids))
                )
                assets = {a.id: a for a in result.scalars().all()}
            else:
                assets = {}

        if not trades:
            text = "📜 <b>История сделок</b>\n\nЗакрытых сделок нет."
        else:
            lines = [f"📜 <b>Последние сделки ({len(trades)})</b>", ""]
            for trade in trades:
                asset = assets.get(trade.asset_id)
                ticker = asset.ticker if asset else f"asset#{trade.asset_id}"
                net_pnl = trade.pnl or Decimal("0")
                net_sign = "+" if net_pnl >= 0 else ""
                reason = _REASON_LABEL.get(trade.close_reason or "", "⚪ Закрыта")
                lines.append(
                    f"{reason}  <b>{ticker}</b>  "
                    f"чистый P&L: <b>{net_sign}{net_pnl:.2f} ₽</b>"
                )
            text = "\n".join(lines)
    except Exception as e:
        logger.error("Ошибка загрузки истории", error=str(e))
        text = "❌ Не удалось загрузить историю сделок."

    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )


@router.callback_query(lambda c: c.data == "trading:stats")
async def cb_stats(callback: CallbackQuery) -> None:
    """Показать агрегированную статистику по всем сделкам."""
    await callback.answer("⏳ Считаю статистику...")

    try:
        async with get_session() as session:
            stats = await trade_repo.get_trade_stats(session)
    except Exception as e:
        logger.error("Ошибка получения статистики", error=str(e))
        await callback.message.edit_text(
            "❌ Не удалось загрузить статистику.",
            reply_markup=back_to_trading(),
            parse_mode="HTML",
        )
        return

    total = stats["total_closed"]

    if total == 0:
        text = (
            "📊 <b>Статистика</b>\n\n"
            f"Открытых позиций: <b>{stats['open_count']}</b>\n\n"
            "<i>Закрытых сделок пока нет.</i>"
        )
    else:
        pnl = stats["total_pnl"]
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_icon = "🟢" if pnl >= 0 else "🔴"

        avg = stats["avg_pnl"]
        avg_sign = "+" if avg >= 0 else ""

        win_rate = stats["win_rate"]

        # Лучшая / худшая сделка
        best = stats["best"]
        worst = stats["worst"]
        best_str = (
            f"<b>{best['ticker']}</b> +{best['pnl']:.2f} ₽"
            if best else "—"
        )
        worst_str = (
            f"<b>{worst['ticker']}</b> {worst['pnl']:.2f} ₽"
            if worst else "—"
        )

        # Разбивка по причине
        by_r = stats["by_reason"]
        sell_n = by_r.get("SELL_SIGNAL", 0)
        sl_n   = by_r.get("STOP_LOSS",   0)
        tp_n   = by_r.get("TAKE_PROFIT", 0)
        man_n  = by_r.get("MANUAL",      0)

        text = (
            "📊 <b>Статистика торговли</b>\n\n"
            f"Открытых позиций:  <b>{stats['open_count']}</b>\n"
            f"Закрытых сделок:   <b>{total}</b> "
            f"(прибыльных: {stats['wins']}, убыточных: {stats['losses']})\n"
            f"Win rate:          <b>{win_rate:.1f}%</b>\n\n"
            f"{pnl_icon} Суммарный P&L:  <b>{pnl_sign}{pnl:.2f} ₽</b>\n"
            f"Средний P&L:       <b>{avg_sign}{avg:.2f} ₽</b>\n\n"
            f"🏆 Лучшая сделка:  {best_str}\n"
            f"💀 Худшая сделка:  {worst_str}\n\n"
            f"<b>Причины закрытия:</b>\n"
            f"🔵 По сигналу SELL: {sell_n}\n"
            f"✅ Тейк-профит:     {tp_n}\n"
            f"🔴 Стоп-лосс:       {sl_n}\n"
            + (f"🖐 Ручная:          {man_n}\n" if man_n else "")
        )

    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )


@router.callback_query(lambda c: c.data == "trading:status")
async def cb_status(callback: CallbackQuery) -> None:
    """Показать текущие параметры автоторговли, комиссии и баланс счёта."""
    await callback.answer("⏳ Загружаю параметры...")
    cfg = trading_settings
    is_auto = state.is_auto()
    mode_str = "Авто" if is_auto else "Ручной"

    try:
        balance = await get_rub_balance()
        balance_str = f"{balance:.2f} ₽"
    except Exception:
        balance_str = "н/д"

    # Фактическое изменение цены при срабатывании SL/TP (gross, в % от цены входа)
    entry_ref = Decimal("1")
    tp_gross_pct = (adjusted_tp_price(entry_ref, cfg.take_profit_pct) - 1) * 100
    sl_gross_pct = (1 - adjusted_sl_price(entry_ref, cfg.stop_loss_pct)) * 100

    text = (
        "ℹ️ <b>Параметры торговли</b>\n\n"
        f"💵 Свободно:       <b>{balance_str}</b>\n\n"
        f"Режим:             <b>{mode_str}</b>\n"
        f"Уверенность:       <b>≥ {cfg.confidence_threshold * 100:.0f}%</b>\n"
        f"Лотов на сделку:   <b>{cfg.lots_per_ticker}</b>\n"
        f"Макс. позиций:     <b>{cfg.max_open_positions}</b>\n"
        f"Интервал:          <b>{cfg.check_interval_seconds} сек</b>\n\n"
        f"💸 <b>Комиссии:</b>\n"
        f"Брокер:  <b>{cfg.broker_commission_pct * 100:.2f}%</b> за покупку + "
        f"<b>{cfg.broker_commission_pct * 100:.2f}%</b> за продажу\n"
        f"НДФЛ:    <b>{cfg.tax_pct * 100:.0f}%</b> от чистой прибыли\n\n"
        f"📊 <b>Уровни закрытия позиции:</b>\n"
        f"Стоп-лосс:    цена падает на <b>−{sl_gross_pct:.2f}%</b> → чистый убыток <b>−{cfg.stop_loss_pct * 100:.1f}%</b>\n"
        f"Тейк-профит:  цена растёт на  <b>+{tp_gross_pct:.2f}%</b> → чистая прибыль <b>+{cfg.take_profit_pct * 100:.1f}%</b>\n\n"
        "<i>Параметры меняются в файле .env</i>"
    )
    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )
