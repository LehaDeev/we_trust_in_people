"""
Handlers for auto/manual trading section:
    - trading menu with mode toggle
    - open positions list
    - trade history
    - trading settings
    - manual buy (ticker select -> execute)
    - manual sell (position select -> execute)
"""
from decimal import Decimal

from aiogram import Router
from aiogram.types import CallbackQuery
from sqlalchemy import select

from bot.keyboards import (
    back_to_trading,
    manual_buy_tickers,
    manual_sell_positions,
    trading_menu,
)
from config.settings import data_settings, trading_settings
from db.database import get_session
from db import trade_repo
from db.models import Asset, Trade
from tinkoff.market_data import get_last_price
from trading import state
from trading.executor import TradeExecutor
from trading.notifier import notify_close, notify_open
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
    """Вспомогательная функция: отрисовать меню торговли."""
    is_auto = state.is_auto()
    text = f"🤖 <b>Торговля</b>\n\n{_mode_text(is_auto)}"
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


@router.callback_query(lambda c: c.data and c.data.startswith("trading:buy:"))
async def cb_buy_execute(callback: CallbackQuery) -> None:
    """Выполнить ручную покупку выбранного тикера."""
    ticker = callback.data.split(":", 2)[2]
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
                    f"❌ Актив <b>{ticker}</b> не найден в БД.\n"
                    "Запустите сборщик данных.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            # Проверить, нет ли уже открытой позиции
            existing = await trade_repo.get_open_trade_by_asset(session, asset.id)
            if existing:
                await callback.message.edit_text(
                    f"⚠️ По <b>{ticker}</b> уже есть открытая позиция.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            # Получить текущую цену
            current_price = await get_last_price(asset.figi)
            if current_price <= 0:
                await callback.message.edit_text(
                    f"❌ Не удалось получить цену для <b>{ticker}</b>.",
                    reply_markup=back_to_trading(),
                    parse_mode="HTML",
                )
                return

            # Проверить лимит позиций
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
            )

        if trade:
            await notify_open(
                ticker=ticker,
                price=trade.entry_price,
                lots=trade.lots,
                stop_loss=trade.stop_loss_price,
                take_profit=trade.take_profit_price,
            )
            text = (
                f"✅ <b>Куплено: {ticker}</b>\n"
                f"{trade.lots} лот(ов) по {trade.entry_price:.2f} ₽\n"
                f"SL: {trade.stop_loss_price:.2f} ₽  "
                f"TP: {trade.take_profit_price:.2f} ₽"
            )
        else:
            text = f"❌ Не удалось открыть позицию по <b>{ticker}</b>."

    except Exception as e:
        logger.error("Ошибка ручной покупки", ticker=ticker, error=str(e))
        text = f"❌ Ошибка при покупке <b>{ticker}</b>."

    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )


# ── Ручная продажа ────────────────────────────────────────────────────────────

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
            "💸 <b>Ручная продажа</b>\n\nВыбери позицию:",
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


@router.callback_query(lambda c: c.data and c.data.startswith("trading:sell:"))
async def cb_sell_execute(callback: CallbackQuery) -> None:
    """Выполнить ручную продажу выбранной позиции."""
    trade_id_str = callback.data.split(":", 2)[2]
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

            closed_trade = await _executor.close_position(
                session=session,
                trade=trade,
                asset=asset,
                instrument_uid=asset.figi,
                current_price=current_price,
                reason="MANUAL",
            )

        pnl = closed_trade.pnl or Decimal("0")
        pnl_sign = "+" if pnl >= 0 else ""
        await notify_close(
            ticker=asset.ticker,
            entry_price=trade.entry_price,
            exit_price=closed_trade.exit_price or current_price,
            reason="MANUAL",
            pnl=pnl,
        )
        text = (
            f"✅ <b>Продано: {asset.ticker}</b>\n"
            f"Вход: {trade.entry_price:.2f} ₽ → Выход: {closed_trade.exit_price:.2f} ₽\n"
            f"PnL: <b>{pnl_sign}{pnl:.2f} ₽</b>"
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
    """Показать все открытые позиции из базы данных."""
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
            lines = [f"📋 <b>Открытые позиции ({len(trades)})</b>", ""]
            for trade in trades:
                asset = assets.get(trade.asset_id)
                ticker = asset.ticker if asset else f"asset#{trade.asset_id}"
                lines.append(
                    f"<b>{ticker}</b>  {trade.lots} лот(ов)  "
                    f"по {trade.entry_price:.2f} ₽\n"
                    f"  SL: {trade.stop_loss_price:.2f} ₽  "
                    f"TP: {trade.take_profit_price:.2f} ₽"
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
                pnl = trade.pnl or Decimal("0")
                pnl_sign = "+" if pnl >= 0 else ""
                reason = _REASON_LABEL.get(trade.close_reason or "", "⚪ Закрыта")
                lines.append(
                    f"{reason}  <b>{ticker}</b>  "
                    f"<b>{pnl_sign}{pnl:.2f} ₽</b>"
                )
            text = "\n".join(lines)
    except Exception as e:
        logger.error("Ошибка загрузки истории", error=str(e))
        text = "❌ Не удалось загрузить историю сделок."

    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )


@router.callback_query(lambda c: c.data == "trading:status")
async def cb_status(callback: CallbackQuery) -> None:
    """Показать текущие параметры автоторговли из .env."""
    cfg = trading_settings
    is_auto = state.is_auto()
    mode_str = "Авто" if is_auto else "Ручной"
    text = (
        "ℹ️ <b>Параметры торговли</b>\n\n"
        f"Режим:            <b>{mode_str}</b>\n"
        f"Уверенность:      <b>≥ {cfg.confidence_threshold * 100:.0f}%</b>\n"
        f"Лотов на сделку:  <b>{cfg.lots_per_ticker}</b>\n"
        f"Стоп-лосс:        <b>{cfg.stop_loss_pct * 100:.1f}%</b>\n"
        f"Тейк-профит:      <b>{cfg.take_profit_pct * 100:.1f}%</b>\n"
        f"Макс. позиций:    <b>{cfg.max_open_positions}</b>\n"
        f"Интервал:         <b>{cfg.check_interval_seconds} сек</b>\n\n"
        "<i>Числовые параметры — в файле .env</i>"
    )
    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )
    await callback.answer()
