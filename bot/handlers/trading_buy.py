"""
Handlers ручной покупки: выбор тикера → превью → подтверждение → исполнение.
"""
import asyncio
from decimal import Decimal

from aiogram import Router
from aiogram.types import CallbackQuery
from sqlalchemy import select

from bot.keyboards import back_to_trading, confirm_buy, manual_buy_tickers
from config.settings import data_settings, trading_settings
from db import trade_repo
from db.database import get_session
from db.models import Asset
from tinkoff.instruments import get_instrument_by_ticker
from tinkoff.market_data import get_last_price
from tinkoff.portfolio import get_rub_balance
from trading.executor import TradeExecutor
from trading.notifier import notify_open
from trading.profitability import breakeven_pct, calculate_pnl
from utils.logger import logger

router = Router(name="trading_buy")

_executor = TradeExecutor()


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

            sl_pct = Decimal(str(trading_settings.stop_loss_pct))
            tp_pct = Decimal(str(trading_settings.take_profit_pct))
            tp_price = current_price * (Decimal("1") + tp_pct)
            sl_price = current_price * (Decimal("1") - sl_pct)
            tp_breakdown = calculate_pnl(current_price, tp_price, lots, lot_size)
            sl_breakdown = calculate_pnl(current_price, sl_price, lots, lot_size)
            be_pct = breakeven_pct()

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
