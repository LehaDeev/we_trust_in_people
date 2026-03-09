"""
Handlers ручной продажи: список позиций → превью P&L → подтверждение → исполнение.
"""
from decimal import Decimal

from aiogram import Router
from aiogram.types import CallbackQuery
from sqlalchemy import select

from bot.keyboards import back_to_trading, confirm_sell, manual_sell_positions
from db import trade_repo
from db.database import get_session
from db.models import Asset, Trade
from tinkoff.market_data import get_last_price
from trading.executor import TradeExecutor
from trading.notifier import notify_close
from trading.profitability import calculate_pnl, format_pnl_breakdown
from utils.logger import logger

router = Router(name="trading_sell")

_executor = TradeExecutor()


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
