"""
Хендлеры раздела автоторговли:
    - меню торговли
    - открытые позиции
    - история сделок
    - параметры автоторговли
"""
from decimal import Decimal

from aiogram import Router
from aiogram.types import CallbackQuery
from sqlalchemy.orm import selectinload

from bot.keyboards import back_to_trading, trading_menu
from config.settings import trading_settings
from db.database import get_session
from db import trade_repo
from db.models import Trade
from utils.logger import logger

router = Router(name="trading")

_STATUS_EMOJI = {True: "✅ Включена", False: "⛔ Отключена"}
_REASON_LABEL = {
    "STOP_LOSS": "🔴 Стоп-лосс",
    "TAKE_PROFIT": "✅ Тейк-профит",
    "SELL_SIGNAL": "🔵 Сигнал SELL",
}


@router.callback_query(lambda c: c.data == "menu:trading")
async def cb_trading_menu(callback: CallbackQuery) -> None:
    """Показать меню раздела автоторговли."""
    status = _STATUS_EMOJI[trading_settings.enabled]
    text = (
        "🤖 <b>Автоторговля</b>\n\n"
        f"Статус: <b>{status}</b>"
    )
    await callback.message.edit_text(
        text, reply_markup=trading_menu(), parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(lambda c: c.data == "trading:positions")
async def cb_positions(callback: CallbackQuery) -> None:
    """Показать все открытые позиции из базы данных."""
    await callback.answer("⏳ Загружаю позиции...")
    try:
        async with get_session() as session:
            trades = await trade_repo.get_open_trades(session)
            # Подгружаем связанный Asset для каждой позиции
            from sqlalchemy import select
            from db.models import Asset
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
                pnl_sign = ""
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
            from sqlalchemy import select
            from db.models import Asset
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
    status = _STATUS_EMOJI[cfg.enabled]
    text = (
        "ℹ️ <b>Параметры торговли</b>\n\n"
        f"Статус:           <b>{status}</b>\n"
        f"Уверенность:      <b>≥ {cfg.confidence_threshold * 100:.0f}%</b>\n"
        f"Лотов на сделку:  <b>{cfg.lots_per_ticker}</b>\n"
        f"Стоп-лосс:        <b>{cfg.stop_loss_pct * 100:.1f}%</b>\n"
        f"Тейк-профит:      <b>{cfg.take_profit_pct * 100:.1f}%</b>\n"
        f"Макс. позиций:    <b>{cfg.max_open_positions}</b>\n"
        f"Интервал:         <b>{cfg.check_interval_seconds} сек</b>\n\n"
        "<i>Изменить параметры — в файле .env</i>"
    )
    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )
    await callback.answer()
