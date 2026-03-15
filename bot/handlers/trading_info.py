"""
Handlers информационных экранов торговли:
    - открытые позиции с нереализованным P&L
    - история последних сделок
    - агрегированная статистика
    - параметры автоторговли
    - статус ML-моделей и ночного обучения
"""
import json
import zoneinfo
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from aiogram import Router
from aiogram.types import CallbackQuery
from sqlalchemy import select

from bot.keyboards import back_to_trading
from config.settings import ml_settings, trading_settings
from db import trade_repo
from db.database import get_session
from db.models import Asset
from tinkoff.market_data import get_last_prices
from tinkoff.portfolio import get_rub_balance
from trading import state
from trading.profitability import adjusted_sl_price, adjusted_tp_price, calculate_pnl
from utils.logger import logger

router = Router(name="trading_info")

_REASON_LABEL = {
    "STOP_LOSS": "🔴 Стоп-лосс",
    "TAKE_PROFIT": "✅ Тейк-профит",
    "SELL_SIGNAL": "🔵 Сигнал SELL",
    "MANUAL": "🖐 Ручная продажа",
}


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

        best = stats["best"]
        worst = stats["worst"]
        best_str = f"<b>{best['ticker']}</b> +{best['pnl']:.2f} ₽" if best else "—"
        worst_str = f"<b>{worst['ticker']}</b> {worst['pnl']:.2f} ₽" if worst else "—"

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

    entry_ref = Decimal("1")
    tp_gross_pct = (adjusted_tp_price(entry_ref, cfg.take_profit_pct) - 1) * 100
    sl_gross_pct = (1 - adjusted_sl_price(entry_ref, cfg.stop_loss_pct)) * 100

    text = (
        "ℹ️ <b>Параметры торговли</b>\n\n"
        f"💵 Свободно:       <b>{balance_str}</b>\n\n"
        f"Режим:             <b>{mode_str}</b>\n"
        f"Порог P&L входа:   <b>per-ticker</b> (fallback ≥ {cfg.confidence_threshold * 100:.3f}%)\n"
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


_RESULTS_PATH = Path("ml/weights/last_results.json")
_RETRAIN_TZ = zoneinfo.ZoneInfo("Europe/Moscow")


def _ml_nightly_status(trained_msk: datetime, force_tune: bool) -> str:
    """Сформировать строку статуса ночного дообучения.

    Считается «обучение сегодня» если прошло не более 26 часов —
    покрывает ночной запуск в любом часовом поясе и ручной запуск.
    """
    hours_ago = (datetime.now(_RETRAIN_TZ) - trained_msk).total_seconds() / 3600
    if abs(hours_ago) <= 26:
        label = "ручной с Optuna" if force_tune else "ночной"
        h = max(0, round(hours_ago))
        return f"✅ Да  ({label}, {h}ч назад)"
    return f"⚠️ Нет  (последнее {trained_msk.strftime('%d.%m %H:%M')} МСК, {hours_ago:.0f}ч назад)"


def _load_ticker_threshold(ticker: str) -> str:
    """Загрузить порог net P&L тикера из файла весов. Возвращает строку вида '+0.30%'."""
    path = _RESULTS_PATH.parent / f"best_threshold_{ticker}_{ml_settings.model_version}.json"
    try:
        with open(path) as f:
            thr = json.load(f)["threshold"]
        sign = "+" if thr >= 0 else ""
        return f"{sign}{thr * 100:.3f}%"
    except Exception:
        return "—"


@router.callback_query(lambda c: c.data == "trading:ml_status")
async def cb_ml_status(callback: CallbackQuery) -> None:
    """Показать статус ML-моделей: дата обучения, Spearman и порог уверенности по тикерам."""
    await callback.answer("⏳ Загружаю данные моделей...")

    if not _RESULTS_PATH.exists():
        await callback.message.edit_text(
            "🧠 <b>ML модели</b>\n\n❌ Файл метрик не найден. Запустите обучение.",
            reply_markup=back_to_trading(),
            parse_mode="HTML",
        )
        return

    try:
        with open(_RESULTS_PATH) as f:
            data = json.load(f)

        trained_at = datetime.fromisoformat(data["trained_at"])
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
        trained_msk = trained_at.astimezone(_RETRAIN_TZ)
        force_tune: bool = data["force_tune"]
        # Поддержка форматов: spearman_scores (новый) → sortino_scores → f1_scores (старые)
        spearman_scores: dict[str, float] = (
            data.get("spearman_scores")
            or data.get("sortino_scores")
            or data.get("f1_scores", {})
        )
        failed: list[str] = data.get("failed", [])

        # Загружаем предыдущие результаты для сравнения дельты
        prev_path = _RESULTS_PATH.parent / "last_results_prev.json"
        prev_scores: dict[str, float] = {}
        if prev_path.exists():
            try:
                with open(prev_path) as pf:
                    prev_data = json.load(pf)
                prev_scores = (
                    prev_data.get("spearman_scores")
                    or prev_data.get("sortino_scores")
                    or prev_data.get("f1_scores", {})
                )
            except Exception:
                pass

        avg = sum(spearman_scores.values()) / len(spearman_scores) if spearman_scores else 0.0
        prev_avg = sum(prev_scores.values()) / len(prev_scores) if prev_scores else None

        def _delta(ticker: str, cur: float) -> str:
            if ticker not in prev_scores:
                return ""
            d = cur - prev_scores[ticker]
            if abs(d) < 0.0001:
                return " (=)"
            return f" ({d:+.4f})"

        # Строки: тикер | Spearman | порог P&L | дельта
        score_lines = "\n".join(
            f"  {ticker:<6} {score:+.4f}  thr={_load_ticker_threshold(ticker)}{_delta(ticker, score)}"
            for ticker, score in sorted(spearman_scores.items())
        )
        avg_delta = ""
        if prev_avg is not None:
            d = avg - prev_avg
            avg_delta = f" ({d:+.4f})" if abs(d) >= 0.0001 else " (=)"
        failed_str = f"\n\n⚠️ Ошибки: {', '.join(failed)}" if failed else ""

        text = (
            "🧠 <b>ML модели</b>\n\n"
            f"Последнее обучение:  <b>{trained_msk.strftime('%d.%m.%Y %H:%M')} МСК</b>\n"
            f"Ночное сегодня:      <b>{_ml_nightly_status(trained_msk, force_tune)}</b>\n"
            f"Optuna (force_tune): <b>{'Да' if force_tune else 'Нет'}</b>\n\n"
            f"<b>Spearman / порог P&L по тикерам:</b>\n"
            f"<code>{score_lines}\n{'─'*32}\n  {'Среднее':<6} {avg:+.4f}{avg_delta}</code>"
            f"{failed_str}"
        )
    except Exception as e:
        logger.error("Ошибка загрузки метрик ML", error=str(e))
        text = "❌ Не удалось загрузить данные моделей."

    await callback.message.edit_text(
        text, reply_markup=back_to_trading(), parse_mode="HTML"
    )
