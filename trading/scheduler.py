"""
Главный цикл автоматической торговли.

Алгоритм одного тика (_tick):
    1. Загружаем все открытые позиции из БД
    2. Получаем текущие цены инструментов
    3. Получаем дивидендные корректировки через Tinkoff API (кешируются 24ч)
    4. Проверяем стоп-лосс и тейк-профит для каждой открытой позиции.
       В экс-дивидендную дату SL-порог понижается на размер дивиденда,
       чтобы предсказуемый гэп не вызвал ложное срабатывание.
    5. Получаем ML-сигналы для всех тикеров
    6. SELL-сигнал: проверяем рентабельность; закрываем только если чистый PnL > 0
       (SL/TP всегда исполняются — это управление риском, не прибылью)
    7. BUY-сигнал: открываем позицию если нет открытой и confidence >= threshold

Безопасность:
    - TRADING_ENABLED=false → тик логируется, но ордера не выставляются
    - Все исключения перехватываются — цикл не прерывается
    - Максимум TRADING_MAX_POSITIONS одновременно открытых позиций
"""
import asyncio
import json
import math
from datetime import time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

_WEIGHTS_DIR = Path(__file__).parent.parent / "ml" / "weights"


def _ticker_threshold(ticker: str) -> float:
    """Загрузить per-ticker порог уверенности из best_threshold_{ticker}_{version}.json.

    При отсутствии файла возвращает глобальный TRADING_CONFIDENCE_THRESHOLD.
    """
    from config.settings import ml_settings
    path = _WEIGHTS_DIR / f"best_threshold_{ticker}_{ml_settings.model_version}.json"
    try:
        with open(path) as f:
            return json.load(f)["threshold"]
    except Exception:
        return trading_settings.confidence_threshold


def _apply_regime_filter(lots: int, regime: int) -> int:
    """
    Скорректировать количество лотов с учётом рыночного режима.

    Режимы рынка (из predict_signal / compute_features):
        +1 = аптренд  → лоты не изменяются.
        -1 = даунтренд → BUY всегда блокируется (lots = 0).
         0 = флет     → поведение зависит от TRADING_REGIME_FILTER_MODE:
             "soft": лоты × TRADING_REGIME_FLAT_MULTIPLIER (по умолчанию 0.5).
             "hard": BUY блокируется полностью (lots = 0).

    При TRADING_REGIME_FILTER_ENABLED=false возвращает lots без изменений
    (backward compatibility — фильтр отключён).

    Аргументы:
        lots:   расчётное количество лотов (>= 0).
        regime: режим рынка (-1, 0 или +1).

    Возвращает:
        Скорректированное количество лотов (0 = BUY пропускается).
    """
    ts = trading_settings
    if not ts.regime_filter_enabled:
        return lots
    if regime == -1:
        # Даунтренд: BUY заблокирован в любом режиме фильтра
        return 0
    if regime == 0:
        if ts.regime_filter_mode == "hard":
            # Жёсткий режим: флет = блокировка
            return 0
        # Мягкий режим: уменьшить лоты пропорционально множителю
        return max(0, int(lots * ts.regime_flat_lots_multiplier))
    # regime == +1 (аптренд): без ограничений
    return lots


def _compute_dynamic_sltp(atr_ratio: float) -> tuple[float, float]:
    """
    Вычислить динамические SL/TP на основе ATR-волатильности последнего бара.

    Алгоритм:
        sl_pct = clamp(atr_ratio × ATR_SL_MULTIPLIER, min_sl, max_sl)
        tp_pct = sl_pct × ATR_RISK_REWARD_RATIO

    При TRADING_DYNAMIC_SLTP_ENABLED=false или некорректном atr_ratio (0, NaN, inf)
    возвращает фиксированные значения TRADING_STOP_LOSS_PCT / TRADING_TAKE_PROFIT_PCT
    из настроек — полная обратная совместимость.

    Аргументы:
        atr_ratio: ATR(14) / close последнего бара из predict_signal().
                   Типичный диапазон MOEX 1h: 0.005–0.015.
                   0.0 означает «не вычислено» → возврат фиксированных значений.

    Возвращает:
        (sl_pct, tp_pct): доли от цены входа (например 0.025, 0.042).
    """
    ts = trading_settings
    if not ts.dynamic_sltp_enabled or not math.isfinite(atr_ratio) or atr_ratio <= 0.0:
        return ts.stop_loss_pct, ts.take_profit_pct

    sl = float(max(ts.atr_min_sl_pct, min(atr_ratio * ts.atr_sl_multiplier, ts.atr_max_sl_pct)))
    tp = sl * ts.atr_risk_reward_ratio
    # TP не может быть меньше самого SL — минимальное соотношение 1:1
    tp = max(tp, sl)
    return sl, tp


from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import data_settings, ml_settings, trading_settings
from db.database import get_session
from db.models import Asset, Trade
from db import trade_repo
from ml.predict import predict_all
from tinkoff.dividend_gap_stats import get_gap_protection_days_bulk
from tinkoff.dividends import get_dividend_drops_bulk
from tinkoff.instruments import get_instrument_by_ticker
from tinkoff.market_data import get_last_prices, get_min_price_increment
from tinkoff.portfolio import get_active_order_ids, get_order_state, get_rub_balance, get_stop_order_ids, post_stop_order
from t_tech.invest.schemas import StopOrderType
from t_tech.invest.schemas import OrderExecutionReportStatus, StopOrderDirection
from scripts.collect_candles import run_collection
from trading import state
from trading.executor import TradeExecutor
from trading.notifier import notify_close, notify_insufficient_balance, notify_open
from trading.position_sizing import apply_confidence_scaling, compute_lots
from trading.profitability import calculate_pnl, round_sl_to_step, round_tp_to_step
from utils.logger import logger
from utils.redis_cache import get_redis


class TradingScheduler:
    """Оркестратор автоматической торговли: запускает тики по расписанию."""

    def __init__(self) -> None:
        """Инициализировать исполнителя сделок."""
        self._executor = TradeExecutor()

    async def run(self) -> None:
        """
        Запустить бесконечный цикл автоторговли.

        Параллельно запускает:
        - основной цикл тиков каждые TRADING_INTERVAL_SECONDS
        - утренний тик в 9:50 МСК для перевыставления TP до открытия сессии
        """
        logger.info(
            "TradingScheduler запущен",
            enabled=trading_settings.enabled,
            interval_seconds=trading_settings.check_interval_seconds,
        )
        await asyncio.gather(
            self._main_loop(),
            self._morning_tp_loop(),
        )

    async def _main_loop(self) -> None:
        """Основной цикл тиков по расписанию."""
        interval = trading_settings.check_interval_seconds
        while True:
            try:
                await self._tick()
            except Exception as e:
                logger.error("Критическая ошибка тика", error=str(e), exc_info=True)
            logger.debug("Следующий тик", interval_seconds=interval)
            await asyncio.sleep(interval)

    async def _morning_tp_loop(self) -> None:
        """Ежедневно в 9:50 МСК запускает тик для перевыставления TP перед открытием сессии."""
        from datetime import datetime, timedelta
        while True:
            now = datetime.now(ZoneInfo("Europe/Moscow"))
            target = now.replace(hour=9, minute=50, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait_seconds = (target - now).total_seconds()
            logger.debug(
                "Утренний тик TP запланирован",
                at=target.strftime("%Y-%m-%d %H:%M МСК"),
                wait_seconds=int(wait_seconds),
            )
            await asyncio.sleep(wait_seconds)
            logger.info("Утренний тик: перевыставление TP (9:50 МСК)")
            try:
                await self._tick()
            except Exception as e:
                logger.error("Ошибка утреннего тика TP", error=str(e), exc_info=True)

    async def _tick(self) -> None:
        """
        Один прогон автоторговли.

        Открывает и закрывает позиции в рамках одной транзакции.
        """
        if not state.is_auto():
            logger.info("Автоторговля в ручном режиме — тик пропущен")
            return

        logger.info("Начало тика автоторговли")

        async with get_session() as session:
            # ── 1. Открытые позиции ───────────────────────────────────────────
            open_trades = await trade_repo.get_open_trades(session)

            # Карта asset_id → Trade для быстрого поиска
            open_by_asset: dict[int, Trade] = {
                t.asset_id: t for t in open_trades
            }

            # ── 2. Все активы в БД (карта figi → Asset, ticker → Asset) ──────
            result = await session.execute(select(Asset))
            all_assets = list(result.scalars().all())
            figi_to_asset: dict[str, Asset] = {a.figi: a for a in all_assets}
            ticker_to_asset: dict[str, Asset] = {a.ticker: a for a in all_assets}
            asset_id_to_figi: dict[int, str] = {a.id: a.figi for a in all_assets}

            # ── 3. Текущие цены для открытых позиций ─────────────────────────
            if open_trades:
                open_figis_list = [
                    asset_id_to_figi[t.asset_id]
                    for t in open_trades
                    if t.asset_id in asset_id_to_figi
                ]
                prices = await get_last_prices(open_figis_list) if open_figis_list else {}

                # ── 4. Дивидендные корректировки SL ──────────────────────────
                # В экс-дивидендную дату акция падает на размер дивиденда.
                # Чтобы это предсказуемое падение не вызвало ложное срабатывание
                # стоп-лосса, понижаем эффективный порог SL на величину дивиденда.
                # Приоритет: ручные переопределения (.env) → БД (Asset.dividend_gap_days)
                #            → авто-вычисление из истории → глобальный дефолт.
                dividend_drops: dict[str, Decimal] = {}
                if trading_settings.dividend_protection_days > 0:
                    try:
                        manual_overrides = trading_settings.dividend_override

                        # Активы с открытыми позициями (для передачи в gap_stats)
                        open_assets = [
                            figi_to_asset[figi]
                            for figi in open_figis_list
                            if figi in figi_to_asset
                        ]

                        # Ручные переопределения из .env — наивысший приоритет
                        manual_per_figi: dict[str, int] = {
                            a.figi: manual_overrides[a.ticker]
                            for a in open_assets
                            if a.ticker in manual_overrides
                        }
                        if manual_per_figi:
                            logger.info(
                                "Дивидендная защита: ручные переопределения применены",
                                overrides={
                                    figi_to_asset[f].ticker: d
                                    for f, d in manual_per_figi.items()
                                },
                            )

                        # Для остальных — читаем из БД (или пересчитываем если устарело)
                        auto_assets = [a for a in open_assets if a.figi not in manual_per_figi]
                        auto_per_figi = await get_gap_protection_days_bulk(
                            auto_assets,
                            session=session,
                            fallback=trading_settings.dividend_protection_days,
                        ) if auto_assets else {}

                        per_figi_days = {**auto_per_figi, **manual_per_figi}

                        dividend_drops = await get_dividend_drops_bulk(
                            open_figis_list,
                            per_figi_days=per_figi_days,
                        )
                    except Exception as e:
                        logger.warning("Ошибка получения дивидендных данных", error=str(e))

                # ── 5. Проверяем SL/TP (всегда, независимо от рентабельности) ─

                # Получаем ID активных ордеров один раз для всех позиций.
                # stop_ids содержит ОБА типа: TP-стоп-ордера и SL-стоп-ордера.
                # limit_orders_fetched нужен только для backward compat с legacy tp_order_id.
                active_stop_ids: set[str] = set()
                active_order_ids: set[str] = set()
                stop_orders_fetched: bool = False
                limit_orders_fetched: bool = False
                try:
                    active_stop_ids = await get_stop_order_ids()
                    stop_orders_fetched = True
                except Exception as e:
                    logger.warning("Не удалось получить список стоп-ордеров", error=str(e))

                # Лимитные ордера нужны только для старых позиций с tp_order_id (legacy)
                has_legacy_tp = any(t.tp_order_id for t in open_trades)
                if has_legacy_tp:
                    try:
                        active_order_ids = await get_active_order_ids()
                        limit_orders_fetched = True
                    except Exception as e:
                        logger.warning("Не удалось получить список лимитных ордеров", error=str(e))

                for trade in open_trades:
                    figi = asset_id_to_figi.get(trade.asset_id)
                    if not figi:
                        continue
                    current_price = prices.get(figi, Decimal("0"))
                    if current_price <= 0:
                        continue

                    asset = figi_to_asset.get(figi)
                    if not asset:
                        continue

                    close_reason, tp_fill_price, do_continue = await _check_trade_orders(
                        session=session,
                        trade=trade,
                        asset=asset,
                        figi=figi,
                        current_price=current_price,
                        active_stop_ids=active_stop_ids,
                        active_order_ids=active_order_ids,
                        stop_orders_fetched=stop_orders_fetched,
                        limit_orders_fetched=limit_orders_fetched,
                    )
                    if do_continue:
                        continue

                        # Фолбэк: не удалось выставить ордера — сравниваем цену
                        dividend_adj = dividend_drops.get(figi, Decimal("0"))
                        effective_sl = max(
                            trade.stop_loss_price - dividend_adj, Decimal("0.01")
                        )
                        if dividend_adj > 0:
                            logger.info(
                                "Дивидендная защита SL применена",
                                ticker=asset.ticker,
                                original_sl=str(trade.stop_loss_price),
                                effective_sl=str(effective_sl),
                                dividend_adj=str(dividend_adj),
                            )
                        if current_price <= effective_sl:
                            close_reason = "STOP_LOSS"
                        elif current_price >= trade.take_profit_price:
                            close_reason = "TAKE_PROFIT"

                    if close_reason:
                        # При TP используем фактическую цену исполнения биржевого ордера
                        exit_price_for_pnl = (
                            tp_fill_price
                            if close_reason == "TAKE_PROFIT" and tp_fill_price
                            else current_price
                        )
                        lot_size = getattr(trade, "lot_size", 1) or 1
                        breakdown = calculate_pnl(
                            entry_price=trade.entry_price,
                            exit_price=exit_price_for_pnl,
                            lots=trade.lots,
                            lot_size=lot_size,
                        )
                        logger.info(
                            "Срабатывание SL/TP",
                            ticker=asset.ticker,
                            reason=close_reason,
                            exit_price=str(exit_price_for_pnl),
                            gross_pnl=str(breakdown.gross_pnl),
                            net_pnl=str(breakdown.net_pnl),
                        )
                        try:
                            closed_trade = await self._executor.close_position(
                                session=session,
                                trade=trade,
                                asset=asset,
                                instrument_uid=figi,
                                current_price=exit_price_for_pnl,
                                reason=close_reason,
                            )
                        except Exception as close_err:
                            logger.error(
                                "Ошибка закрытия позиции — сделка остаётся открытой в БД",
                                ticker=asset.ticker,
                                reason=close_reason,
                                error=str(close_err),
                            )
                            continue
                        open_by_asset.pop(trade.asset_id, None)
                        # Пересчитываем breakdown от фактической цены исполнения
                        actual_exit = closed_trade.exit_price or exit_price_for_pnl
                        breakdown = calculate_pnl(
                            entry_price=trade.entry_price,
                            exit_price=actual_exit,
                            lots=trade.lots,
                            lot_size=lot_size,
                        )
                        await notify_close(
                            ticker=asset.ticker,
                            entry_price=trade.entry_price,
                            exit_price=actual_exit,
                            reason=close_reason,
                            net_pnl=closed_trade.pnl or breakdown.net_pnl,
                            gross_pnl=breakdown.gross_pnl,
                            commission=breakdown.buy_commission + breakdown.sell_commission,
                            tax=breakdown.tax,
                        )
            else:
                prices = {}

            # ── 6. Обновляем свечи из Tinkoff API перед инференсом ───────────
            await _update_candles()

            # ── 7. ML-сигналы ─────────────────────────────────────────────────
            signals = await predict_all()
            logger.info("Сигналы получены", count=len(signals))

            open_count = len(open_by_asset)
            max_positions = trading_settings.max_open_positions

            # Получаем баланс один раз перед циклом BUY-сигналов.
            # Обновляем локально после каждой успешной покупки, чтобы
            # не отправлять заявки когда средства уже зарезервированы.
            try:
                rub_balance = await get_rub_balance()
            except Exception as e:
                logger.warning("Не удалось получить баланс", error=str(e))
                rub_balance = Decimal("0")

            for sig in signals:
                ticker: str = sig.get("ticker", "")
                signal_type: str = sig.get("signal", "HOLD")
                confidence: float = sig.get("confidence", 0.0)

                asset = ticker_to_asset.get(ticker)
                if not asset:
                    logger.debug("Актив не найден в БД", ticker=ticker)
                    continue

                figi = asset.figi

                # ── 7. SELL-сигнал: закрываем только если прибыльно ──────────
                if signal_type == "SELL" and asset.id in open_by_asset:
                    trade = open_by_asset[asset.id]
                    current_price = prices.get(figi) or await _fetch_price(figi)
                    if current_price <= 0:
                        continue

                    lot_size = getattr(trade, "lot_size", 1) or 1
                    breakdown = calculate_pnl(
                        entry_price=trade.entry_price,
                        exit_price=current_price,
                        lots=trade.lots,
                        lot_size=lot_size,
                    )

                    if not breakdown.is_profitable:
                        # Продажа убыточна после комиссий/налога — ждём лучшей цены
                        logger.info(
                            "SELL-сигнал: продажа нерентабельна, позиция удерживается",
                            ticker=ticker,
                            gross_pnl=str(breakdown.gross_pnl),
                            net_pnl=str(breakdown.net_pnl),
                            commission=str(
                                breakdown.buy_commission + breakdown.sell_commission
                            ),
                        )
                        continue

                    closed_trade = await self._executor.close_position(
                        session=session,
                        trade=trade,
                        asset=asset,
                        instrument_uid=figi,
                        current_price=current_price,
                        reason="SELL_SIGNAL",
                    )
                    open_by_asset.pop(asset.id, None)
                    open_count -= 1
                    # Пересчитываем breakdown от фактической цены исполнения
                    actual_exit = closed_trade.exit_price or current_price
                    breakdown = calculate_pnl(
                        entry_price=trade.entry_price,
                        exit_price=actual_exit,
                        lots=trade.lots,
                        lot_size=lot_size,
                    )
                    await notify_close(
                        ticker=ticker,
                        entry_price=trade.entry_price,
                        exit_price=actual_exit,
                        reason="SELL_SIGNAL",
                        net_pnl=closed_trade.pnl or breakdown.net_pnl,
                        gross_pnl=breakdown.gross_pnl,
                        commission=breakdown.buy_commission + breakdown.sell_commission,
                        tax=breakdown.tax,
                    )

                # ── 8. BUY-сигнал: открываем позицию ─────────────────────────
                elif signal_type == "BUY":
                    if asset.id in open_by_asset:
                        logger.debug("Позиция уже открыта", ticker=ticker)
                        continue

                    # confidence = предсказанный net P&L (регрессор).
                    # Сравниваем с per-ticker порогом входа из best_threshold_*.json
                    # (диапазон [0.0, 0.02]), оптимизированным на holdout по Sortino.
                    buy_confidence: float = sig.get("confidence", 0.0)
                    ticker_threshold = _ticker_threshold(ticker)
                    if buy_confidence < ticker_threshold:
                        logger.debug(
                            "Уверенность ниже порога",
                            ticker=ticker,
                            buy_proba=buy_proba,
                            threshold=ticker_threshold,
                        )
                        continue

                    # Фильтр подтверждения объёмом: открываем только если объём
                    # последнего бара превышает SMA_20(объём) на заданный коэффициент.
                    # Повышенный объём подтверждает направленное движение,
                    # снижает число ложных сигналов в боковике.
                    volume_ratio: float = sig.get("volume_ratio", 1.0)
                    if volume_ratio < trading_settings.volume_min_ratio:
                        logger.debug(
                            "BUY-сигнал пропущен: объём ниже порога подтверждения",
                            ticker=ticker,
                            volume_ratio=volume_ratio,
                            min_ratio=trading_settings.volume_min_ratio,
                        )
                        continue

                    if open_count >= max_positions:
                        logger.info(
                            "Достигнут лимит открытых позиций",
                            open=open_count,
                            max=max_positions,
                        )
                        break

                    current_price = await _fetch_price(figi)
                    if current_price <= 0:
                        logger.warning("Не удалось получить цену", ticker=ticker)
                        continue

                    # Получаем размер лота через API (с graceful degradation)
                    lot_size = await _get_lot_size(ticker)

                    # Вычисляем динамические SL/TP на основе ATR-волатильности сигнала.
                    # Выполняем ДО расчёта лотов — sl_pct нужен для compute_lots.
                    # При TRADING_DYNAMIC_SLTP_ENABLED=false возвращает фиксированные значения.
                    _atr_ratio: float = sig.get("atr_ratio", 0.0)
                    _sl_pct, _tp_pct = _compute_dynamic_sltp(_atr_ratio)

                    # Рассчитываем число лотов по методу Fixed Fractional Risk.
                    # При TRADING_POSITION_SIZING='fixed_lots' — возвращает lots_per_ticker.
                    lots = compute_lots(rub_balance, current_price, lot_size, _sl_pct)

                    # Фильтр рыночного режима: корректируем лоты с учётом тренда.
                    # В даунтренде (-1) BUY всегда блокируется.
                    # В флете (0) при "soft" лоты умножаются на TRADING_REGIME_FLAT_MULTIPLIER.
                    # При TRADING_REGIME_FILTER_ENABLED=false — фильтр не применяется.
                    _regime: int = sig.get("market_regime", 1)
                    lots = _apply_regime_filter(lots, _regime)

                    # Масштабирование по уверенности ML-сигнала (третий слой position sizing).
                    # При TRADING_CONFIDENCE_SCALING_ENABLED=false — возвращает lots без изменений.
                    # При lots=0 (блокировка режим-фильтра) — не вмешивается.
                    lots = apply_confidence_scaling(lots, buy_confidence)

                    if lots == 0:
                        logger.info(
                            "BUY-сигнал пропущен: рыночный режим не допускает открытие позиции",
                            ticker=ticker,
                            market_regime=_regime,
                            regime_filter_enabled=trading_settings.regime_filter_enabled,
                            regime_filter_mode=trading_settings.regime_filter_mode,
                        )
                        continue

                    needed = current_price * Decimal(lots) * Decimal(lot_size)

                    # Проверяем баланс перед выставлением ордера
                    if rub_balance < needed:
                        logger.warning(
                            "BUY-сигнал пропущен: недостаточно средств",
                            ticker=ticker,
                            needed=str(needed),
                            available=str(rub_balance),
                        )
                        await notify_insufficient_balance(ticker, needed, rub_balance)
                        continue

                    new_trade = await self._executor.open_position(
                        session=session,
                        asset=asset,
                        instrument_uid=figi,
                        current_price=current_price,
                        lot_size=lot_size,
                        sl_pct=_sl_pct,
                        tp_pct=_tp_pct,
                        lots=lots,
                    )
                    if new_trade:
                        open_by_asset[asset.id] = new_trade
                        open_count += 1
                        rub_balance -= needed  # уменьшаем локально для следующих BUY в этом тике
                        await notify_open(
                            ticker=ticker,
                            price=new_trade.entry_price,
                            lots=new_trade.lots,
                            lot_size=lot_size,
                            stop_loss=new_trade.stop_loss_price,
                            take_profit=new_trade.take_profit_price,
                        )

        logger.info("Тик завершён", open_positions=open_count)


async def _fetch_price(figi: str) -> Decimal:
    """
    Получить текущую цену одного инструмента.

    Аргументы:
        figi: FIGI инструмента

    Возвращает:
        Цена или Decimal("0") при ошибке.
    """
    try:
        prices = await get_last_prices([figi])
        return prices.get(figi, Decimal("0"))
    except Exception as e:
        logger.error("Ошибка получения цены", figi=figi, error=str(e))
        return Decimal("0")


def _is_moex_session_open() -> bool:
    """
    Проверить, открыта ли торговая сессия МОЕХ прямо сейчас.

    Лимитные ордера (TP) принимаются только во время сессии:
    - Предторговый: 09:45 – 10:00 МСК (лимитные T0 принимаются)
    - Основная:     10:00 – 18:50 МСК
    - Вечерняя:     19:05 – 23:50 МСК

    Стоп-ордера (SL) принимаются круглосуточно — эта проверка для них не нужна.
    """
    from datetime import datetime
    now_msk = datetime.now(ZoneInfo("Europe/Moscow")).time()
    pretrade   = time(9, 45)  <= now_msk < time(10, 0)
    main_open  = time(10, 0)  <= now_msk < time(18, 50)
    evening_open = time(19, 5) <= now_msk < time(23, 50)
    return pretrade or main_open or evening_open


async def _update_candles() -> None:
    """
    Инкрементально обновить свечи из Tinkoff API и сбросить Redis-кеш.

    Вызывается перед каждым ML-инференсом, чтобы модель работала
    на актуальных данных, а не только на ночном снимке.
    При ошибке — логирует и продолжает тик без обновления.
    """
    try:
        await run_collection(pause_seconds=trading_settings.candle_update_pause_seconds)
    except Exception as e:
        logger.warning("Не удалось обновить свечи перед инференсом", error=str(e))
        return

    # Инвалидируем Redis-кеш свечей и сигналов — следующий predict_signal()
    # прочитает обновлённые данные из БД вместо устаревшего кеша.
    try:
        redis = await get_redis()
        if redis is not None:
            interval = data_settings.candle_interval
            version = ml_settings.model_version
            for ticker in data_settings.tickers + ["USDRUB"]:
                await redis.delete(f"candles:{ticker}:{interval}")
            for ticker in data_settings.tickers:
                await redis.delete(f"signal:{ticker}:{version}")
            logger.debug("Redis-кеш свечей и сигналов инвалидирован")
    except Exception as e:
        logger.warning("Ошибка инвалидации Redis-кеша после обновления свечей", error=str(e))


async def _check_trade_orders(
    session: "AsyncSession",
    trade: "Trade",
    asset: "Asset",
    figi: str,
    current_price: Decimal,
    active_stop_ids: set[str],
    active_order_ids: set[str],
    stop_orders_fetched: bool,
    limit_orders_fetched: bool,
) -> tuple[str | None, Decimal | None, bool]:
    """
    Проверить статус биржевых ордеров для одной открытой позиции.

    Обрабатывает TP стоп-ордер, legacy TP лимитный ордер, SL стоп-ордер.
    При необходимости перевыставляет пропавшие ордера.

    Аргументы:
        session:             AsyncSession для записи в БД
        trade:               открытая сделка
        asset:               актив сделки
        figi:                FIGI инструмента
        current_price:       текущая рыночная цена
        active_stop_ids:     множество ID активных стоп-ордеров
        active_order_ids:    множество ID активных лимитных ордеров (legacy)
        stop_orders_fetched: успешно ли получены стоп-ордера
        limit_orders_fetched: успешно ли получены лимитные ордера

    Возвращает:
        Кортеж (close_reason, tp_fill_price, do_continue):
            close_reason  — "TAKE_PROFIT" / "STOP_LOSS" / None
            tp_fill_price — цена исполнения TP или None
            do_continue   — True если нужно перейти к следующему trade без закрытия
    """
    close_reason: str | None = None
    tp_fill_price: Decimal | None = None

    # ── Проверка TP стоп-ордера (новый формат) ───────────────────────────────
    if trade.tp_stop_order_id:
        tp_stop_gone = stop_orders_fetched and trade.tp_stop_order_id not in active_stop_ids
        if tp_stop_gone:
            # Ценовой признак: стоп-ордера не имеют API-статуса исполнения
            if current_price >= trade.take_profit_price:
                close_reason = "TAKE_PROFIT"
                tp_fill_price = trade.take_profit_price
            else:
                logger.warning(
                    "TP стоп-ордер исчез, цена ниже TP — перевыставляем",
                    ticker=asset.ticker,
                    tp_stop_order_id=trade.tp_stop_order_id,
                    current_price=str(current_price),
                    take_profit_price=str(trade.take_profit_price),
                )
                try:
                    price_step = await get_min_price_increment(figi)
                    tp_rounded = round_tp_to_step(trade.take_profit_price, price_step)
                    new_tp_stop_id = await post_stop_order(
                        instrument_id=figi,
                        quantity=trade.lots,
                        stop_price=tp_rounded,
                        direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                        stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
                    )
                    trade.tp_stop_order_id = new_tp_stop_id
                    await trade_repo.update_trade(session, trade)
                    logger.info("TP стоп-ордер перевыставлен", ticker=asset.ticker,
                                stop_order_id=new_tp_stop_id, price=str(tp_rounded))
                except Exception as re_e:
                    logger.error("Не удалось перевыставить TP стоп-ордер",
                                 ticker=asset.ticker, error=str(re_e))

    # ── Проверка TP лимитного ордера (legacy) ────────────────────────────────
    elif trade.tp_order_id:
        tp_gone = limit_orders_fetched and trade.tp_order_id not in active_order_ids
        if tp_gone:
            try:
                tp_status, tp_fill_price = await get_order_state(trade.tp_order_id)
                if tp_status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL:
                    close_reason = "TAKE_PROFIT"
            except Exception:
                pass  # ордер заархивирован — считаем истёкшим
            if close_reason != "TAKE_PROFIT":
                if not _is_moex_session_open():
                    trade.tp_order_id = None
                    await trade_repo.update_trade(session, trade)
                    logger.debug("Legacy TP лимит-ордер исчез, биржа закрыта — ID очищен",
                                 ticker=asset.ticker)
                else:
                    logger.warning("Legacy TP лимит-ордер исчез — перевыставляем как стоп-TP",
                                   ticker=asset.ticker, order_id=trade.tp_order_id)
                    try:
                        price_step = await get_min_price_increment(figi)
                        tp_rounded = round_tp_to_step(trade.take_profit_price, price_step)
                        new_tp_stop_id = await post_stop_order(
                            instrument_id=figi,
                            quantity=trade.lots,
                            stop_price=tp_rounded,
                            direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                            stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
                        )
                        trade.tp_order_id = None
                        trade.tp_stop_order_id = new_tp_stop_id
                        await trade_repo.update_trade(session, trade)
                        logger.info("Legacy TP мигрирован на стоп-TP", ticker=asset.ticker,
                                    stop_order_id=new_tp_stop_id, price=str(tp_rounded))
                    except Exception as re_e:
                        logger.error("Не удалось перевыставить TP как стоп-ордер",
                                     ticker=asset.ticker, error=str(re_e))

    # ── Если SL не был выставлен при открытии (сбой API) ─────────────────────
    if close_reason is None and not trade.sl_stop_order_id:
        if current_price <= trade.stop_loss_price:
            close_reason = "STOP_LOSS"
        else:
            try:
                price_step = await get_min_price_increment(figi)
                sl_rounded = round_sl_to_step(trade.stop_loss_price, price_step)
                new_sl_id = await post_stop_order(
                    instrument_id=figi,
                    quantity=trade.lots,
                    stop_price=sl_rounded,
                    direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                    stop_order_type=StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
                )
                trade.sl_stop_order_id = new_sl_id
                await trade_repo.update_trade(session, trade)
                logger.info("SL стоп-ордер выставлен (не был создан при открытии)",
                            ticker=asset.ticker, stop_order_id=new_sl_id)
            except Exception as re_e:
                logger.error("Не удалось выставить SL стоп-ордер",
                             ticker=asset.ticker, error=str(re_e))

    # ── Проверяем исполнение SL стоп-ордера ───────────────────────────────────
    if (close_reason is None and trade.sl_stop_order_id
            and stop_orders_fetched and trade.sl_stop_order_id not in active_stop_ids):
        if current_price <= trade.stop_loss_price:
            close_reason = "STOP_LOSS"
        else:
            logger.warning(
                "SL стоп-ордер исчез, но цена выше SL — перевыставляем",
                ticker=asset.ticker,
                current_price=str(current_price),
                stop_loss_price=str(trade.stop_loss_price),
                sl_stop_order_id=trade.sl_stop_order_id,
            )
            try:
                price_step = await get_min_price_increment(figi)
                sl_rounded = round_sl_to_step(trade.stop_loss_price, price_step)
                new_sl_id = await post_stop_order(
                    instrument_id=figi,
                    quantity=trade.lots,
                    stop_price=sl_rounded,
                    direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                    stop_order_type=StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
                )
                trade.sl_stop_order_id = new_sl_id
                await trade_repo.update_trade(session, trade)
                logger.info("SL стоп-ордер перевыставлен", ticker=asset.ticker,
                            stop_order_id=new_sl_id)
            except Exception as re_e:
                logger.error("Не удалось перевыставить SL стоп-ордер",
                             ticker=asset.ticker, error=str(re_e))

    # ── Нет ни TP ни SL — выставляем оба ─────────────────────────────────────
    if close_reason is None and not trade.tp_stop_order_id and not trade.tp_order_id and not trade.sl_stop_order_id:
        try:
            price_step = await get_min_price_increment(figi)
            tp_rounded = round_tp_to_step(trade.take_profit_price, price_step)
            new_tp_stop_id = await post_stop_order(
                instrument_id=figi,
                quantity=trade.lots,
                stop_price=tp_rounded,
                direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
            )
            trade.tp_stop_order_id = new_tp_stop_id
            logger.info("TP стоп-ордер выставлен (не был создан при открытии)",
                        ticker=asset.ticker, stop_order_id=new_tp_stop_id, price=str(tp_rounded))
        except Exception as re_e:
            logger.error("Не удалось выставить TP стоп-ордер", ticker=asset.ticker, error=str(re_e))
        if not trade.sl_stop_order_id:
            try:
                price_step = await get_min_price_increment(figi)
                sl_rounded = round_sl_to_step(trade.stop_loss_price, price_step)
                new_sl_id = await post_stop_order(
                    instrument_id=figi,
                    quantity=trade.lots,
                    stop_price=sl_rounded,
                    direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                    stop_order_type=StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
                )
                trade.sl_stop_order_id = new_sl_id
                logger.info("SL стоп-ордер выставлен (не был создан при открытии)",
                            ticker=asset.ticker, stop_order_id=new_sl_id)
            except Exception as re_e:
                logger.error("Не удалось выставить SL стоп-ордер", ticker=asset.ticker, error=str(re_e))
        if trade.tp_stop_order_id or trade.sl_stop_order_id:
            await trade_repo.update_trade(session, trade)
            return None, None, True  # do_continue=True

        return None, None, False  # нет ордеров и не удалось выставить → фолбэк

    return close_reason, tp_fill_price, False


async def _get_lot_size(ticker: str) -> int:
    """
    Получить количество бумаг в 1 лоте для тикера.

    Использует API Tinkoff; при ошибке возвращает 1 (безопасный дефолт).

    Аргументы:
        ticker: тикер инструмента

    Возвращает:
        Размер лота (>= 1).
    """
    try:
        info = await get_instrument_by_ticker(ticker)
        return info.lot if info and info.lot > 0 else 1
    except Exception as e:
        logger.warning("Не удалось получить lot_size", ticker=ticker, error=str(e))
        return 1
