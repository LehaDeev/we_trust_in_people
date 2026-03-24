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
from decimal import Decimal
from zoneinfo import ZoneInfo

from trading.order_monitor import _check_trade_orders
from trading.scheduler_helpers import (
    _apply_regime_filter,
    _compute_dynamic_sltp,
    _fetch_price,
    _get_lot_size,
    _ticker_threshold,
    _update_candles,
)


from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import trading_settings
from db.database import get_session
from db.models import Asset, Trade
from db import trade_repo
from ml.predict import predict_all
from tinkoff.dividend_gap_stats import get_gap_protection_days_bulk
from tinkoff.dividends import get_dividend_drops_bulk
from tinkoff.market_data import get_last_prices
from tinkoff.portfolio import get_active_order_ids, get_rub_balance, get_stop_order_ids
from trading import state
from trading.executor import TradeExecutor
from trading.notifier import notify_close, notify_insufficient_balance, notify_open
from trading.position_sizing import apply_confidence_scaling, compute_lots
from trading.profitability import calculate_pnl
from utils.logger import logger


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
                            confidence=buy_confidence,
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


