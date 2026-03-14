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
from tinkoff.portfolio import get_active_order_ids, get_order_state, get_rub_balance, get_stop_order_ids, post_limit_order, post_stop_order
from t_tech.invest.schemas import OrderDirection, OrderExecutionReportStatus, StopOrderDirection
from scripts.collect_candles import run_collection
from trading import state
from trading.executor import TradeExecutor
from trading.notifier import notify_close, notify_insufficient_balance, notify_open
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

        Каждые TRADING_INTERVAL_SECONDS выполняет один тик.
        Все исключения перехватываются — цикл не останавливается.
        """
        interval = trading_settings.check_interval_seconds
        logger.info(
            "TradingScheduler запущен",
            enabled=trading_settings.enabled,
            interval_seconds=interval,
        )

        while True:
            try:
                await self._tick()
            except Exception as e:
                logger.error("Критическая ошибка тика", error=str(e), exc_info=True)

            logger.debug("Следующий тик", interval_seconds=interval)
            await asyncio.sleep(interval)

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

                # Получаем ID активных ордеров один раз для всех позиций
                active_stop_ids: set[str] = set()
                active_order_ids: set[str] = set()
                stop_orders_fetched: bool = False
                limit_orders_fetched: bool = False
                try:
                    active_stop_ids = await get_stop_order_ids()
                    stop_orders_fetched = True
                except Exception as e:
                    logger.warning("Не удалось получить список стоп-ордеров", error=str(e))
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

                    close_reason: str | None = None
                    tp_fill_price: Decimal | None = None

                    # Приоритет: проверка биржевых ордеров (если были выставлены)
                    if trade.tp_order_id:
                        # Если список активных ордеров получен и TP в нём отсутствует —
                        # ордер либо исполнился, либо истёк (конец сессии)
                        tp_gone = limit_orders_fetched and trade.tp_order_id not in active_order_ids
                        if tp_gone:
                            try:
                                tp_status, tp_fill_price = await get_order_state(trade.tp_order_id)
                                if tp_status == OrderExecutionReportStatus.EXECUTION_REPORT_STATUS_FILL:
                                    close_reason = "TAKE_PROFIT"
                            except Exception:
                                pass  # ордер заархивирован — считаем истёкшим

                            # Не FILL (истёк или заархивирован) — перевыставляем TP если сессия открыта
                            if close_reason != "TAKE_PROFIT":
                                if not _is_moex_session_open():
                                    logger.debug(
                                        "TP ордер исчез, но биржа закрыта — перевыставим позже",
                                        ticker=asset.ticker,
                                    )
                                else:
                                    logger.warning(
                                        "TP ордер исчез (истёк или отменён), перевыставляем",
                                        ticker=asset.ticker,
                                        order_id=trade.tp_order_id,
                                    )
                                    try:
                                        price_step = await get_min_price_increment(figi)
                                        tp_rounded = round_tp_to_step(trade.take_profit_price, price_step)
                                        tp_resp = await post_limit_order(
                                            instrument_id=figi,
                                            quantity=trade.lots,
                                            price=tp_rounded,
                                            direction=OrderDirection.ORDER_DIRECTION_SELL,
                                        )
                                        trade.tp_order_id = tp_resp.order_id
                                        await trade_repo.update_trade(session, trade)
                                        logger.info(
                                            "TP ордер перевыставлен",
                                            ticker=asset.ticker,
                                            order_id=tp_resp.order_id,
                                            price=str(tp_rounded),
                                        )
                                    except Exception as re_e:
                                        logger.error(
                                            "Не удалось перевыставить TP ордер",
                                            ticker=asset.ticker,
                                            error=str(re_e),
                                        )

                        # Если SL ордер не был выставлен при открытии (сбой API) — выставляем сейчас
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
                                    )
                                    trade.sl_stop_order_id = new_sl_id
                                    await trade_repo.update_trade(session, trade)
                                    logger.info(
                                        "SL стоп-ордер выставлен (не был создан при открытии)",
                                        ticker=asset.ticker,
                                        stop_order_id=new_sl_id,
                                    )
                                except Exception as re_e:
                                    logger.error(
                                        "Не удалось выставить SL стоп-ордер",
                                        ticker=asset.ticker,
                                        error=str(re_e),
                                    )

                        # Проверяем исполнение стоп-ордера SL (пропал из активных → исполнился)
                        if close_reason is None and trade.sl_stop_order_id and stop_orders_fetched and trade.sl_stop_order_id not in active_stop_ids:
                            # Дополнительная проверка: если цена выше SL — ордер был
                            # отменён биржей (не исполнился), а не сработал
                            if current_price <= trade.stop_loss_price:
                                close_reason = "STOP_LOSS"
                            else:
                                # SL ордер исчез (истёк или отменён биржей) — перевыставляем
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
                                    )
                                    trade.sl_stop_order_id = new_sl_id
                                    await trade_repo.update_trade(session, trade)
                                    logger.info(
                                        "SL стоп-ордер перевыставлен",
                                        ticker=asset.ticker,
                                        stop_order_id=new_sl_id,
                                    )
                                except Exception as re_e:
                                    logger.error(
                                        "Не удалось перевыставить SL стоп-ордер",
                                        ticker=asset.ticker,
                                        error=str(re_e),
                                    )
                    else:
                        # Нет ни TP ни SL ордеров — выставляем оба на бирже
                        if _is_moex_session_open():
                            try:
                                price_step = await get_min_price_increment(figi)
                                tp_rounded = round_tp_to_step(trade.take_profit_price, price_step)
                                tp_resp = await post_limit_order(
                                    instrument_id=figi,
                                    quantity=trade.lots,
                                    price=tp_rounded,
                                    direction=OrderDirection.ORDER_DIRECTION_SELL,
                                )
                                trade.tp_order_id = tp_resp.order_id
                                logger.info(
                                    "TP ордер выставлен (не был создан при открытии)",
                                    ticker=asset.ticker,
                                    order_id=tp_resp.order_id,
                                    price=str(tp_rounded),
                                )
                            except Exception as re_e:
                                logger.error(
                                    "Не удалось выставить TP ордер",
                                    ticker=asset.ticker,
                                    error=str(re_e),
                                )
                        try:
                            price_step = await get_min_price_increment(figi)
                            sl_rounded = round_sl_to_step(trade.stop_loss_price, price_step)
                            new_sl_id = await post_stop_order(
                                instrument_id=figi,
                                quantity=trade.lots,
                                stop_price=sl_rounded,
                                direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                            )
                            trade.sl_stop_order_id = new_sl_id
                            logger.info(
                                "SL стоп-ордер выставлен (не был создан при открытии)",
                                ticker=asset.ticker,
                                stop_order_id=new_sl_id,
                            )
                        except Exception as re_e:
                            logger.error(
                                "Не удалось выставить SL стоп-ордер",
                                ticker=asset.ticker,
                                error=str(re_e),
                            )
                        if trade.tp_order_id or trade.sl_stop_order_id:
                            await trade_repo.update_trade(session, trade)
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
                        closed_trade = await self._executor.close_position(
                            session=session,
                            trade=trade,
                            asset=asset,
                            instrument_uid=figi,
                            current_price=exit_price_for_pnl,
                            reason=close_reason,
                        )
                        open_by_asset.pop(trade.asset_id, None)
                        await notify_close(
                            ticker=asset.ticker,
                            entry_price=trade.entry_price,
                            exit_price=closed_trade.exit_price or current_price,
                            reason=close_reason,
                            net_pnl=closed_trade.pnl or Decimal("0"),
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
                    await notify_close(
                        ticker=ticker,
                        entry_price=trade.entry_price,
                        exit_price=closed_trade.exit_price or current_price,
                        reason="SELL_SIGNAL",
                        net_pnl=closed_trade.pnl or Decimal("0"),
                        gross_pnl=breakdown.gross_pnl,
                        commission=breakdown.buy_commission + breakdown.sell_commission,
                        tax=breakdown.tax,
                    )

                # ── 8. BUY-сигнал: открываем позицию ─────────────────────────
                elif signal_type == "BUY":
                    if asset.id in open_by_asset:
                        logger.debug("Позиция уже открыта", ticker=ticker)
                        continue

                    ticker_threshold = _ticker_threshold(ticker)
                    if confidence < ticker_threshold:
                        logger.debug(
                            "Уверенность ниже порога",
                            ticker=ticker,
                            confidence=confidence,
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

                    # Проверяем баланс перед выставлением ордера
                    lots = trading_settings.lots_per_ticker
                    needed = current_price * lots * lot_size
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
    - Основная:  10:00 – 18:50 МСК
    - Вечерняя: 19:00 – 23:50 МСК

    Стоп-ордера (SL) принимаются круглосуточно — эта проверка для них не нужна.
    """
    from datetime import datetime
    now_msk = datetime.now(ZoneInfo("Europe/Moscow")).time()
    main_open = time(10, 0) <= now_msk < time(18, 50)
    evening_open = time(19, 0) <= now_msk < time(23, 50)
    return main_open or evening_open


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
