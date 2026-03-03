"""
Главный цикл автоматической торговли.

Алгоритм одного тика (_tick):
    1. Загружаем все открытые позиции из БД
    2. Получаем текущие цены инструментов
    3. Проверяем стоп-лосс и тейк-профит для каждой открытой позиции
    4. Получаем ML-сигналы для всех тикеров
    5. Открываем позиции по BUY-сигналам (если нет открытой позиции)
    6. Закрываем позиции по SELL-сигналам

Безопасность:
    - TRADING_ENABLED=false → тик логируется, но ордера не выставляются
    - Все исключения перехватываются — цикл не прерывается
    - Максимум TRADING_MAX_POSITIONS одновременно открытых позиций
"""
import asyncio
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import trading_settings
from db.database import get_session
from db.models import Asset, Trade
from db import trade_repo
from ml.predict import predict_all
from tinkoff.market_data import get_last_prices
from trading.executor import TradeExecutor
from trading.notifier import notify_close, notify_open
from utils.logger import logger


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

            logger.debug("Следующий тик через %d секунд", interval)
            await asyncio.sleep(interval)

    async def _tick(self) -> None:
        """
        Один прогон автоторговли.

        Открывает и закрывает позиции в рамках одной транзакции.
        """
        if not trading_settings.enabled:
            logger.info("Автоторговля отключена (TRADING_ENABLED=false) — тик пропущен")
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

            # ── 3. Текущие цены для всех открытых позиций ────────────────────
            open_figis = [
                figi_to_asset[a_id].figi
                for a_id in open_by_asset
                if a_id in {a.id for a in all_assets}
                and any(a.id == a_id for a in all_assets)
            ]
            # Строим map asset_id → figi для открытых позиций
            asset_id_to_figi: dict[int, str] = {a.id: a.figi for a in all_assets}

            if open_trades:
                open_figis_list = [
                    asset_id_to_figi[t.asset_id]
                    for t in open_trades
                    if t.asset_id in asset_id_to_figi
                ]
                if open_figis_list:
                    prices = await get_last_prices(open_figis_list)
                else:
                    prices = {}

                # ── 4. Проверяем SL/TP ────────────────────────────────────────
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
                    if current_price <= trade.stop_loss_price:
                        close_reason = "STOP_LOSS"
                    elif current_price >= trade.take_profit_price:
                        close_reason = "TAKE_PROFIT"

                    if close_reason:
                        closed_trade = await self._executor.close_position(
                            session=session,
                            trade=trade,
                            asset=asset,
                            instrument_uid=figi,
                            current_price=current_price,
                            reason=close_reason,
                        )
                        # Убираем из карты открытых — позиция уже закрыта
                        open_by_asset.pop(trade.asset_id, None)
                        await notify_close(
                            ticker=asset.ticker,
                            entry_price=trade.entry_price,
                            exit_price=closed_trade.exit_price or current_price,
                            reason=close_reason,
                            pnl=closed_trade.pnl or Decimal("0"),
                        )
            else:
                prices = {}

            # ── 5. ML-сигналы ─────────────────────────────────────────────────
            signals = await predict_all()
            logger.info("Сигналы получены", count=len(signals))

            # Количество реально открытых позиций после проверки SL/TP
            open_count = len(open_by_asset)
            max_positions = trading_settings.max_open_positions

            for sig in signals:
                ticker: str = sig.get("ticker", "")
                signal_type: str = sig.get("signal", "HOLD")
                confidence: float = sig.get("confidence", 0.0)

                asset = ticker_to_asset.get(ticker)
                if not asset:
                    logger.debug("Актив не найден в БД", ticker=ticker)
                    continue

                figi = asset.figi

                # ── 6. SELL-сигнал: закрываем открытую позицию ───────────────
                if signal_type == "SELL" and asset.id in open_by_asset:
                    trade = open_by_asset[asset.id]
                    current_price = prices.get(figi) or await _fetch_price(figi)
                    if current_price > 0:
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
                            pnl=closed_trade.pnl or Decimal("0"),
                        )

                # ── 7. BUY-сигнал: открываем позицию ─────────────────────────
                elif signal_type == "BUY":
                    if asset.id in open_by_asset:
                        logger.debug("Позиция уже открыта", ticker=ticker)
                        continue

                    if confidence < trading_settings.confidence_threshold:
                        logger.debug(
                            "Уверенность ниже порога",
                            ticker=ticker,
                            confidence=confidence,
                            threshold=trading_settings.confidence_threshold,
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

                    new_trade = await self._executor.open_position(
                        session=session,
                        asset=asset,
                        instrument_uid=figi,
                        current_price=current_price,
                    )
                    if new_trade:
                        open_by_asset[asset.id] = new_trade
                        open_count += 1
                        await notify_open(
                            ticker=ticker,
                            price=new_trade.entry_price,
                            lots=new_trade.lots,
                            stop_loss=new_trade.stop_loss_price,
                            take_profit=new_trade.take_profit_price,
                        )

        logger.info(
            "Тик завершён",
            open_positions=open_count,
        )


async def _fetch_price(figi: str) -> Decimal:
    """
    Получить текущую цену одного инструмента.

    Вспомогательная функция для случаев когда цена не была загружена заранее.

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
