"""
Мониторинг исполнения ордеров в реальном времени.

Два параллельных механизма:
- gRPC stream (trades_stream): мгновенно обрабатывает TP для старых позиций
  с tp_order_id (лимитный ордер). Матчинг по order_id.

- Быстрый polling (каждые TRADING_ORDER_POLL_SECONDS секунд): проверяет SL
  и TP для новых позиций (tp_stop_order_id).
  - SL-стоп: при срабатывании создаёт НОВЫЙ рыночный ордер с другим ID —
    матчинг через stream невозможен. Признак: stop_order_id исчез + цена <= SL.
  - TP-стоп: аналогично. Признак: stop_order_id исчез + цена >= TP.

Оба механизма переподключаются при сбое. Конкурентность со scheduler решается
через повторное чтение статуса сделки из БД перед закрытием.
"""
import asyncio
from decimal import Decimal

from sqlalchemy import select

from config.settings import tinkoff_settings, trading_settings
from db.database import get_session
from db.models import Asset, Trade
from db import trade_repo
from t_tech.invest import AsyncClient
from t_tech.invest.utils import quotation_to_decimal
from tinkoff.market_data import get_last_prices
from tinkoff.portfolio import cancel_order, cancel_stop_order, get_stop_order_ids
from trading.executor import TradeExecutor
from trading.notifier import notify_close
from trading.profitability import calculate_pnl
from utils.logger import logger


class OrderWatcher:
    """
    Мониторинг исполнения биржевых ордеров в реальном времени.

    Запускается как фоновая asyncio-задача рядом с TradingScheduler.
    """

    def __init__(self) -> None:
        """Инициализировать исполнителя сделок."""
        self._executor = TradeExecutor()

    async def run(self) -> None:
        """Запустить оба цикла мониторинга параллельно."""
        logger.info("OrderWatcher запущен")
        await asyncio.gather(
            self._stream_loop(),
            self._poll_loop(),
        )

    # ── gRPC stream: мгновенное обнаружение legacy TP (лимитный ордер) ────────

    async def _stream_loop(self) -> None:
        """Бесконечный цикл подключения к trades_stream с переподключением."""
        while True:
            try:
                await self._stream_once()
            except Exception as e:
                logger.warning(
                    "OrderWatcher stream прерван, переподключение через 5с",
                    error=str(e),
                )
                await asyncio.sleep(5)

    async def _stream_once(self) -> None:
        """Подключиться к trades_stream и обрабатывать события до обрыва."""
        logger.info("OrderWatcher: подключение к trades_stream")
        async with AsyncClient(tinkoff_settings.token) as client:
            async for response in client.orders_stream.trades_stream(
                accounts=[tinkoff_settings.account_id]
            ):
                if not response.order_trades:
                    continue
                ot = response.order_trades
                if not ot.trades:
                    continue
                # Средняя цена исполнения по всем сделкам в ответе
                try:
                    fill_price = quotation_to_decimal(ot.trades[-1].price)
                except Exception:
                    fill_price = None

                await self._handle_tp_fill(
                    order_id=ot.order_id,
                    figi=ot.figi or ot.instrument_uid,
                    fill_price=fill_price,
                )

    async def _handle_tp_fill(
        self,
        order_id: str,
        figi: str,
        fill_price: Decimal | None,
    ) -> None:
        """
        Обработать исполнение ордера из stream.

        Проверяет, является ли order_id тейк-профитом одной из открытых позиций
        (только для legacy tp_order_id — лимитные ордера).
        Новые позиции с tp_stop_order_id обрабатываются через polling (_check_tp_sl).
        """
        async with get_session() as session:
            open_trades = await trade_repo.get_open_trades(session)
            # Матчинг только по legacy tp_order_id (лимитный ордер)
            trade = next((t for t in open_trades if t.tp_order_id == order_id), None)
            if trade is None:
                return

            result = await session.execute(select(Asset).where(Asset.id == trade.asset_id))
            asset = result.scalar_one_or_none()
            if asset is None:
                return

            logger.info(
                "OrderWatcher: legacy TP исполнен (stream)",
                ticker=asset.ticker,
                trade_id=trade.id,
                order_id=order_id,
                fill_price=str(fill_price) if fill_price else "unknown",
            )

            # Отменяем SL стоп-ордер
            if trade.sl_stop_order_id:
                try:
                    await cancel_stop_order(trade.sl_stop_order_id)
                except Exception as e:
                    logger.warning(
                        "OrderWatcher: не удалось отменить SL после legacy TP",
                        stop_order_id=trade.sl_stop_order_id,
                        error=str(e),
                    )

            exit_price = fill_price or trade.take_profit_price
            lot_size = getattr(trade, "lot_size", 1) or 1
            breakdown = calculate_pnl(
                entry_price=trade.entry_price,
                exit_price=exit_price,
                lots=trade.lots,
                lot_size=lot_size,
            )

            # Закрываем позицию в БД (без рыночного ордера — биржа уже закрыла)
            trade.exit_price = exit_price
            trade.status = "CLOSED"
            trade.close_reason = "TAKE_PROFIT"
            trade.pnl = breakdown.net_pnl
            from datetime import datetime, timezone
            trade.closed_at = datetime.now(timezone.utc)
            await trade_repo.update_trade(session, trade)

            await notify_close(
                ticker=asset.ticker,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                reason="TAKE_PROFIT",
                net_pnl=breakdown.net_pnl,
                gross_pnl=breakdown.gross_pnl,
                commission=breakdown.buy_commission + breakdown.sell_commission,
                tax=breakdown.tax,
            )

    # ── Быстрый polling: обнаружение SL и нового TP стоп-ордера ──────────────

    async def _poll_loop(self) -> None:
        """Проверять SL и TP стоп-ордера каждые TRADING_ORDER_POLL_SECONDS секунд."""
        interval = trading_settings.order_poll_seconds
        logger.info("OrderWatcher: polling SL/TP запущен", interval_seconds=interval)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._check_tp_sl()
            except Exception as e:
                logger.warning("OrderWatcher: ошибка проверки SL/TP", error=str(e))

    async def _check_tp_sl(self) -> None:
        """
        Проверить исполнение SL и TP стоп-ордеров для открытых позиций.

        Логика SL: sl_stop_order_id исчез из активных + цена <= stop_loss_price.
        Логика TP (стоп): tp_stop_order_id исчез из активных + цена >= take_profit_price.

        Для legacy tp_order_id (лимитный ордер) — обрабатывается через stream.
        """
        async with get_session() as session:
            open_trades = await trade_repo.get_open_trades(session)
            # Нас интересуют позиции хотя бы с одним стоп-ордером (SL или TP-стоп)
            trades_with_stops = [
                t for t in open_trades
                if t.sl_stop_order_id or t.tp_stop_order_id
            ]
            if not trades_with_stops:
                return

            # Загружаем активные стоп-ордера и текущие цены
            try:
                active_stop_ids = await get_stop_order_ids()
            except Exception as e:
                logger.warning("OrderWatcher: не удалось получить стоп-ордера", error=str(e))
                return

            result = await session.execute(select(Asset))
            all_assets = {a.id: a for a in result.scalars().all()}
            asset_to_figi = {a.id: a.figi for a in all_assets.values()}

            figis = [
                asset_to_figi[t.asset_id]
                for t in trades_with_stops
                if t.asset_id in asset_to_figi
            ]
            if not figis:
                return

            try:
                prices = await get_last_prices(figis)
            except Exception as e:
                logger.warning("OrderWatcher: не удалось получить цены", error=str(e))
                return

            for trade in trades_with_stops:
                figi = asset_to_figi.get(trade.asset_id)
                if not figi:
                    continue

                current_price = prices.get(figi, Decimal("0"))
                if current_price <= 0:
                    continue

                asset = all_assets.get(trade.asset_id)
                if asset is None:
                    continue

                # ── Проверка TP стоп-ордера ────────────────────────────────
                if trade.tp_stop_order_id and trade.tp_stop_order_id not in active_stop_ids:
                    if current_price >= trade.take_profit_price:
                        # TP стоп-ордер исчез и цена >= TP → TP исполнился
                        logger.info(
                            "OrderWatcher: TP стоп-ордер сработал (polling)",
                            ticker=asset.ticker,
                            trade_id=trade.id,
                            current_price=str(current_price),
                            take_profit_price=str(trade.take_profit_price),
                        )

                        # Отменяем SL стоп-ордер
                        if trade.sl_stop_order_id:
                            try:
                                await cancel_stop_order(trade.sl_stop_order_id)
                            except Exception as e:
                                logger.warning(
                                    "OrderWatcher: не удалось отменить SL после TP",
                                    stop_order_id=trade.sl_stop_order_id,
                                    error=str(e),
                                )

                        exit_price = current_price
                        lot_size = getattr(trade, "lot_size", 1) or 1
                        breakdown = calculate_pnl(
                            entry_price=trade.entry_price,
                            exit_price=exit_price,
                            lots=trade.lots,
                            lot_size=lot_size,
                        )

                        # Закрываем в БД (биржа уже продала через стоп-TP)
                        trade.exit_price = exit_price
                        trade.status = "CLOSED"
                        trade.close_reason = "TAKE_PROFIT"
                        trade.pnl = breakdown.net_pnl
                        from datetime import datetime, timezone
                        trade.closed_at = datetime.now(timezone.utc)
                        await trade_repo.update_trade(session, trade)

                        await notify_close(
                            ticker=asset.ticker,
                            entry_price=trade.entry_price,
                            exit_price=exit_price,
                            reason="TAKE_PROFIT",
                            net_pnl=breakdown.net_pnl,
                            gross_pnl=breakdown.gross_pnl,
                            commission=breakdown.buy_commission + breakdown.sell_commission,
                            tax=breakdown.tax,
                        )
                        continue  # позиция закрыта — следующий trade

                # ── Проверка SL стоп-ордера ────────────────────────────────
                if not trade.sl_stop_order_id:
                    continue
                if trade.sl_stop_order_id in active_stop_ids:
                    continue  # SL ещё активен

                if current_price > trade.stop_loss_price:
                    # Стоп-ордер исчез, но цена выше SL — не SL, игнорируем
                    # (scheduler обработает перевыставление на следующем тике)
                    continue

                logger.info(
                    "OrderWatcher: SL сработал (polling)",
                    ticker=asset.ticker,
                    trade_id=trade.id,
                    current_price=str(current_price),
                    stop_loss_price=str(trade.stop_loss_price),
                )

                # Отменяем TP стоп-ордер (новый формат) или лимитный (legacy)
                if trade.tp_stop_order_id:
                    try:
                        await cancel_stop_order(trade.tp_stop_order_id)
                    except Exception as e:
                        logger.warning(
                            "OrderWatcher: не удалось отменить TP стоп-ордер после SL",
                            stop_order_id=trade.tp_stop_order_id,
                            error=str(e),
                        )
                if trade.tp_order_id:
                    try:
                        await cancel_order(trade.tp_order_id)
                    except Exception as e:
                        logger.warning(
                            "OrderWatcher: не удалось отменить legacy TP после SL",
                            order_id=trade.tp_order_id,
                            error=str(e),
                        )

                lot_size = getattr(trade, "lot_size", 1) or 1
                breakdown = calculate_pnl(
                    entry_price=trade.entry_price,
                    exit_price=current_price,
                    lots=trade.lots,
                    lot_size=lot_size,
                )

                # Закрываем в БД (биржа уже продала через стоп-ордер)
                trade.exit_price = current_price
                trade.status = "CLOSED"
                trade.close_reason = "STOP_LOSS"
                trade.pnl = breakdown.net_pnl
                from datetime import datetime, timezone
                trade.closed_at = datetime.now(timezone.utc)
                await trade_repo.update_trade(session, trade)

                await notify_close(
                    ticker=asset.ticker,
                    entry_price=trade.entry_price,
                    exit_price=current_price,
                    reason="STOP_LOSS",
                    net_pnl=breakdown.net_pnl,
                    gross_pnl=breakdown.gross_pnl,
                    commission=breakdown.buy_commission + breakdown.sell_commission,
                    tax=breakdown.tax,
                )
