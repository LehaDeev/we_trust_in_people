"""
Исполнитель торговых операций: открывает и закрывает позиции.

Использует рыночные ордера через tinkoff/portfolio.py.
Сохраняет и обновляет объекты Trade через trade_repo.
PnL рассчитывается через trading/profitability.py (чистый, после комиссий и НДФЛ).
"""
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession
from t_tech.invest.schemas import OrderDirection, StopOrderDirection
from t_tech.invest.utils import quotation_to_decimal

from config.settings import trading_settings
from db.models import Asset, Trade
from db import trade_repo
from tinkoff.market_data import get_min_price_increment
from tinkoff.portfolio import (
    cancel_order,
    cancel_stop_order,
    post_limit_order,
    post_market_order,
    post_stop_order,
)
from trading.profitability import (
    adjusted_sl_price,
    adjusted_tp_price,
    calculate_pnl,
    round_sl_to_step,
    round_tp_to_step,
)
from utils.logger import logger


class TradeExecutor:
    """Открывает и закрывает позиции через Tinkoff API."""

    async def open_position(
        self,
        session: AsyncSession,
        asset: Asset,
        instrument_uid: str,
        current_price: Decimal,
        lot_size: int = 1,
    ) -> Trade | None:
        """
        Открыть длинную позицию по рыночной цене.

        Выставляет рыночный ордер BUY, рассчитывает стоп-лосс и тейк-профит,
        сохраняет Trade в базе данных.

        Аргументы:
            session:        активная async-сессия SQLAlchemy
            asset:          объект Asset (тикер, id)
            instrument_uid: UID инструмента для Tinkoff API
            current_price:  текущая цена за 1 бумагу (используется для расчёта SL/TP)
            lot_size:       количество бумаг в 1 лоте

        Возвращает:
            Trade с status='OPEN' при успехе, None при ошибке API.
        """
        lots = trading_settings.lots_per_ticker

        logger.info(
            "Открываем позицию",
            ticker=asset.ticker,
            lots=lots,
            lot_size=lot_size,
            price=str(current_price),
        )

        try:
            response = await post_market_order(
                instrument_id=instrument_uid,
                quantity=lots,
                direction=OrderDirection.ORDER_DIRECTION_BUY,
            )
        except Exception as e:
            logger.error(
                "Ошибка выставления ордера BUY",
                ticker=asset.ticker,
                error=str(e),
            )
            return None

        # Цена исполнения из ответа API (если доступна), иначе текущая цена
        executed_price = current_price
        if response.executed_order_price:
            try:
                executed_price = quotation_to_decimal(response.executed_order_price)
            except Exception:
                pass

        # Рассчитываем уровни SL/TP с учётом комиссий и НДФЛ:
        # при достижении этих цен чистый P&L будет ровно ±pct от суммы входа
        stop_loss_price = adjusted_sl_price(executed_price, trading_settings.stop_loss_pct)
        take_profit_price = adjusted_tp_price(executed_price, trading_settings.take_profit_pct)

        trade = Trade(
            asset_id=asset.id,
            order_id=response.order_id,
            lots=lots,
            lot_size=lot_size,
            entry_price=executed_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            status="OPEN",
        )
        trade = await trade_repo.save_trade(session, trade)

        # Выставляем биржевые ордера TP и SL сразу после покупки
        tp_order_id: str | None = None
        sl_stop_order_id: str | None = None

        # Округляем цены до минимального шага инструмента (требование Tinkoff API)
        try:
            price_step = await get_min_price_increment(instrument_uid)
        except Exception as e:
            logger.warning("Не удалось получить шаг цены, использую 0.01", ticker=asset.ticker, error=str(e))
            price_step = Decimal("0.01")

        tp_price_rounded = round_tp_to_step(take_profit_price, price_step)
        sl_price_rounded = round_sl_to_step(stop_loss_price, price_step)

        try:
            tp_order_id = (await post_limit_order(
                instrument_id=instrument_uid,
                quantity=lots,
                price=tp_price_rounded,
                direction=OrderDirection.ORDER_DIRECTION_SELL,
            )).order_id
        except Exception as e:
            logger.warning("Не удалось выставить лимитный ордер TP", ticker=asset.ticker, error=str(e))

        try:
            sl_stop_order_id = await post_stop_order(
                instrument_id=instrument_uid,
                quantity=lots,
                stop_price=sl_price_rounded,
                direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
            )
        except Exception as e:
            logger.warning("Не удалось выставить стоп-ордер SL", ticker=asset.ticker, error=str(e))

        if tp_order_id or sl_stop_order_id:
            trade.tp_order_id = tp_order_id
            trade.sl_stop_order_id = sl_stop_order_id
            trade = await trade_repo.update_trade(session, trade)

        logger.info(
            "Позиция открыта",
            ticker=asset.ticker,
            trade_id=trade.id,
            entry_price=str(trade.entry_price),
            lot_size=lot_size,
            stop_loss=str(sl_price_rounded),
            take_profit=str(tp_price_rounded),
            tp_order_id=tp_order_id,
            sl_stop_order_id=sl_stop_order_id,
        )
        return trade

    async def close_position(
        self,
        session: AsyncSession,
        trade: Trade,
        asset: Asset,
        instrument_uid: str,
        current_price: Decimal,
        reason: str,
    ) -> Trade:
        """
        Закрыть открытую позицию рыночным ордером SELL.

        Выставляет рыночный ордер SELL, рассчитывает чистый PnL
        (с учётом комиссий брокера и НДФЛ), обновляет Trade в БД (status='CLOSED').

        Аргументы:
            session:        активная async-сессия SQLAlchemy
            trade:          открытая сделка (Trade со status='OPEN')
            asset:          объект Asset (тикер)
            instrument_uid: UID инструмента для Tinkoff API
            current_price:  текущая цена за 1 бумагу (используется если API не вернул цену)
            reason:         причина закрытия ("SELL_SIGNAL" | "STOP_LOSS" | "TAKE_PROFIT" | "MANUAL")

        Возвращает:
            Обновлённый Trade с status='CLOSED'.
        """
        logger.info(
            "Закрываем позицию",
            ticker=asset.ticker,
            trade_id=trade.id,
            reason=reason,
            current_price=str(current_price),
        )

        exit_price = current_price

        # Отмена биржевых ордеров по логике OCO:
        # - SL сработал → отменяем TP (позицию уже закрыла биржа через SL)
        # - TP сработал → отменяем SL (позицию уже закрыла биржа через TP)
        # - Иной сигнал  → отменяем оба, затем выставляем рыночный SELL
        if reason == "STOP_LOSS" and trade.tp_order_id:
            try:
                await cancel_order(trade.tp_order_id)
            except Exception as e:
                logger.warning("Не удалось отменить TP ордер после SL", order_id=trade.tp_order_id, error=str(e))
        elif reason == "TAKE_PROFIT" and trade.sl_stop_order_id:
            try:
                await cancel_stop_order(trade.sl_stop_order_id)
            except Exception as e:
                logger.warning("Не удалось отменить SL стоп-ордер после TP", stop_order_id=trade.sl_stop_order_id, error=str(e))
        else:
            if trade.tp_order_id:
                try:
                    await cancel_order(trade.tp_order_id)
                except Exception as e:
                    logger.warning("Не удалось отменить TP ордер", order_id=trade.tp_order_id, error=str(e))
            if trade.sl_stop_order_id:
                try:
                    await cancel_stop_order(trade.sl_stop_order_id)
                except Exception as e:
                    logger.warning("Не удалось отменить SL стоп-ордер", stop_order_id=trade.sl_stop_order_id, error=str(e))

        # Если TP/SL сработал на бирже — позиция уже закрыта биржей,
        # рыночный ордер выставлять не нужно
        if reason in ("TAKE_PROFIT", "STOP_LOSS") and trade.tp_order_id:
            # Позиция закрыта биржей, используем текущую цену как цену выхода
            pass
        else:
            try:
                response = await post_market_order(
                    instrument_id=instrument_uid,
                    quantity=trade.lots,
                    direction=OrderDirection.ORDER_DIRECTION_SELL,
                )
                if response.executed_order_price:
                    try:
                        exit_price = quotation_to_decimal(response.executed_order_price)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(
                    "Ошибка выставления ордера SELL",
                    ticker=asset.ticker,
                    trade_id=trade.id,
                    error=str(e),
                )
                exit_price = current_price

        # Чистый PnL: учитываем комиссии и НДФЛ
        lot_size = getattr(trade, "lot_size", 1) or 1
        breakdown = calculate_pnl(
            entry_price=trade.entry_price,
            exit_price=exit_price,
            lots=trade.lots,
            lot_size=lot_size,
        )
        net_pnl = breakdown.net_pnl

        trade.exit_price = exit_price
        trade.status = "CLOSED"
        trade.close_reason = reason
        trade.pnl = net_pnl
        trade.closed_at = datetime.now(timezone.utc)
        trade = await trade_repo.update_trade(session, trade)

        logger.info(
            "Позиция закрыта",
            ticker=asset.ticker,
            trade_id=trade.id,
            reason=reason,
            entry_price=str(trade.entry_price),
            exit_price=str(exit_price),
            gross_pnl=str(breakdown.gross_pnl),
            commission=str(breakdown.buy_commission + breakdown.sell_commission),
            tax=str(breakdown.tax),
            net_pnl=str(net_pnl),
        )
        return trade
