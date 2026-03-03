"""
Исполнитель торговых операций: открывает и закрывает позиции.

Использует рыночные ордера через tinkoff/portfolio.py.
Сохраняет и обновляет объекты Trade через trade_repo.
"""
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession
from t_tech.invest.schemas import OrderDirection
from t_tech.invest.utils import quotation_to_decimal

from config.settings import trading_settings
from db.models import Asset, Trade
from db import trade_repo
from tinkoff.portfolio import post_market_order
from utils.logger import logger


class TradeExecutor:
    """Открывает и закрывает позиции через Tinkoff API."""

    async def open_position(
        self,
        session: AsyncSession,
        asset: Asset,
        instrument_uid: str,
        current_price: Decimal,
    ) -> Trade | None:
        """
        Открыть длинную позицию по рыночной цене.

        Выставляет рыночный ордер BUY, рассчитывает стоп-лосс и тейк-профит,
        сохраняет Trade в базе данных.

        Аргументы:
            session:        активная async-сессия SQLAlchemy
            asset:          объект Asset (тикер, id)
            instrument_uid: UID инструмента для Tinkoff API
            current_price:  текущая цена (используется для расчёта SL/TP)

        Возвращает:
            Trade с status='OPEN' при успехе, None при ошибке API.
        """
        lots = trading_settings.lots_per_ticker

        logger.info(
            "Открываем позицию",
            ticker=asset.ticker,
            lots=lots,
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

        # Рассчитываем уровни SL/TP
        sl_pct = Decimal(str(trading_settings.stop_loss_pct))
        tp_pct = Decimal(str(trading_settings.take_profit_pct))
        stop_loss_price = executed_price * (Decimal("1") - sl_pct)
        take_profit_price = executed_price * (Decimal("1") + tp_pct)

        trade = Trade(
            asset_id=asset.id,
            order_id=response.order_id,
            lots=lots,
            entry_price=executed_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            status="OPEN",
        )
        trade = await trade_repo.save_trade(session, trade)

        logger.info(
            "Позиция открыта",
            ticker=asset.ticker,
            trade_id=trade.id,
            entry_price=str(trade.entry_price),
            stop_loss=str(stop_loss_price),
            take_profit=str(take_profit_price),
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

        Выставляет рыночный ордер SELL, рассчитывает PnL,
        обновляет Trade в базе данных (status='CLOSED').

        Аргументы:
            session:        активная async-сессия SQLAlchemy
            trade:          открытая сделка (Trade со status='OPEN')
            asset:          объект Asset (тикер)
            instrument_uid: UID инструмента для Tinkoff API
            current_price:  текущая цена (используется для расчёта PnL если API не вернул)
            reason:         причина закрытия ("SELL_SIGNAL" | "STOP_LOSS" | "TAKE_PROFIT")

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
            # Даже при ошибке API помечаем позицию закрытой (позиция уже может быть закрыта)
            exit_price = current_price

        # PnL = (цена выхода - цена входа) × количество лотов
        pnl = (exit_price - trade.entry_price) * trade.lots

        trade.exit_price = exit_price
        trade.status = "CLOSED"
        trade.close_reason = reason
        trade.pnl = pnl
        trade.closed_at = datetime.now(timezone.utc)
        trade = await trade_repo.update_trade(session, trade)

        logger.info(
            "Позиция закрыта",
            ticker=asset.ticker,
            trade_id=trade.id,
            reason=reason,
            entry_price=str(trade.entry_price),
            exit_price=str(exit_price),
            pnl=str(pnl),
        )
        return trade
