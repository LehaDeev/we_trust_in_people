"""
Исполнитель торговых операций: открывает и закрывает позиции.

Использует рыночные ордера через tinkoff/portfolio.py.
Сохраняет и обновляет объекты Trade через trade_repo.
PnL рассчитывается через trading/profitability.py (чистый, после комиссий и НДФЛ).
"""
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession
from t_tech.invest.schemas import OrderDirection, StopOrderDirection, StopOrderType
from t_tech.invest.utils import quotation_to_decimal

from config.settings import trading_settings
from db.models import Asset, Trade
from db import trade_repo
from tinkoff.market_data import get_min_price_increment
from tinkoff.portfolio import (
    cancel_order,
    cancel_stop_order,
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
        sl_pct: float | None = None,
        tp_pct: float | None = None,
        lots: int | None = None,
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
            sl_pct:         целевой чистый убыток для SL (доля; None = взять из trading_settings)
            tp_pct:         целевая чистая прибыль для TP (доля; None = взять из trading_settings)
            lots:           число лотов для покупки; None = взять из trading_settings.lots_per_ticker
                            (обратная совместимость с ручным режимом и тестами)

        Возвращает:
            Trade с status='OPEN' при успехе, None при ошибке API.
        """
        _lots = lots if lots is not None else trading_settings.lots_per_ticker

        logger.info(
            "Открываем позицию",
            ticker=asset.ticker,
            lots=_lots,
            lot_size=lot_size,
            price=str(current_price),
        )

        try:
            response = await post_market_order(
                instrument_id=instrument_uid,
                quantity=_lots,
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
        # при достижении этих цен чистый P&L будет ровно ±pct от суммы входа.
        # Используем переданные значения (динамические ATR) или фиксированные из настроек.
        _sl_pct = sl_pct if sl_pct is not None else trading_settings.stop_loss_pct
        _tp_pct = tp_pct if tp_pct is not None else trading_settings.take_profit_pct
        stop_loss_price = adjusted_sl_price(executed_price, _sl_pct)
        take_profit_price = adjusted_tp_price(executed_price, _tp_pct)

        trade = Trade(
            asset_id=asset.id,
            order_id=response.order_id,
            lots=_lots,
            lot_size=lot_size,
            entry_price=executed_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            status="OPEN",
        )
        trade = await trade_repo.save_trade(session, trade)

        # Выставляем биржевые стоп-ордера TP и SL сразу после покупки.
        # Оба используют post_stop_order — стоп-ордера не блокируют акции
        # в портфеле до момента срабатывания, устраняя OCO-конфликт
        # "Недостаточно бумаг в портфеле".
        tp_stop_order_id: str | None = None
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
            # TP как STOP_ORDER_TYPE_TAKE_PROFIT: срабатывает когда цена >= tp_price.
            # Стоп-ордер не резервирует акции — нет конфликта с SL стоп-ордером.
            tp_stop_order_id = await post_stop_order(
                instrument_id=instrument_uid,
                quantity=_lots,
                stop_price=tp_price_rounded,
                direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
            )
        except Exception as e:
            logger.warning("Не удалось выставить стоп-ордер TP", ticker=asset.ticker, error=str(e))

        try:
            sl_stop_order_id = await post_stop_order(
                instrument_id=instrument_uid,
                quantity=_lots,
                stop_price=sl_price_rounded,
                direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
                stop_order_type=StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
            )
        except Exception as e:
            logger.warning("Не удалось выставить стоп-ордер SL", ticker=asset.ticker, error=str(e))

        if tp_stop_order_id or sl_stop_order_id:
            trade.tp_stop_order_id = tp_stop_order_id
            trade.sl_stop_order_id = sl_stop_order_id
            trade = await trade_repo.update_trade(session, trade)

        logger.info(
            "Позиция открыта",
            ticker=asset.ticker,
            trade_id=trade.id,
            entry_price=str(trade.entry_price),
            lots=_lots,
            lot_size=lot_size,
            position_value=str(executed_price * _lots * lot_size),
            sl_pct=round(_sl_pct, 4),
            tp_pct=round(_tp_pct, 4),
            stop_loss=str(sl_price_rounded),
            take_profit=str(tp_price_rounded),
            tp_stop_order_id=tp_stop_order_id,
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
        #
        # Backward compatibility: tp_order_id — старый лимитный ордер (cancel_order),
        # tp_stop_order_id — новый стоп-ордер TAKE_PROFIT (cancel_stop_order).
        if reason == "STOP_LOSS":
            # Отменяем TP (новый стоп-TP или старый лимитный)
            if trade.tp_stop_order_id:
                try:
                    await cancel_stop_order(trade.tp_stop_order_id)
                except Exception as e:
                    logger.warning(
                        "Не удалось отменить TP стоп-ордер после SL",
                        stop_order_id=trade.tp_stop_order_id,
                        error=str(e),
                    )
            if trade.tp_order_id:
                try:
                    await cancel_order(trade.tp_order_id)
                except Exception as e:
                    logger.warning(
                        "Не удалось отменить TP лимитный ордер после SL (legacy)",
                        order_id=trade.tp_order_id,
                        error=str(e),
                    )
        elif reason == "TAKE_PROFIT":
            if trade.sl_stop_order_id:
                try:
                    await cancel_stop_order(trade.sl_stop_order_id)
                except Exception as e:
                    logger.warning(
                        "Не удалось отменить SL стоп-ордер после TP",
                        stop_order_id=trade.sl_stop_order_id,
                        error=str(e),
                    )
        else:
            # SELL_SIGNAL или MANUAL: отменяем оба
            if trade.tp_stop_order_id:
                try:
                    await cancel_stop_order(trade.tp_stop_order_id)
                except Exception as e:
                    logger.warning(
                        "Не удалось отменить TP стоп-ордер",
                        stop_order_id=trade.tp_stop_order_id,
                        error=str(e),
                    )
            if trade.tp_order_id:
                try:
                    await cancel_order(trade.tp_order_id)
                except Exception as e:
                    logger.warning(
                        "Не удалось отменить TP лимитный ордер (legacy)",
                        order_id=trade.tp_order_id,
                        error=str(e),
                    )
            if trade.sl_stop_order_id:
                try:
                    await cancel_stop_order(trade.sl_stop_order_id)
                except Exception as e:
                    logger.warning(
                        "Не удалось отменить SL стоп-ордер",
                        stop_order_id=trade.sl_stop_order_id,
                        error=str(e),
                    )

        # Если TP исполнился на бирже (подтверждено через get_order_state + FILL) —
        # позиция уже закрыта биржей, рыночный ордер выставлять не нужно.
        # Для STOP_LOSS всегда выставляем рыночный SELL: стоп-ордер мог не исполниться
        # (например, биржа вернула "Недостаточно бумаг") и акции остались в портфеле.
        if reason == "TAKE_PROFIT":
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
                    "Ошибка выставления ордера SELL — позиция НЕ закрыта в БД",
                    ticker=asset.ticker,
                    trade_id=trade.id,
                    error=str(e),
                )
                raise

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
