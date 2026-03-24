"""
Мониторинг биржевых ордеров для открытых позиций.

Проверяет исполнение TP/SL стоп-ордеров, перевыставляет пропавшие ордера,
возвращает причину закрытия позиции.
"""
from datetime import datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession
from t_tech.invest.schemas import OrderExecutionReportStatus, StopOrderDirection, StopOrderType

from db import trade_repo
from db.models import Asset, Trade
from tinkoff.market_data import get_min_price_increment
from tinkoff.portfolio import get_order_state, post_stop_order
from trading.profitability import round_sl_to_step, round_tp_to_step
from utils.logger import logger


def _is_moex_session_open() -> bool:
    """
    Проверить, открыта ли торговая сессия МОЕХ прямо сейчас.

    Лимитные ордера (TP) принимаются только во время сессии:
    - Предторговый: 09:45 – 10:00 МСК (лимитные T0 принимаются)
    - Основная:     10:00 – 18:50 МСК
    - Вечерняя:     19:05 – 23:50 МСК

    Стоп-ордера (SL) принимаются круглосуточно — эта проверка для них не нужна.
    """
    now_msk = datetime.now(ZoneInfo("Europe/Moscow")).time()
    pretrade    = time(9, 45)  <= now_msk < time(10, 0)
    main_open   = time(10, 0)  <= now_msk < time(18, 50)
    evening_open = time(19, 5) <= now_msk < time(23, 50)
    return pretrade or main_open or evening_open


async def _check_trade_orders(
    session: AsyncSession,
    trade: Trade,
    asset: Asset,
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

    # ── Нет TP (SL может уже быть) — выставляем недостающие ордера ──────────
    if close_reason is None and not trade.tp_stop_order_id and not trade.tp_order_id:
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
