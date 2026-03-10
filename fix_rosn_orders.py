"""
Одноразовый скрипт: выставить биржевые TP/SL ордера для позиции ROSN (trade_id=1).
Цены округляются до минимального шага 0.05.
"""
import asyncio
from decimal import Decimal, ROUND_UP, ROUND_DOWN

from config.settings import tinkoff_settings
from db.database import get_session
from db import trade_repo
from tinkoff.portfolio import post_limit_order, post_stop_order
from t_tech.invest.schemas import OrderDirection, StopOrderDirection

TRADE_ID = 1
MIN_STEP = Decimal("0.05")


def round_to_step(price: Decimal, step: Decimal, rounding) -> Decimal:
    return (price / step).to_integral_value(rounding=rounding) * step


async def main():
    async with get_session() as session:
        from db.models import Trade
        from sqlalchemy import select as sa_select
        result = await session.execute(sa_select(Trade).where(Trade.id == TRADE_ID))
        trade = result.scalar_one_or_none()
        if trade is None:
            print(f"Trade {TRADE_ID} не найдена")
            return

        print(f"Trade {TRADE_ID}: entry={trade.entry_price}, tp={trade.take_profit_price}, sl={trade.stop_loss_price}")
        print(f"  tp_order_id={trade.tp_order_id}, sl_stop_order_id={trade.sl_stop_order_id}")

        if trade.tp_order_id or trade.sl_stop_order_id:
            print("Ордера уже выставлены!")
            return

        # Округляем TP вверх (лимитный ордер чуть выше — консервативно)
        tp_price = round_to_step(trade.take_profit_price, MIN_STEP, ROUND_UP)
        # Округляем SL вниз (стоп чуть ниже — даём запас)
        sl_price = round_to_step(trade.stop_loss_price, MIN_STEP, ROUND_DOWN)

        print(f"  TP исходная: {trade.take_profit_price} → округлённая: {tp_price}")
        print(f"  SL исходная: {trade.stop_loss_price} → округлённая: {sl_price}")

        # Нужно узнать instrument_uid для ROSN
        from db.models import Asset
        from sqlalchemy import select as sa_select2
        result = await session.execute(sa_select2(Asset).where(Asset.id == trade.asset_id))
        asset = result.scalar_one_or_none()
        if asset is None:
            print("Asset не найден")
            return
        instrument_id = asset.figi
        print(f"  Asset: {asset.ticker}, figi={instrument_id}")

        tp_order_id = None
        sl_stop_order_id = None

        # TP — лимитный ордер SELL
        try:
            resp = await post_limit_order(
                instrument_id=instrument_id,
                quantity=trade.lots,
                price=tp_price,
                direction=OrderDirection.ORDER_DIRECTION_SELL,
            )
            tp_order_id = resp.order_id
            print(f"  TP ордер выставлен: {tp_order_id}")
        except Exception as e:
            print(f"  ОШИБКА TP: {e}")

        # SL — стоп-ордер SELL
        try:
            sl_stop_order_id = await post_stop_order(
                instrument_id=instrument_id,
                quantity=trade.lots,
                stop_price=sl_price,
                direction=StopOrderDirection.STOP_ORDER_DIRECTION_SELL,
            )
            print(f"  SL стоп-ордер выставлен: {sl_stop_order_id}")
        except Exception as e:
            print(f"  ОШИБКА SL: {e}")

        if tp_order_id or sl_stop_order_id:
            trade.tp_order_id = tp_order_id
            trade.sl_stop_order_id = sl_stop_order_id
            await trade_repo.update_trade(session, trade)
            print("Trade обновлена в БД.")
        else:
            print("Ни один ордер не выставлен — БД не обновлена.")


if __name__ == "__main__":
    asyncio.run(main())
