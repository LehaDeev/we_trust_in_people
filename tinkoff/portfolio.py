"""
Операции с портфелем и ордерами через Tinkoff Invest API.
Баланс, позиции, выставление и отмена заявок.
"""
from decimal import Decimal

from t_tech.invest.schemas import (
    OrderDirection,
    OrderType,
    PortfolioPosition,
    PortfolioResponse,
    PositionsResponse,
    PostOrderResponse,
)
from t_tech.invest.utils import money_to_decimal, quotation_to_decimal

from config.settings import tinkoff_settings
from tinkoff.client import get_client
from utils.logger import logger

ACCOUNT_ID = tinkoff_settings.account_id


async def get_portfolio() -> PortfolioResponse:
    """
    Получить текущий портфель (позиции + стоимость).

    Returns:
        PortfolioResponse с полями positions, total_amount_shares,
        total_amount_bonds, total_amount_etf, total_amount_currencies
    """
    async with get_client() as client:
        portfolio = await client.operations.get_portfolio(
            account_id=ACCOUNT_ID,
        )
    logger.info(
        "Portfolio fetched",
        positions_count=len(portfolio.positions),
    )
    return portfolio


async def get_positions() -> PositionsResponse:
    """
    Получить текущие позиции по счёту (упрощённый вид).

    Returns:
        PositionsResponse с securities, futures, options, currencies, money
    """
    async with get_client() as client:
        positions = await client.operations.get_positions(
            account_id=ACCOUNT_ID,
        )
    return positions


async def get_portfolio_summary() -> dict:
    """
    Краткая сводка по портфелю в удобном формате.

    Returns:
        Словарь с суммами по типам активов и списком позиций
    """
    portfolio = await get_portfolio()

    positions = []
    for pos in portfolio.positions:
        positions.append({
            "figi": pos.figi,
            "instrument_type": pos.instrument_type,
            "quantity": quotation_to_decimal(pos.quantity),
            "current_price": money_to_decimal(pos.current_price),
            "current_nkd": money_to_decimal(pos.current_nkd),
            "average_buy_price": money_to_decimal(pos.average_position_price),
            "expected_yield": money_to_decimal(pos.expected_yield),
        })

    return {
        "total_shares": money_to_decimal(portfolio.total_amount_shares),
        "total_bonds": money_to_decimal(portfolio.total_amount_bonds),
        "total_etf": money_to_decimal(portfolio.total_amount_etf),
        "total_currencies": money_to_decimal(portfolio.total_amount_currencies),
        "positions": positions,
    }


async def post_market_order(
    instrument_id: str,
    quantity: int,
    direction: OrderDirection,
    order_id: str | None = None,
) -> PostOrderResponse:
    """
    Выставить рыночную заявку.

    Args:
        instrument_id: FIGI или UID инструмента
        quantity: количество лотов
        direction: ORDER_DIRECTION_BUY или ORDER_DIRECTION_SELL
        order_id: уникальный ID заявки (генерируется автоматически если None)

    Returns:
        PostOrderResponse с order_id, status, executed_order_price
    """
    import uuid
    order_id = order_id or str(uuid.uuid4())

    async with get_client() as client:
        response = await client.orders.post_order(
            instrument_id=instrument_id,
            quantity=quantity,
            direction=direction,
            account_id=ACCOUNT_ID,
            order_type=OrderType.ORDER_TYPE_MARKET,
            order_id=order_id,
        )

    logger.info(
        "Market order posted",
        instrument_id=instrument_id,
        quantity=quantity,
        direction=direction.name,
        order_id=response.order_id,
        status=response.execution_report_status.name,
    )
    return response


async def post_limit_order(
    instrument_id: str,
    quantity: int,
    price: Decimal,
    direction: OrderDirection,
    order_id: str | None = None,
) -> PostOrderResponse:
    """
    Выставить лимитную заявку.

    Args:
        instrument_id: FIGI или UID инструмента
        quantity: количество лотов
        price: цена за единицу
        direction: ORDER_DIRECTION_BUY или ORDER_DIRECTION_SELL
        order_id: уникальный ID заявки

    Returns:
        PostOrderResponse
    """
    import uuid
    from t_tech.invest.utils import decimal_to_quotation

    order_id = order_id or str(uuid.uuid4())

    async with get_client() as client:
        response = await client.orders.post_order(
            instrument_id=instrument_id,
            quantity=quantity,
            price=decimal_to_quotation(price),
            direction=direction,
            account_id=ACCOUNT_ID,
            order_type=OrderType.ORDER_TYPE_LIMIT,
            order_id=order_id,
        )

    logger.info(
        "Limit order posted",
        instrument_id=instrument_id,
        quantity=quantity,
        price=str(price),
        direction=direction.name,
        order_id=response.order_id,
        status=response.execution_report_status.name,
    )
    return response


async def cancel_order(order_id: str) -> None:
    """
    Отменить выставленную заявку.

    Args:
        order_id: ID заявки для отмены
    """
    async with get_client() as client:
        await client.orders.cancel_order(
            account_id=ACCOUNT_ID,
            order_id=order_id,
        )
    logger.info("Order cancelled", order_id=order_id)


async def get_open_orders() -> list[dict]:
    """
    Получить все активные (незакрытые) заявки по счёту.

    Returns:
        Список словарей с информацией о заявках
    """
    async with get_client() as client:
        response = await client.orders.get_orders(account_id=ACCOUNT_ID)

    orders = []
    for order in response.orders:
        orders.append({
            "order_id": order.order_id,
            "figi": order.figi,
            "direction": order.direction.name,
            "order_type": order.order_type.name,
            "status": order.execution_report_status.name,
            "lots_requested": order.lots_requested,
            "lots_executed": order.lots_executed,
            "initial_order_price": money_to_decimal(order.initial_order_price),
        })

    logger.debug("Open orders fetched", count=len(orders))
    return orders
