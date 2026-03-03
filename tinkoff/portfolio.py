"""
Операции с портфелем и ордерами через Tinkoff Invest API.
Баланс, позиции, выставление и отмена заявок.

Лимиты API (актуально с февраля 2025):
    - PostOrder:      15 заявок/сек — ограничен через rate_limiter.post_order_limiter
    - PostOrderAsync: без ограничений данным лимитом

Кеширование:
    get_portfolio_summary() кешируется в Redis на REDIS_PORTFOLIO_TTL секунд.
    При недоступном Redis — прямой вызов Tinkoff API (graceful degradation).
"""
import json
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

from config.settings import redis_settings, tinkoff_settings
from tinkoff.client import get_client
from tinkoff.rate_limiter import post_order_limiter
from utils.logger import logger
from utils.redis_cache import get_redis

ACCOUNT_ID = tinkoff_settings.account_id


async def get_portfolio() -> PortfolioResponse:
    """
    Получить текущий портфель (позиции + стоимость).

    Возвращает:
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

    Возвращает:
        PositionsResponse с securities, futures, options, currencies, money
    """
    async with get_client() as client:
        positions = await client.operations.get_positions(
            account_id=ACCOUNT_ID,
        )
    return positions


async def get_rub_balance() -> Decimal:
    """
    Получить доступный остаток средств в рублях.

    Использует get_positions() — не кешируется, всегда актуальный баланс.

    Возвращает:
        Остаток в рублях или Decimal("0") если рублёвого остатка нет.
    """
    positions = await get_positions()
    for money in positions.money:
        if getattr(money, "currency", "").lower() == "rub":
            return money_to_decimal(money)
    logger.warning("RUB balance not found in positions")
    return Decimal("0")


async def get_portfolio_summary() -> dict:
    """
    Краткая сводка по портфелю в удобном формате.

    Результат кешируется в Redis на REDIS_PORTFOLIO_TTL секунд.
    При недоступном Redis — прямой вызов Tinkoff API (graceful degradation).

    Возвращает:
        Словарь с суммами по типам активов и списком позиций
    """
    # ── Redis: проверяем кеш ─────────────────────────────────────────────────
    cache_key = f"portfolio:{ACCOUNT_ID}"
    redis = await get_redis()
    if redis is not None:
        try:
            cached = await redis.get(cache_key)
            if cached:
                logger.debug("Portfolio cache hit", key=cache_key)
                return json.loads(cached, parse_float=lambda x: Decimal(x))
        except Exception as e:
            logger.warning("Redis get error", key=cache_key, error=str(e))

    # ── Запрос к Tinkoff API ─────────────────────────────────────────────────
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

    summary = {
        "total_shares": money_to_decimal(portfolio.total_amount_shares),
        "total_bonds": money_to_decimal(portfolio.total_amount_bonds),
        "total_etf": money_to_decimal(portfolio.total_amount_etf),
        "total_currencies": money_to_decimal(portfolio.total_amount_currencies),
        "positions": positions,
    }

    # ── Redis: сохраняем в кеш (Decimal сериализуем как строку) ──────────────
    if redis is not None:
        try:
            await redis.setex(
                cache_key,
                redis_settings.portfolio_ttl,
                json.dumps(summary, default=str),
            )
            logger.debug("Portfolio cached", ttl=redis_settings.portfolio_ttl)
        except Exception as e:
            logger.warning("Redis setex error", key=cache_key, error=str(e))

    return summary


async def post_market_order(
    instrument_id: str,
    quantity: int,
    direction: OrderDirection,
    order_id: str | None = None,
) -> PostOrderResponse:
    """
    Выставить рыночную заявку.

    Соблюдает лимит PostOrder: 15 заявок/сек — при превышении автоматически ждёт.

    Аргументы:
        instrument_id: FIGI или UID инструмента
        quantity:      количество лотов
        direction:     ORDER_DIRECTION_BUY или ORDER_DIRECTION_SELL
        order_id:      уникальный ID заявки (генерируется автоматически если None)

    Возвращает:
        PostOrderResponse с order_id, status, executed_order_price
    """
    import uuid
    order_id = order_id or str(uuid.uuid4())

    # Ожидаем разрешения от ограничителя (15 заявок/сек)
    await post_order_limiter.acquire()

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

    Соблюдает лимит PostOrder: 15 заявок/сек — при превышении автоматически ждёт.

    Аргументы:
        instrument_id: FIGI или UID инструмента
        quantity:      количество лотов
        price:         цена за единицу
        direction:     ORDER_DIRECTION_BUY или ORDER_DIRECTION_SELL
        order_id:      уникальный ID заявки

    Возвращает:
        PostOrderResponse
    """
    import uuid
    from t_tech.invest.utils import decimal_to_quotation

    order_id = order_id or str(uuid.uuid4())

    # Ожидаем разрешения от ограничителя (15 заявок/сек)
    await post_order_limiter.acquire()

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

    Аргументы:
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

    Возвращает:
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
