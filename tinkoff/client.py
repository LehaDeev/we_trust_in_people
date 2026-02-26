"""
Async gRPC клиент Tinkoff Invest API.

Использование:
    async with get_client() as client:
        accounts = await client.users.get_accounts()

Или через синглтон:
    client = await TinkoffClient.get()
    async with client as c:
        ...
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from t_tech.invest import AsyncClient
from t_tech.invest.async_services import AsyncServices

from config.settings import tinkoff_settings
from utils.logger import logger


@asynccontextmanager
async def get_client() -> AsyncGenerator[AsyncServices, None]:
    """
    Async context manager для работы с Tinkoff API.

    Пример:
        async with get_client() as client:
            candles = await client.market_data.get_candles(...)
    """
    async with AsyncClient(token=tinkoff_settings.token) as client:
        logger.debug("Tinkoff gRPC client connected")
        try:
            yield client
        finally:
            logger.debug("Tinkoff gRPC client disconnected")
