"""
Синглтон async Redis-клиента для кеширования.

Использование:
    from utils.redis_cache import get_redis, init_redis, close_redis

    # При старте приложения:
    await init_redis()

    # В коде:
    redis = await get_redis()
    if redis is not None:
        cached = await redis.get("key")

    # При остановке:
    await close_redis()

Если Redis недоступен — get_redis() возвращает None (graceful degradation):
вызывающий код пропускает кеш и идёт напрямую к БД / API.
"""
from redis.asyncio import Redis, from_url

from config.settings import redis_settings
from utils.logger import logger

_client: Redis | None = None


async def init_redis() -> None:
    """
    Инициализировать Redis-клиент и проверить соединение (ping).

    При ошибке подключения логирует предупреждение и оставляет клиент None
    (бот работает без кеширования).
    """
    global _client
    try:
        client = from_url(redis_settings.url, decode_responses=True)
        await client.ping()
        _client = client
        logger.info(
            "Redis инициализирован",
            host=redis_settings.host,
            port=redis_settings.port,
            db=redis_settings.db,
        )
    except Exception as e:
        logger.warning(
            "Redis недоступен — кеш отключён, работаем без кеширования",
            error=str(e),
        )
        _client = None


async def close_redis() -> None:
    """Закрыть соединение с Redis при завершении приложения."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("Redis соединение закрыто")


async def get_redis() -> Redis | None:
    """
    Получить Redis-клиент.

    Возвращает:
        Redis-клиент или None если Redis не инициализирован / недоступен.
        При None вызывающий код должен пропустить кеш (graceful degradation).
    """
    return _client
