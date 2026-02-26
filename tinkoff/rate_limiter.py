"""
Ограничитель частоты запросов к Tinkoff Invest API.

Реализует алгоритм Token Bucket для соблюдения лимитов API.

Актуальные лимиты Tinkoff Invest API (февраль 2025):
    - PostOrder:      15 заявок в секунду (900 в минуту)
    - PostOrderAsync: без ограничений данным лимитом

Источник: официальный Telegram-канал Tinkoff Invest API.
"""
import asyncio
import time

from utils.logger import logger


class TokenBucketRateLimiter:
    """
    Асинхронный ограничитель частоты запросов по алгоритму Token Bucket.

    Позволяет до `rate` вызовов за `period` секунд.
    При превышении лимита — ожидает ровно столько, чтобы не нарушить ограничение.
    Потокобезопасен через asyncio.Lock.
    """

    def __init__(self, rate: int, period: float = 1.0) -> None:
        """
        Аргументы:
            rate:   максимальное количество вызовов за период.
            period: длина периода в секундах (по умолчанию 1.0 = в секунду).
        """
        self._rate = rate
        self._period = period
        self._tokens: float = float(rate)   # начинаем с полным ведром
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Занять один токен, при необходимости подождать.

        Вызывать перед каждым защищённым API-запросом.
        """
        async with self._lock:
            await self._refill_and_wait()

    async def _refill_and_wait(self) -> None:
        """Пополнить токены по прошедшему времени и при необходимости подождать."""
        now = time.monotonic()
        elapsed = now - self._last_refill

        # Пополняем токены пропорционально прошедшему времени
        refill = elapsed * (self._rate / self._period)
        self._tokens = min(float(self._rate), self._tokens + refill)
        self._last_refill = now

        if self._tokens < 1.0:
            # Ждём ровно столько, чтобы накопился 1 токен
            wait_time = (1.0 - self._tokens) * (self._period / self._rate)
            logger.debug(
                "Rate limit reached, waiting",
                wait_seconds=round(wait_time, 3),
                rate=self._rate,
                period=self._period,
            )
            await asyncio.sleep(wait_time)
            self._tokens = 0.0
        else:
            self._tokens -= 1.0


# ── Синглтоны для каждого метода API ────────────────────────────────────────

# PostOrder: 15 заявок в секунду (лимит с февраля 2025)
post_order_limiter = TokenBucketRateLimiter(rate=15, period=1.0)
