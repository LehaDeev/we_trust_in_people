"""
Настройки проекта, загружаемые из файла .env.
Использует pydantic-settings для валидации и типизации.

Все изменяемые параметры проекта находятся здесь.
Для изменения настроек редактируй .env — код трогать не нужно.
"""
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TinkoffSettings(BaseSettings):
    """Параметры подключения к Tinkoff Invest API."""

    token: str = Field(..., alias="TINKOFF_TOKEN")
    account_id: str = Field(..., alias="TINKOFF_ACCOUNT_ID")
    sandbox: bool = Field(True, alias="TINKOFF_SANDBOX")
    # Лимит PostOrder (заявок в секунду). Актуально: 15/сек с февраля 2025.
    post_order_rate: int = Field(15, alias="TINKOFF_POST_ORDER_RATE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class TelegramSettings(BaseSettings):
    """Параметры Telegram-бота."""

    bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class PostgresSettings(BaseSettings):
    """Параметры подключения к PostgreSQL."""

    host: str = Field("localhost", alias="POSTGRES_HOST")
    port: int = Field(5432, alias="POSTGRES_PORT")
    db: str = Field("we_trust_db", alias="POSTGRES_DB")
    user: str = Field("postgres", alias="POSTGRES_USER")
    password: str = Field(..., alias="POSTGRES_PASSWORD")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def dsn(self) -> str:
        """Async DSN для SQLAlchemy + asyncpg (спецсимволы URL-encoded)."""
        pwd = quote_plus(self.password)
        return (
            f"postgresql+asyncpg://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def asyncpg_dsn(self) -> str:
        """Raw asyncpg DSN (без префикса SQLAlchemy, спецсимволы URL-encoded)."""
        pwd = quote_plus(self.password)
        return (
            f"postgresql://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class DataSettings(BaseSettings):
    """
    Настройки сбора рыночных данных.

    Пример .env:
        DATA_TICKERS=SBER,GAZP,LKOH,YDEX,NVTK
        DATA_CANDLE_INTERVAL=1h
        DATA_HISTORY_DAYS=365
    """

    # Тикеры через запятую: SBER,GAZP,LKOH,...
    # str, а не list[str] — pydantic-settings 2.x пытается JSON-декодировать list-поля,
    # что ломается на строках вида "SBER,GAZP". Список отдаётся через property ниже.
    tickers_raw: str = Field(
        default="SBER,GAZP,LKOH,YDEX,NVTK,GMKN,MGNT,TATN,ROSN,MTSS",
        alias="DATA_TICKERS",
    )
    # Интервал свечей: 1min, 5min, 15min, 1h, 1d, 1w
    candle_interval: str = Field("1h", alias="DATA_CANDLE_INTERVAL")
    # Глубина истории при первом запуске (дней)
    history_days: int = Field(365, alias="DATA_HISTORY_DAYS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def tickers(self) -> list[str]:
        """Список тикеров, распарсенный из строки с запятыми."""
        return [t.strip() for t in self.tickers_raw.split(",") if t.strip()]


class MLSettings(BaseSettings):
    """
    Настройки ML-pipeline: параметры меток, кросс-валидации и Optuna.

    Пример .env:
        ML_MODEL_VERSION=v2
        ML_LOOKAHEAD=4
        ML_THRESHOLD=0.01
        ML_OPTUNA_TRIALS_LGBM=50
    """

    # Версия модели — суффикс для имён файлов весов (ensemble_v2.pkl)
    model_version: str = Field("v2", alias="ML_MODEL_VERSION")

    # Параметры генерации меток
    lookahead: int = Field(4, alias="ML_LOOKAHEAD")          # свечей вперёд
    threshold: float = Field(0.01, alias="ML_THRESHOLD")     # порог ±1% BUY/SELL

    # Кросс-валидация
    n_splits: int = Field(5, alias="ML_N_SPLITS")             # фолдов TimeSeriesSplit
    random_state: int = Field(42, alias="ML_RANDOM_STATE")

    # Количество итераций Optuna для каждой модели
    optuna_trials_lgbm: int = Field(50, alias="ML_OPTUNA_TRIALS_LGBM")
    optuna_trials_xgb: int = Field(50, alias="ML_OPTUNA_TRIALS_XGB")
    optuna_trials_rf: int = Field(30, alias="ML_OPTUNA_TRIALS_RF")

    # Минимум свечей для инференса (50 прогрев + 200 буфер)
    min_candles_predict: int = Field(250, alias="ML_MIN_CANDLES_PREDICT")

    # Принудительный повтор Optuna при следующем запуске обучения.
    # true — игнорировать кеш best_params_*.json и запустить HPO заново.
    # После использования вернуть в false, иначе HPO будет запускаться каждый раз.
    force_tune: bool = Field(False, alias="ML_FORCE_TUNE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class AppSettings(BaseSettings):
    """Общие настройки приложения."""

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    debug: bool = Field(False, alias="DEBUG")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RedisSettings(BaseSettings):
    """
    Параметры Redis-кеша.

    Все TTL в секундах. При REDIS_PASSWORD="" — подключение без аутентификации.
    """

    host: str = Field("localhost", alias="REDIS_HOST")
    port: int = Field(6379, alias="REDIS_PORT")
    db: int = Field(0, alias="REDIS_DB")
    # Пустая строка = без пароля (Redis по умолчанию без auth)
    password: str = Field("", alias="REDIS_PASSWORD")

    # TTL кешей (секунды)
    signal_ttl: int = Field(60, alias="REDIS_SIGNAL_TTL")        # сигнал BUY/SELL/HOLD
    candles_ttl: int = Field(300, alias="REDIS_CANDLES_TTL")     # свечи из БД
    portfolio_ttl: int = Field(60, alias="REDIS_PORTFOLIO_TTL")  # портфель из Tinkoff API
    price_ttl: int = Field(30, alias="REDIS_PRICE_TTL")          # последние цены

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def url(self) -> str:
        """Redis URL для подключения через redis.asyncio.from_url()."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class AgentSettings(BaseSettings):
    """
    Параметры 3-агентной системы (Coder / Reviewer / Architect).

    Требует ANTHROPIC_API_KEY в .env.
    При пустом api_key агенты поднимают ValueError при первом вызове.
    """

    # API-ключ Anthropic (обязателен для работы агентов)
    api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    # Модель для всех трёх агентов
    model: str = Field("claude-opus-4-6", alias="AGENT_MODEL")
    # Максимум раундов ревизии кода (Coder → Reviewer → Coder → ...)
    max_revision_rounds: int = Field(2, alias="AGENT_MAX_REVISIONS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Синглтоны — импортировать в других модулях
tinkoff_settings = TinkoffSettings()
telegram_settings = TelegramSettings()
postgres_settings = PostgresSettings()
data_settings = DataSettings()
ml_settings = MLSettings()
app_settings = AppSettings()
redis_settings = RedisSettings()
agent_settings = AgentSettings()
