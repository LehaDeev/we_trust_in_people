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
    # Глубина истории при первом запуске (дней) — используется если DATA_START_DATE не задана
    history_days: int = Field(365, alias="DATA_HISTORY_DAYS")
    # Фиксированная дата начала истории (формат YYYY-MM-DD).
    # Если задана — новые тикеры загружают данные именно с этой даты,
    # игнорируя DATA_HISTORY_DAYS. Пустая строка = не используется.
    start_date: str = Field("", alias="DATA_START_DATE")
    # FIGI инструмента USD/RUB (USDRUB_TOM на MOEX) — используется как ML-признак
    usdrub_figi: str = Field("BBG0013HGFT4", alias="DATA_USDRUB_FIGI")
    # Пауза между тикерами при сборе свечей (секунды). При больших объёмах данных
    # (DATA_HISTORY_DAYS > 730) API выдаёт RESOURCE_EXHAUSTED — увеличить до 30-60.
    collect_pause_seconds: int = Field(15, alias="DATA_COLLECT_PAUSE_SECONDS")

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
    optuna_trials_et: int = Field(30, alias="ML_OPTUNA_TRIALS_ET")
    # SVC медленнее деревьев из-за Platt scaling — держать <= 20 трайлов
    optuna_trials_svc: int = Field(20, alias="ML_OPTUNA_TRIALS_SVC")
    optuna_trials_catboost: int = Field(30, alias="ML_OPTUNA_TRIALS_CATBOOST")

    # Минимум свечей для инференса (50 прогрев + 200 буфер)
    min_candles_predict: int = Field(250, alias="ML_MIN_CANDLES_PREDICT")

    # Отбор признаков per-ticker по нормализованной importance.
    # После быстрого фита на всех признаках удаляются признаки с importance < порога.
    # Каждый тикер получает свой набор и сохраняет его в features_{ticker}_{version}.json.
    # Инференс автоматически использует тикерный набор.
    # 0.0 = отключить отбор (использовать все признаки).
    feature_importance_threshold: float = Field(0.01, alias="ML_FEATURE_IMPORTANCE_THRESHOLD")

    # Выводить таблицу важности признаков после обучения каждого тикера.
    # Полезно при ручном запуске train_model для анализа. При работе бота
    # (ночное переобучение) вывод не нужен — установить в false.
    print_feature_importance: bool = Field(False, alias="ML_PRINT_FEATURE_IMPORTANCE")

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
    dividend_ttl: int = Field(86400, alias="REDIS_DIVIDEND_TTL") # дивидендные данные (24 часа)

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


class TradingSettings(BaseSettings):
    """
    Параметры автоматической торговли.

    По умолчанию TRADING_ENABLED=false — торговля выключена для безопасности.
    Включить только после проверки всех параметров в .env.
    """

    # Главный выключатель автоторговли (false = только логируем, не торгуем)
    enabled: bool = Field(False, alias="TRADING_ENABLED")
    # Минимальная уверенность модели для открытия позиции (0.0 - 1.0)
    confidence_threshold: float = Field(0.65, alias="TRADING_CONFIDENCE_THRESHOLD")
    # Количество лотов на каждую сделку
    lots_per_ticker: int = Field(1, alias="TRADING_LOTS_PER_TICKER")
    # Стоп-лосс: закрыть позицию при убытке (доля от цены входа, 0.03 = 3%)
    stop_loss_pct: float = Field(0.03, alias="TRADING_STOP_LOSS_PCT")
    # Тейк-профит: закрыть позицию при прибыли (доля от цены входа, 0.05 = 5%)
    take_profit_pct: float = Field(0.05, alias="TRADING_TAKE_PROFIT_PCT")
    # Максимальное количество одновременно открытых позиций
    max_open_positions: int = Field(5, alias="TRADING_MAX_POSITIONS")
    # Интервал проверки сигналов и открытых позиций (секунды, 3600 = 1 час)
    check_interval_seconds: int = Field(3600, alias="TRADING_INTERVAL_SECONDS")
    # Telegram chat_id для уведомлений о сделках (пустая строка = уведомления отключены)
    notification_chat_id: str = Field("", alias="TRADING_CHAT_ID")
    # Комиссия брокера/биржи за сделку (доля от суммы, 0.003 = 0.3%)
    broker_commission_pct: float = Field(0.003, alias="TRADING_BROKER_COMMISSION_PCT")
    # Ставка НДФЛ на прибыль от продажи (0.13 = 13%, 0.15 = 15% при доходе >5 млн)
    tax_pct: float = Field(0.13, alias="TRADING_TAX_PCT")
    # Защита от дивидендного гэпа: корректировать SL на размер дивиденда
    # в течение этого количества дней вокруг экс-дивидендной даты (0 = отключено)
    dividend_protection_days: int = Field(1, alias="TRADING_DIVIDEND_PROTECTION_DAYS")
    # Ручное переопределение окна защиты для конкретных тикеров.
    # Формат: TICKER:дни,TICKER:дни (например SBER:45,GAZP:90).
    # Эти значения имеют приоритет над авто-вычислением из истории.
    dividend_override_raw: str = Field("", alias="TRADING_DIVIDEND_OVERRIDE")
    # Минимальный volume_ratio для подтверждения BUY-сигнала.
    # volume_ratio = объём последнего бара / SMA_20(объём).
    # 1.0 = фильтр отключён (любой объём); 1.3 = объём должен быть на 30% выше среднего.
    volume_min_ratio: float = Field(1.0, alias="TRADING_VOLUME_MIN_RATIO")

    @property
    def dividend_override(self) -> dict[str, int]:
        """Парсит SBER:45,GAZP:90 → {"SBER": 45, "GAZP": 90}."""
        result: dict[str, int] = {}
        for item in self.dividend_override_raw.split(","):
            item = item.strip()
            if ":" not in item:
                continue
            ticker, days_str = item.split(":", 1)
            try:
                result[ticker.strip().upper()] = int(days_str.strip())
            except ValueError:
                pass
        return result

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RetrainSettings(BaseSettings):
    """
    Параметры ночного дообучения ML-моделей.

    Каждую ночь в RETRAIN_HOUR:RETRAIN_MINUTE (по московскому времени)
    бот инкрементально собирает новые свечи из Tinkoff API и переобучает
    per-ticker ансамбли на всех накопленных данных.
    Optuna HPO при этом пропускается — используются кешированные параметры.
    """

    # Включить/выключить ночное дообучение
    enabled: bool = Field(True, alias="RETRAIN_ENABLED")
    # Час запуска по RETRAIN_TIMEZONE (0–23)
    hour: int = Field(2, alias="RETRAIN_HOUR")
    # Минута запуска (0–59)
    minute: int = Field(0, alias="RETRAIN_MINUTE")
    # Часовой пояс по имени IANA (например "Europe/Moscow")
    timezone: str = Field("Europe/Moscow", alias="RETRAIN_TIMEZONE")
    # Принудительный перезапуск Optuna HPO при ночном переобучении.
    # Оставить false — HPO занимает часы, нецелесообразно каждую ночь.
    force_tune: bool = Field(False, alias="RETRAIN_FORCE_TUNE")

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
trading_settings = TradingSettings()
retrain_settings = RetrainSettings()
