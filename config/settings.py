"""
Настройки проекта, загружаемые из файла .env.
Использует pydantic-settings для валидации и типизации.

Все изменяемые параметры проекта находятся здесь.
Для изменения настроек редактируй .env — код трогать не нужно.
"""
from urllib.parse import quote_plus

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TinkoffSettings(BaseSettings):
    """Параметры подключения к Tinkoff Invest API."""

    token: str = Field(..., alias="TINKOFF_TOKEN")
    account_id: str = Field(..., alias="TINKOFF_ACCOUNT_ID")
    sandbox: bool = Field(True, alias="TINKOFF_SANDBOX")
    # Лимит PostOrder (заявок в секунду). Актуально: 15/сек с февраля 2025.
    post_order_rate: int = Field(15, alias="TINKOFF_POST_ORDER_RATE")
    # Таймаут gRPC-соединения в миллисекундах (keepalive timeout)
    grpc_keepalive_timeout_ms: int = Field(20000, alias="TINKOFF_GRPC_KEEPALIVE_TIMEOUT_MS")

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
    start_date: Optional[str] = Field(None, alias="DATA_START_DATE")
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

    # Горизонт удержания позиции при генерации целей P&L
    lookahead: int = Field(4, alias="ML_LOOKAHEAD")          # свечей вперёд
    # Минимальный предсказанный net P&L для открытия сделки (fallback если нет per-ticker файла).
    # Оптимальное значение подбирается per-ticker через Optuna → best_threshold_{ticker}.json.
    threshold: float = Field(0.0, alias="ML_THRESHOLD")      # 0.0 = любой положительный прогноз

    # Кросс-валидация
    n_splits: int = Field(5, alias="ML_N_SPLITS")             # фолдов TimeSeriesSplit
    random_state: int = Field(42, alias="ML_RANDOM_STATE")

    # Количество итераций Optuna для подбора порога уверенности per-ticker
    threshold_n_trials: int = Field(50, alias="ML_THRESHOLD_N_TRIALS")

    # Минимальное число сделок для расчёта Sharpe ratio.
    # Меньше — возвращает 0.0 (штраф моделям с редкими BUY-сигналами).
    sharpe_min_trades: int = Field(10, alias="ML_SHARPE_MIN_TRADES")

    # Количество итераций Optuna для каждой модели ансамбля
    optuna_trials_lgbm: int = Field(25, alias="ML_OPTUNA_TRIALS_LGBM")
    optuna_trials_et: int = Field(15, alias="ML_OPTUNA_TRIALS_ET")
    optuna_trials_hist_gbm: int = Field(40, alias="ML_OPTUNA_TRIALS_HIST_GBM")

    # Минимум свечей для инференса (50 прогрев + 200 буфер)
    min_candles_predict: int = Field(250, alias="ML_MIN_CANDLES_PREDICT")

    # Отбор признаков per-ticker по нормализованной importance.
    # После быстрого фита на всех признаках удаляются признаки с importance < порога.
    # Каждый тикер получает свой набор и сохраняет его в features_{ticker}_{version}.json.
    # Инференс автоматически использует тикерный набор.
    # 0.0 = отключить отбор (использовать все признаки).
    feature_importance_threshold: float = Field(0.01, alias="ML_FEATURE_IMPORTANCE_THRESHOLD")

    # Метод отбора признаков per-ticker.
    # "permutation" — OOS Permutation Importance на последнем WalkForward-фолде
    #                 (рекомендуется): напрямую измеряет вклад в Spearman, нет утечки.
    # "importance"  — legacy: нормализованная impurity importance по трём моделям ансамбля.
    #                 Быстрее, но нестабильнее и с bias к высококардинальным признакам.
    # "none"        — отбор отключён, используются все признаки.
    feature_selection_method: str = Field("permutation", alias="ML_FEATURE_SELECTION_METHOD")

    # Количество повторов перемешивания на признак при permutation importance.
    # Больше повторов → стабильнее оценка, но дольше (линейно).
    # 10 повторов × 58 признаков × ~500 val-баров ≈ 1–2 мин на тикер.
    permutation_n_repeats: int = Field(10, alias="ML_PERMUTATION_N_REPEATS")

    # Ограничение числа отбираемых признаков (top-N по importance).
    # 0 = не ограничивать — использовать ML_FEATURE_IMPORTANCE_THRESHOLD.
    # > 0 = взять ровно N лучших признаков (игнорирует порог threshold).
    feature_top_k: int = Field(0, alias="ML_FEATURE_TOP_K")

    # Выводить таблицу важности признаков после обучения каждого тикера.
    # Полезно при ручном запуске train_model для анализа. При работе бота
    # (ночное переобучение) вывод не нужен — установить в false.
    print_feature_importance: bool = Field(False, alias="ML_PRINT_FEATURE_IMPORTANCE")

    # Принудительный повтор Optuna при следующем запуске обучения.
    # true — игнорировать кеш best_params_*.json и запустить HPO заново.
    # После использования вернуть в false, иначе HPO будет запускаться каждый раз.
    force_tune: bool = Field(False, alias="ML_FORCE_TUNE")

    # Максимальное число моделей в in-memory LRU-кеше.
    # Тикеры обрабатываются поочерёдно → достаточно 1.
    # Увеличить если нужен параллельный инференс нескольких тикеров.
    model_cache_size: int = Field(1, alias="ML_MODEL_CACHE_SIZE")

    # Параметры новых признаков (режим рынка и подтверждение объёма)
    # Окно для роллинговой автокорреляции доходностей (лаг=1).
    # 8 баров = 8 часов: достаточно для детекции краткосрочного режима (импульс vs возврат к среднему).
    autocorr_window: int = Field(8, alias="ML_AUTOCORR_WINDOW")
    # Окно для роллинговой корреляции доходность × изменение объёма.
    # 20 баров совпадает с другими 20-барными индикаторами (CMF, SMA20) для согласованности.
    price_vol_corr_window: int = Field(20, alias="ML_PRICE_VOL_CORR_WINDOW")
    # Период Williams %R — осциллятор перекупленности/перепроданности на основе диапазона high/low.
    # 14 совпадает со стандартным периодом RSI/CCI/MFI — сопоставимо по масштабу.
    williams_r_period: int = Field(14, alias="ML_WILLIAMS_R_PERIOD")
    # Лаг дельты CCI: направление изменения CCI за N баров.
    # 4 бара совпадает с rsi_delta_4h / stoch_k_delta_4h / macd_hist_delta_4h — единая логика дельт.
    cci_delta_period: int = Field(4, alias="ML_CCI_DELTA_PERIOD")

    # Параметры WalkForwardSplit (роллинговая кросс-валидация)
    # Размер фиксированного обучающего окна (баров).
    # 3000 баров ≈ 1.7 года при 1h-свечах — охватывает несколько режимов рынка,
    # исключает устаревшие данные до структурного разрыва MOEX февраль 2022.
    wf_train_size: int = Field(3000, alias="ML_WF_TRAIN_SIZE")
    # Размер валидационного окна (баров).
    # 500 баров ≈ 3 месяца при 1h-свечах — достаточно для надёжной оценки Spearman.
    wf_val_size: int = Field(500, alias="ML_WF_VAL_SIZE")
    # Embargo после каждого val-окна (баров).
    # Метки последних барóв val зависят от lookahead барóв вперёд — без embargo
    # эти бары попадают в следующее обучающее окно и создают утечку меток.
    # Рекомендованный минимум: = ML_LOOKAHEAD (4). Можно увеличить до max(rolling_window)
    # для дополнительной защиты от rolling-признаков (autocorr_returns=8, price_vol_corr=20).
    wf_embargo: int = Field(4, alias="ML_WF_EMBARGO")

    # Температура softmax при вычислении адаптивных весов ансамбля по Spearman OOS.
    # Веса вычисляются на последнем walk-forward фолде: w_i = softmax(Spearman_i / temp).
    # temp→∞ (например 100.0): равные веса (поведение как до внедрения фичи).
    # temp→0 (например 0.1): winner-takes-all (вся масса у лучшей модели).
    # Рекомендованный диапазон: [0.5, 2.0]. По умолчанию 1.0 — умеренная дифференциация.
    ensemble_weight_temp: float = Field(1.0, alias="ML_ENSEMBLE_WEIGHT_TEMP")

    # Порог ADX для определения режима рынка (торговый фильтр market_regime).
    # ADX < порога → флет (нет выраженного тренда); ADX >= порога → тренд.
    # Стандарт для часовых данных: 20. Для дневных: 25.
    # Используется в compute_features() при вычислении колонки market_regime.
    regime_adx_threshold: float = Field(20.0, alias="ML_REGIME_ADX_THRESHOLD")

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
    # Минимальный предсказанный net P&L для открытия позиции (fallback, если нет per-ticker файла).
    # Регрессор возвращает ожидаемую чистую доходность сделки (доля, например 0.005 = 0.5%).
    # Значение 0.0 = входим при любом положительном прогнозе; реальные пороги ~0.003..0.015.
    confidence_threshold: float = Field(0.0, alias="TRADING_CONFIDENCE_THRESHOLD")
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
    # Интервал быстрого polling SL стоп-ордеров в OrderWatcher (секунды).
    # Чем меньше — тем быстрее обнаруживается срабатывание SL, но больше API-запросов.
    order_poll_seconds: int = Field(30, alias="TRADING_ORDER_POLL_SECONDS")
    # Пауза между тикерами при внутрисессионном инкрементальном сборе свечей (секунды).
    # Для ночного переобучения используется DATA_COLLECT_PAUSE_SECONDS (дольше, но безопаснее).
    # Здесь загружается 1–2 свечи на тикер — короткая пауза не вызывает RESOURCE_EXHAUSTED.
    candle_update_pause_seconds: int = Field(3, alias="TRADING_CANDLE_UPDATE_PAUSE_SECONDS")
    # Динамические SL/TP на основе ATR-волатильности.
    # При enabled=true sl_pct = clamp(atr_ratio × multiplier, min_sl, max_sl),
    # tp_pct = sl_pct × risk_reward_ratio.
    # При enabled=false или atr_ratio=0 — используются фиксированные TRADING_STOP_LOSS_PCT / TRADING_TAKE_PROFIT_PCT.
    dynamic_sltp_enabled: bool = Field(True, alias="TRADING_DYNAMIC_SLTP_ENABLED")
    # Множитель ATR для расчёта SL: sl_pct = atr_ratio × multiplier.
    # Рекомендуется 1.5–2.5 для часовых свечей (дневная/свинг-торговля).
    atr_sl_multiplier: float = Field(2.0, alias="TRADING_ATR_SL_MULTIPLIER")
    # Соотношение риск/прибыль: tp_pct = sl_pct × ratio.
    # 1.67 ≈ RR 5:3 (при SL=3% → TP=5%); 2.0 = RR 2:1.
    atr_risk_reward_ratio: float = Field(1.67, alias="TRADING_ATR_RISK_REWARD_RATIO")
    # Минимальный SL при динамическом расчёте (защита от слишком узких стопов при тихом рынке).
    atr_min_sl_pct: float = Field(0.015, alias="TRADING_ATR_MIN_SL_PCT")
    # Максимальный SL при динамическом расчёте (защита от чрезмерных убытков при кризисе).
    atr_max_sl_pct: float = Field(0.05, alias="TRADING_ATR_MAX_SL_PCT")

    # ── Управление размером позиции (Position Sizing) ───────────────────────────
    # Метод расчёта количества лотов на сделку:
    # 'fixed_risk' — лоты масштабируются так, чтобы при срабатывании SL потеря
    #                была ровно risk_pct_per_trade × balance (рекомендуется).
    # 'fixed_lots' — всегда lots_per_ticker лотов (старое поведение, обратная совместимость).
    position_sizing: str = Field("fixed_risk", alias="TRADING_POSITION_SIZING")
    # Максимальная доля баланса, которую рискуем потерять в одной сделке.
    # При срабатывании SL фактический убыток = balance × risk_pct_per_trade.
    # Стандартный диапазон: 0.005 (0.5%) — 0.02 (2%). Рекомендуется 0.01 (1%).
    risk_pct_per_trade: float = Field(0.01, alias="TRADING_RISK_PCT_PER_TRADE")
    # Жёсткий максимум лотов на одну сделку.
    # Защита от аномальных расчётов при очень большом балансе или очень узком SL.
    max_lots_per_trade: int = Field(10, alias="TRADING_MAX_LOTS_PER_TRADE")

    # ── Фильтр рыночного режима (market_regime) ─────────────────────────────
    # Включить фильтр режима рынка при открытии BUY-позиций.
    # При false — фильтр отключён, лоты не изменяются (backward compatibility).
    regime_filter_enabled: bool = Field(True, alias="TRADING_REGIME_FILTER_ENABLED")
    # Режим фильтра:
    #   "soft" — в даунтренде BUY блокируется (lots=0), в флете лоты × multiplier.
    #   "hard" — и в даунтренде, и в флете BUY блокируется полностью (lots=0).
    regime_filter_mode: str = Field("soft", alias="TRADING_REGIME_FILTER_MODE")
    # Множитель лотов в режиме флета при "soft"-фильтре.
    # 0.0 = полная блокировка в флете (идентично "hard" для флета).
    # 0.5 = открывать вдвое меньше лотов (рекомендуется — сохраняем часть сигналов).
    # 1.0 = флет не ограничивает (фильтрует только даунтренд).
    regime_flat_lots_multiplier: float = Field(0.5, alias="TRADING_REGIME_FLAT_MULTIPLIER")

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
    # Окно наверстывания (часов): если бот запустился после RETRAIN_HOUR,
    # но не позднее чем через N часов — запустить дообучение сразу.
    catchup_hours: int = Field(8, alias="RETRAIN_CATCHUP_HOURS")

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
