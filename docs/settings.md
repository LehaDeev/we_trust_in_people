# Настройки `.env`

Все параметры настраиваются через `.env` (скопировать из `.env.example`).

## Tinkoff API

| Переменная | Описание | По умолчанию |
|---|---|---|
| `TINKOFF_TOKEN` | API-токен Tinkoff Invest | — |
| `TINKOFF_ACCOUNT_ID` | ID счёта | — |
| `TINKOFF_SANDBOX` | Режим песочницы | `false` |
| `TINKOFF_POST_ORDER_RATE` | Лимит PostOrder (заявок/сек) | `15` |
| `TINKOFF_GRPC_KEEPALIVE_TIMEOUT_MS` | Таймаут gRPC keepalive (мс) | `20000` |

## Сбор данных

| Переменная | Описание | По умолчанию |
|---|---|---|
| `DATA_TICKERS` | Тикеры через запятую | `SBER,GAZP,...` |
| `DATA_CANDLE_INTERVAL` | Интервал свечей (`1h`, `1d`, `15min`) | `1h` |
| `DATA_HISTORY_DAYS` | Глубина истории при первом запуске | `365` |
| `DATA_START_DATE` | Фиксированная дата начала истории (`YYYY-MM-DD`); если задана — игнорирует `DATA_HISTORY_DAYS` | — |
| `DATA_USDRUB_FIGI` | FIGI инструмента USD/RUB | `BBG0013HGFT4` |
| `DATA_COLLECT_PAUSE_SECONDS` | Пауза между тикерами при сборе (сек) | `15` |

## ML

| Переменная | Описание | По умолчанию |
|---|---|---|
| `ML_MODEL_VERSION` | Суффикс файлов весов | `v2` |
| `ML_LOOKAHEAD` | Свечей вперёд для генерации меток | `8` |
| `ML_THRESHOLD` | Порог доходности ±% для BUY/SELL | `0.007` |
| `ML_OPTUNA_TRIALS_LGBM` | Итераций Optuna для LightGBM | `50` |
| `ML_OPTUNA_TRIALS_ET` | Итераций Optuna для ExtraTreesClassifier | `30` |
| `ML_OPTUNA_TRIALS_SVC` | Итераций Optuna для SVC (Platt scaling — держать ≤ 20) | `20` |
| `ML_FEATURE_IMPORTANCE_THRESHOLD` | Порог importance для отбора признаков per-ticker (`0.0` = отключить) | `0.01` |
| `ML_PRINT_FEATURE_IMPORTANCE` | Выводить таблицу важности признаков после обучения | `false` |
| `ML_FORCE_TUNE` | Принудительный перезапуск Optuna | `false` |
| `ML_MIN_CANDLES_PREDICT` | Минимум свечей для инференса | `250` |

## Ночное дообучение

| Переменная | Описание | По умолчанию |
|---|---|---|
| `RETRAIN_ENABLED` | Включить ночное дообучение | `true` |
| `RETRAIN_HOUR` | Час запуска (0–23) | `2` |
| `RETRAIN_MINUTE` | Минута запуска (0–59) | `0` |
| `RETRAIN_TIMEZONE` | Часовой пояс (IANA) | `Europe/Moscow` |
| `RETRAIN_FORCE_TUNE` | Перезапустить Optuna HPO (занимает часы) | `false` |

## Redis

| Переменная | Описание | По умолчанию |
|---|---|---|
| `REDIS_HOST` | Хост Redis | `localhost` |
| `REDIS_PORT` | Порт Redis | `6379` |
| `REDIS_DB` | Номер базы Redis | `0` |
| `REDIS_PASSWORD` | Пароль (пустая строка = без аутентификации) | — |
| `REDIS_SIGNAL_TTL` | TTL кеша сигналов (сек) | `60` |
| `REDIS_CANDLES_TTL` | TTL кеша свечей (сек) | `300` |
| `REDIS_PORTFOLIO_TTL` | TTL кеша портфеля (сек) | `60` |
| `REDIS_PRICE_TTL` | TTL кеша цен (сек) | `30` |
| `REDIS_DIVIDEND_TTL` | TTL кеша дивидендных выплат (сек) | `86400` |

## Автоторговля

| Переменная | Описание | По умолчанию |
|---|---|---|
| `TRADING_ENABLED` | Включить автоторговлю | `false` |
| `TRADING_CONFIDENCE_THRESHOLD` | Мин. уверенность модели для BUY | `0.65` |
| `TRADING_LOTS_PER_TICKER` | Лотов на каждую сделку | `1` |
| `TRADING_STOP_LOSS_PCT` | Целевой чистый убыток для стоп-лосса (после комиссий) | `0.03` |
| `TRADING_TAKE_PROFIT_PCT` | Целевая чистая прибыль для тейк-профита (после комиссий и НДФЛ) | `0.05` |
| `TRADING_MAX_POSITIONS` | Макс. одновременных позиций | `5` |
| `TRADING_INTERVAL_SECONDS` | Интервал проверки сигналов (сек) | `3600` |
| `TRADING_CHAT_ID` | Telegram chat_id для уведомлений | — |
| `TRADING_BROKER_COMMISSION_PCT` | Комиссия брокера за сделку | `0.003` |
| `TRADING_TAX_PCT` | НДФЛ на прибыль от продажи | `0.13` |
| `TRADING_DIVIDEND_PROTECTION_DAYS` | Дней защиты SL после экс-даты (глобальный фоллбэк) | `1` |
| `TRADING_DIVIDEND_OVERRIDE` | Ручные окна защиты по тикерам (`SBER:45,GAZP:90`) | — |
| `TRADING_VOLUME_MIN_RATIO` | Мин. `volume_ratio` для подтверждения BUY (`1.0` = отключён) | `1.0` |

## Приложение

| Переменная | Описание | По умолчанию |
|---|---|---|
| `LOG_LEVEL` | Уровень логирования (`DEBUG`, `INFO`, `WARNING`) | `INFO` |
| `DEBUG` | Режим отладки | `false` |
| `ANTHROPIC_API_KEY` | Ключ Anthropic API (для 3-агентной системы) | — |
| `AGENT_MODEL` | Модель для агентов | `claude-opus-4-6` |
