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
| `ML_LOOKAHEAD` | Горизонт удержания позиции при генерации целевых P&L значений (свечей) | `4` |
| `ML_THRESHOLD` | Fallback порог входа (предсказанный net P&L); per-ticker оптимизируется Optuna → `best_threshold_*.json` | `0.0` |
| `ML_THRESHOLD_N_TRIALS` | Итераций Optuna для подбора порога входа per-ticker | `50` |
| `ML_SHARPE_MIN_TRADES` | Минимум сделок для расчёта Sharpe ratio (меньше — штраф 0.0) | `10` |
| `ML_N_SPLITS` | Максимальное число фолдов `WalkForwardSplit` (берутся последние N ближайших к OOS) | `5` |
| `ML_RANDOM_STATE` | Seed для воспроизводимости (Optuna, ExtraTrees) | `42` |
| `ML_OPTUNA_TRIALS_LGBM` | Итераций Optuna для LightGBM | `75` |
| `ML_OPTUNA_TRIALS_ET` | Итераций Optuna для ExtraTreesRegressor | `40` |
| `ML_OPTUNA_TRIALS_HIST_GBM` | Итераций Optuna для HistGradientBoostingRegressor | `40` |
| `ML_FEATURE_IMPORTANCE_THRESHOLD` | Порог importance для отбора признаков per-ticker (`0.0` = отключить) | `0.01` |
| `ML_PRINT_FEATURE_IMPORTANCE` | Выводить таблицу важности признаков после обучения | `false` |
| `ML_FEATURE_SELECTION_METHOD` | Метод отбора признаков: `"permutation"` (OOS, рекомендуется), `"importance"` (legacy impurity), `"none"` (отключить) | `permutation` |
| `ML_PERMUTATION_N_REPEATS` | Количество повторов перемешивания на признак при `permutation`; больше = стабильнее, но дольше | `10` |
| `ML_FEATURE_TOP_K` | Ограничить топ-N признаков по importance (`0` = не ограничивать, использовать `ML_FEATURE_IMPORTANCE_THRESHOLD`) | `0` |
| `ML_FORCE_TUNE` | Принудительный перезапуск Optuna | `false` |
| `ML_MIN_CANDLES_PREDICT` | Минимум свечей для инференса | `250` |
| `ML_MODEL_CACHE_SIZE` | Макс. число моделей в RAM одновременно (LRU). `1` = ~100 MB, `10` = ~1.1 GB. Увеличивать только при 4+ GB RAM | `1` |
| `ML_AUTOCORR_WINDOW` | Окно роллинговой автокорреляции доходностей (лаг=1): > 0 = импульс, < 0 = возврат к среднему | `8` |
| `ML_PRICE_VOL_CORR_WINDOW` | Окно корреляции доходность × изменение объёма: > 0 = объём подтверждает цену, < 0 = дивергенция | `20` |
| `ML_WILLIAMS_R_PERIOD` | Период Williams %R (TA-Lib WILLR); диапазон [-100, 0]; -20..0 = перекуплен | `14` |
| `ML_CCI_DELTA_PERIOD` | Лаг дельты CCI (баров); аналогично `rsi_delta_4h` / `stoch_k_delta_4h` | `4` |
| `ML_WF_TRAIN_SIZE` | Размер обучающего окна `WalkForwardSplit` (баров); 3000 ≈ 1.7 года при 1h | `3000` |
| `ML_WF_VAL_SIZE` | Размер val-окна `WalkForwardSplit` (баров); 500 ≈ 3 месяца при 1h | `500` |
| `ML_WF_EMBARGO` | Embargo после каждого val-окна (баров); рекомендовано ≥ `ML_LOOKAHEAD` | `4` |
| `ML_ENSEMBLE_WEIGHT_TEMP` | Температура softmax при вычислении адаптивных весов ансамбля по OOS Spearman. `1.0` = умеренная дифференциация; `0.1` = winner-takes-all; `100.0` = равные веса | `1.0` |
| `ML_REGIME_ADX_THRESHOLD` | Порог ADX для определения режима рынка (`market_regime`): ниже порога = флет, выше = тренд. Для часовых свечей: `20`; для дневных: `25` | `20.0` |
| `ML_ROLLING_VWAP_WINDOW` | Окно скользящего VWAP для признака `rolling_vwap_dev` (баров). `20` ≈ 3 торговых часа при 1h-свечах — дополняет `vwap_ratio` (дневной VWAP) | `20` |

## Ночное дообучение

| Переменная | Описание | По умолчанию |
|---|---|---|
| `RETRAIN_ENABLED` | Включить ночное дообучение | `true` |
| `RETRAIN_HOUR` | Час запуска (0–23) | `2` |
| `RETRAIN_MINUTE` | Минута запуска (0–59) | `0` |
| `RETRAIN_TIMEZONE` | Часовой пояс (IANA) | `Europe/Moscow` |
| `RETRAIN_FORCE_TUNE` | Перезапустить Optuna HPO (занимает часы) | `false` |
| `RETRAIN_CATCHUP_HOURS` | Наверстывание: запустить сразу если бот стартовал не позднее N часов после RETRAIN_HOUR | `8` |

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
| `TRADING_CONFIDENCE_THRESHOLD` | Fallback мин. предсказанный net P&L для открытия позиции (per-ticker задаётся Optuna в `best_threshold_*.json`) | `0.0` |
| `TRADING_LOTS_PER_TICKER` | Лотов на каждую сделку (используется при `TRADING_POSITION_SIZING=fixed_lots`) | `1` |
| `TRADING_POSITION_SIZING` | Метод расчёта лотов: `fixed_risk` — масштабировать по риску (рекомендуется), `fixed_lots` — всегда `lots_per_ticker` | `fixed_risk` |
| `TRADING_RISK_PCT_PER_TRADE` | Доля баланса, которую рискуем потерять в одной сделке при срабатывании SL (`0.01` = 1%). Стандартный диапазон: 0.5%–2% | `0.01` |
| `TRADING_MAX_LOTS_PER_TRADE` | Жёсткий максимум лотов на одну сделку (защита от аномальных расчётов) | `10` |
| `TRADING_STOP_LOSS_PCT` | Целевой чистый убыток для стоп-лосса (после комиссий) | `0.03` |
| `TRADING_TAKE_PROFIT_PCT` | Целевая чистая прибыль для тейк-профита (после комиссий и НДФЛ) | `0.05` |
| `TRADING_MAX_POSITIONS` | Макс. одновременных позиций | `5` |
| `TRADING_INTERVAL_SECONDS` | Интервал проверки сигналов (сек) | `3600` |
| `TRADING_CHAT_ID` | Telegram chat_id для уведомлений | — |
| `TRADING_BROKER_COMMISSION_PCT` | Комиссия брокера за сделку (взимается при покупке и продаже). Тариф «Инвестор» = `0.003` (0.3%), тариф «Трейдер» = `0.0005` (0.05%) | `0.0005` |
| `TRADING_TAX_PCT` | НДФЛ на прибыль от продажи | `0.13` |
| `TRADING_DIVIDEND_PROTECTION_DAYS` | Дней защиты SL после экс-даты (глобальный фоллбэк) | `1` |
| `TRADING_DIVIDEND_OVERRIDE` | Ручные окна защиты по тикерам (`SBER:45,GAZP:90`) | — |
| `TRADING_VOLUME_MIN_RATIO` | Мин. `volume_ratio` для подтверждения BUY (`1.0` = отключён) | `1.0` |
| `TRADING_ORDER_POLL_SECONDS` | Интервал polling SL стоп-ордеров в OrderWatcher (сек) | `30` |
| `TRADING_CANDLE_UPDATE_PAUSE_SECONDS` | Пауза между тикерами при внутрисессионном обновлении свечей (сек) | `3` |
| `TRADING_DYNAMIC_SLTP_ENABLED` | Включить динамические SL/TP на основе ATR. При `false` — используются фиксированные `STOP_LOSS_PCT`/`TAKE_PROFIT_PCT` | `true` |
| `TRADING_ATR_SL_MULTIPLIER` | Множитель ATR для расчёта SL: `sl = clamp(atr_ratio × mult, min, max)`. Рекомендуется 1.5–2.5 для часовых свечей | `2.0` |
| `TRADING_ATR_RISK_REWARD_RATIO` | Соотношение RR: `tp = sl × ratio`. `1.67` ≈ 5:3; `2.0` = классический 2:1 | `1.67` |
| `TRADING_ATR_MIN_SL_PCT` | Минимальный SL при ATR-расчёте (защита от узких стопов при тихом рынке) | `0.015` |
| `TRADING_ATR_MAX_SL_PCT` | Максимальный SL при ATR-расчёте (ограничение потерь при кризисной волатильности) | `0.05` |
| `TRADING_REGIME_FILTER_ENABLED` | Включить фильтр рыночного режима при открытии BUY. `false` = отключить (backward compat) | `true` |
| `TRADING_REGIME_FILTER_MODE` | `soft` = уменьшать лоты в флете; `hard` = блокировать BUY в флете и даунтренде | `soft` |
| `TRADING_REGIME_FLAT_MULTIPLIER` | Множитель лотов в флете при режиме `soft`. `0.5` = вдвое меньше; `0.0` = блокировать; `1.0` = без ограничений | `0.5` |

## Приложение

| Переменная | Описание | По умолчанию |
|---|---|---|
| `LOG_LEVEL` | Уровень логирования (`DEBUG`, `INFO`, `WARNING`) | `INFO` |
| `DEBUG` | Режим отладки | `false` |
| `ANTHROPIC_API_KEY` | Ключ Anthropic API (для 3-агентной системы) | — |
| `AGENT_MODEL` | Модель для агентов | `claude-opus-4-6` |
| `AGENT_MAX_REVISIONS` | Максимум раундов ревизии кода (Coder → Reviewer → Coder) | `2` |
