# We Trust in People — Торговый бот

Автоматизированный торговый помощник для Tinkoff Инвестиции с ML-сигналами,
авто/ручной торговлей и учётом комиссий и налогов.

## Возможности

- **Tinkoff Invest API** — асинхронная интеграция, боевой режим, rate limiter
- **Сбор рыночных данных** — исторические свечи по списку тикеров, инкрементальное обновление
- **ML-ансамбль** — LightGBM + XGBoost + RandomForest с Optuna HPO, сигналы BUY / SELL / HOLD
- **Redis-кеш** — свечи, цены, портфель, сигналы; graceful degradation при недоступности Redis
- **Telegram-бот** — полностью inline-интерфейс (без ReplyKeyboard)
- **Портфель** — сводка по счёту + детализация по категориям (акции / облигации / ETF / валюта) с количеством и средней ценой покупки
- **Автоторговля** — рыночные ордера по ML-сигналам, стоп-лосс, тейк-профит
- **Дивидендная защита SL** — автоматический сдвиг стоп-лосса в экс-дивидендный день; окно защиты рассчитывается индивидуально для каждой акции на основе исторических гэпов (хранится в PostgreSQL)
- **Ручная торговля** — покупка/продажа через Telegram с проверкой баланса
- **FIFO при продаже** — закрывается самая ранняя позиция по каждому активу
- **Расчёт рентабельности** — чистый P&L с учётом комиссий брокера и НДФЛ
- **3-агентная система** — Coder / Reviewer / Architect через Anthropic API

## Технологический стек

| Слой | Технологии |
|---|---|
| Язык | Python 3.11+ |
| Telegram | aiogram 3.x (только inline-кнопки) |
| База данных | PostgreSQL + SQLAlchemy 2.x async + asyncpg |
| Миграции | Alembic |
| Кеш | Redis (`redis[asyncio]`) |
| ML | LightGBM, XGBoost, scikit-learn, TA-Lib, Optuna |
| Брокер | t-tech-investments (Tinkoff Invest API gRPC) |
| Логирование | structlog |
| Агенты | Anthropic API (claude-opus-4-6) |

## Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/LehaDeev/we_trust_in_people.git
cd we_trust_in_people

# 2. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Установить зависимости
pip install -r requirements.txt \
    --extra-index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

# 4. Настроить переменные окружения
cp .env.example .env
# Отредактировать .env — вписать токены, данные для подключения к БД и Redis

# 5. Применить миграции БД
alembic upgrade head

# 6. Собрать исторические данные
python -m scripts.collect_candles

# 7. Обучить ML-модель
python -m scripts.train_model

# 8. Запустить Telegram-бот (включает автоторговлю)
python -m scripts.run_bot
```

## Структура проекта

```
we_trust_in_people/
├── config/
│   └── settings.py          # Pydantic Settings: все параметры из .env
├── db/
│   ├── models.py            # ORM-модели: Asset, Candle, Signal, Trade
│   ├── database.py          # Async engine, get_session()
│   ├── candle_repo.py       # CRUD для активов и свечей
│   └── trade_repo.py        # CRUD для сделок (FIFO-порядок)
├── tinkoff/
│   ├── client.py            # Async gRPC клиент (t-tech-investments)
│   ├── market_data.py       # Свечи, последние цены (кеш Redis)
│   ├── instruments.py       # Поиск инструмента по тикеру, lot_size
│   ├── rate_limiter.py      # Rate limiter (TINKOFF_POST_ORDER_RATE)
│   ├── portfolio.py         # Портфель, баланс, рыночные ордера
│   ├── dividends.py         # Размер дивиденда в окне защиты (кеш Redis 24 ч)
│   └── dividend_gap_stats.py # Статистика закрытия дивидендного гэпа (PostgreSQL)
├── ml/
│   ├── features.py          # Feature engineering (TA-Lib индикаторы)
│   ├── dataset.py           # Загрузка данных из БД (кеш Redis)
│   ├── train.py             # Обучение ансамбля + Optuna HPO
│   ├── predict.py           # Инференс, predict_all() (кеш Redis)
│   └── weights/             # Веса моделей (не в репозитории)
├── trading/
│   ├── __init__.py
│   ├── state.py             # Runtime-флаг авто/ручной режим
│   ├── profitability.py     # Расчёт P&L: комиссии + НДФЛ + безубыток
│   ├── executor.py          # Открытие/закрытие позиций через Tinkoff API
│   ├── scheduler.py         # Главный цикл автоторговли (asyncio task)
│   └── notifier.py          # Telegram-уведомления о сделках
├── bot/
│   ├── main.py              # Точка входа бота, запуск scheduler
│   ├── keyboards.py         # Все inline-клавиатуры
│   └── handlers/
│       ├── main.py          # Главное меню
│       ├── signals.py       # ML-сигналы по тикерам
│       ├── portfolio.py     # Портфель: сводка + акции/облигации/ETF/валюта
│       └── trading.py       # Торговля: авто/ручной, позиции, история
├── agents/
│   ├── coder.py             # Agent 1: пишет код
│   ├── reviewer.py          # Agent 2: ревьюит код
│   └── architect.py         # Agent 3: валидирует по спецификации
├── scripts/
│   ├── collect_candles.py   # Сбор исторических свечей
│   ├── train_model.py       # Обучение ML-модели
│   └── run_bot.py           # Запуск Telegram-бота
├── utils/
│   ├── logger.py            # Structured logging (structlog)
│   └── redis_cache.py       # Redis синглтон: init/get/close
├── alembic/                 # Миграции БД
├── .env.example             # Пример конфигурации
└── CLAUDE.md                # Правила проекта для AI-ассистента
```

## Настройки `.env`

Все параметры настраиваются через `.env` (скопировать из `.env.example`).

### Tinkoff API

| Переменная | Описание | По умолчанию |
|---|---|---|
| `TINKOFF_TOKEN` | API-токен Tinkoff Invest | — |
| `TINKOFF_ACCOUNT_ID` | ID счёта | — |
| `TINKOFF_SANDBOX` | Режим песочницы | `false` |
| `TINKOFF_POST_ORDER_RATE` | Лимит PostOrder (заявок/сек) | `15` |

### Сбор данных

| Переменная | Описание | По умолчанию |
|---|---|---|
| `DATA_TICKERS` | Тикеры через запятую | `SBER,GAZP,...` |
| `DATA_CANDLE_INTERVAL` | Интервал свечей (`1h`, `1d`, `15min`) | `1h` |
| `DATA_HISTORY_DAYS` | Глубина истории при первом запуске | `365` |

### ML

| Переменная | Описание | По умолчанию |
|---|---|---|
| `ML_MODEL_VERSION` | Суффикс файлов весов | `v2` |
| `ML_LOOKAHEAD` | Свечей вперёд для генерации меток | `4` |
| `ML_THRESHOLD` | Порог доходности ±% для BUY/SELL | `0.01` |
| `ML_OPTUNA_TRIALS_LGBM` | Итераций Optuna для LightGBM | `50` |
| `ML_FORCE_TUNE` | Принудительный перезапуск Optuna | `false` |

### Redis

| Переменная | Описание | По умолчанию |
|---|---|---|
| `REDIS_HOST` | Хост Redis | `localhost` |
| `REDIS_PORT` | Порт Redis | `6379` |
| `REDIS_SIGNAL_TTL` | TTL кеша сигналов (сек) | `60` |
| `REDIS_CANDLES_TTL` | TTL кеша свечей (сек) | `300` |
| `REDIS_PORTFOLIO_TTL` | TTL кеша портфеля (сек) | `60` |
| `REDIS_PRICE_TTL` | TTL кеша цен (сек) | `30` |
| `REDIS_DIVIDEND_TTL` | TTL кеша дивидендных выплат (сек) | `86400` |

### Автоторговля

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

## Портфель

### Сводная страница

При открытии раздела «Портфель» отображается:
- Общая стоимость счёта (активы + свободный остаток)
- Свободный рублёвый баланс
- Итоговая стоимость по каждой категории

### Детализация по категориям

Четыре кнопки для перехода в список активов:

| Кнопка | Тип инструмента | Показывает |
|---|---|---|
| 📈 Акции | `share` | Тикер, кол-во шт., ср. цена покупки, текущая цена, P&L |
| 📄 Облигации | `bond` | Тикер, кол-во шт., ср. цена покупки, текущая цена, P&L |
| 🏦 ETF | `etf` | Тикер, кол-во шт., ср. цена покупки, текущая цена, P&L |
| 💱 Валюта | `currency` | Валюта, объём, ср. цена покупки, текущая цена, P&L |

Тикер определяется по FIGI из таблицы `assets` в БД.
Для активов вне отслеживаемого списка отображается FIGI[:10].

## Торговля

### Авто-режим

Scheduler запускается как фоновый asyncio-task вместе с ботом.
Каждые `TRADING_INTERVAL_SECONDS` выполняется один тик:

1. Проверяются открытые позиции — срабатывание **стоп-лосс** или **тейк-профит**
   — в экс-дивидендный день SL-порог автоматически снижается на размер дивиденда,
   чтобы предсказуемый гэп не вызвал ложное срабатывание
2. Получаются ML-сигналы для всех тикеров
3. **SELL-сигнал** — позиция закрывается только если чистый P&L > 0 после комиссий и НДФЛ
   _(SL/TP исполняются всегда — это управление риском)_
4. **BUY-сигнал** — позиция открывается при `confidence ≥ TRADING_CONFIDENCE_THRESHOLD`
   и наличии свободных слотов

### Ручной режим

Переключение прямо из бота кнопкой «Переключить на ручной».
Доступны кнопки **Купить** и **Продать**:
- **Покупка**: проверяет баланс, размер лота и лимит позиций; показывает прогноз P&L при TP/SL
- **Продажа**: показывает расчёт чистого P&L → требует подтверждения

### Расчёт P&L

```
entry_total  = цена_входа  × лоты × размер_лота
exit_total   = цена_выхода × лоты × размер_лота
gross_pnl    = exit_total - entry_total
комиссии     = entry_total × commission_pct + exit_total × commission_pct
НДФЛ         = max(0, (gross_pnl - комиссии) × tax_pct)
net_pnl      = gross_pnl - комиссии - НДФЛ
```

`TRADING_STOP_LOSS_PCT` и `TRADING_TAKE_PROFIT_PCT` задаются как **целевой чистый** результат.
Бот сам пересчитывает нужный gross-уровень цены. Пример при комиссии 0.3% и НДФЛ 13%:

```
TP = 1% чистых → цена должна вырасти на +1.75% (gross)
SL = 10% чистых → цена должна упасть на −9.42% (gross)
```

### FIFO

При наличии нескольких открытых позиций по одному активу всегда продаётся
самая ранняя (по дате открытия).

## Дивидендная защита стоп-лосса

Когда компания выплачивает дивиденды, акция предсказуемо падает примерно на
сумму дивиденда в экс-дивидендный день. Чтобы это не вызвало ложное срабатывание
стоп-лосса, бот применяет защитный сдвиг:

```
effective_sl = stop_loss_price - dividend_per_share
```

### Индивидуальное окно защиты

Для каждой акции автоматически вычисляется среднее количество дней, за которые
дивидендный гэп исторически закрывался (данные за 5 лет, дневные свечи):

1. Получить все дивидентные события из Tinkoff API
2. Для каждого — найти первый день, когда цена вернулась к 98 % от уровня до гэпа
3. Среднее (ceiling) сохраняется в PostgreSQL (`assets.dividend_gap_days`)
4. Пересчёт — раз в 30 дней или при первом добавлении актива

### Приоритет источников

| Приоритет | Источник | Как задать |
|---|---|---|
| 1 (высший) | Ручное переопределение | `TRADING_DIVIDEND_OVERRIDE=SBER:45,GAZP:90` |
| 2 | PostgreSQL (`assets.dividend_gap_days`) | Авто или `UPDATE assets SET ...` |
| 3 | Глобальный фоллбэк | `TRADING_DIVIDEND_PROTECTION_DAYS=1` |

## Redis-кеш

Кеш включён для дорогостоящих сетевых операций. При недоступности Redis
всё работает без кеша (graceful degradation).

| Что кешируется | Ключ | TTL |
|---|---|---|
| ML-сигнал | `signal:{ticker}:{version}` | `REDIS_SIGNAL_TTL` |
| Свечи из БД | `candles:{ticker}:{interval}` | `REDIS_CANDLES_TTL` |
| Портфель | `portfolio:{account_id}` | `REDIS_PORTFOLIO_TTL` |
| Последние цены | `last_price:{instrument_id}` | `REDIS_PRICE_TTL` |
| Дивидендные выплаты | `dividend_drop:{figi}:{date}:{days}` | `REDIS_DIVIDEND_TTL` |

## 3-агентная система

Экспериментальный модуль для ассистирования в разработке:

- **Coder** (`agents/coder.py`) — реализует задачи по спецификации
- **Reviewer** (`agents/reviewer.py`) — проверяет баги, стиль, async-корректность
- **Architect** (`agents/architect.py`) — валидирует соответствие спецификации проекта

Требует `ANTHROPIC_API_KEY` в `.env`.

## Важные ограничения

- Бинарные веса ML-моделей (`*.pkl`) **не включены** в репозиторий (>100 МБ)
- Файл `.env` с реальными ключами **никогда не коммитится**
- Перед включением `TRADING_ENABLED=true` убедитесь в корректности всех параметров
- `TINKOFF_SANDBOX=false` — торгуете реальными деньгами
