# We Trust in People — Торговый бот

Автоматизированный торговый помощник для Tinkoff Инвестиции с ML-сигналами,
авто/ручной торговлей и учётом комиссий и налогов.

## Возможности

- **Tinkoff Invest API** — асинхронная интеграция, боевой режим, rate limiter
- **ML-ансамбль** — `VotingClassifier(soft)`: LightGBM + ExtraTrees + SVC, Optuna HPO, per-ticker отбор признаков → [подробнее](docs/ml.md)
- **Ночное дообучение** — инкрементальный сбор свечей и переобучение моделей каждую ночь
- **Автоторговля** — рыночные ордера по ML-сигналам, SL/TP, дивидендная защита → [подробнее](docs/trading.md)
- **Ручная торговля** — покупка/продажа через Telegram с расчётом P&L и подтверждением
- **Портфель** — сводка по счёту + детализация по категориям (акции / облигации / ETF / валюта)
- **Redis-кеш** — свечи, цены, портфель, сигналы; graceful degradation при недоступности
- **3-агентная система** — Coder / Reviewer / Architect через Anthropic API

## Технологический стек

| Слой | Технологии |
|---|---|
| Язык | Python 3.11+ |
| Telegram | aiogram 3.x (только inline-кнопки) |
| База данных | PostgreSQL + SQLAlchemy 2.x async + asyncpg |
| Миграции | Alembic |
| Кеш | Redis (`redis[asyncio]`) |
| ML | LightGBM, scikit-learn, TA-Lib, Optuna |
| Брокер | t-tech-investments (Tinkoff Invest API gRPC) |
| Логирование | structlog |

## Быстрый старт

### 1. Клонировать репозиторий и настроить `.env`

```bash
git clone https://github.com/LehaDeev/we_trust_in_people.git
cd we_trust_in_people

cp .env.example .env
# Вписать в .env: TINKOFF_TOKEN, TINKOFF_ACCOUNT_ID, TELEGRAM_BOT_TOKEN, POSTGRES_PASSWORD
```

Полный список переменных — в [docs/settings.md](docs/settings.md).

### 2. Обучить ML-модели (на хосте, вне Docker)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m scripts.collect_candles   # собрать исторические данные
python -m scripts.train_model       # обучить ансамбли для каждого тикера
```

Веса сохраняются в `ml/weights/` — Docker монтирует эту директорию автоматически.
Подробнее об обучении: [docs/ml.md](docs/ml.md).

### 3. Запустить через Docker Compose

```bash
docker compose up -d
docker compose logs -f bot
```

При первом запуске автоматически применяются миграции Alembic.
Подробнее о Docker: [docs/docker.md](docs/docker.md).

## Документация

| Раздел | Описание |
|---|---|
| [docs/ml.md](docs/ml.md) | ML-ансамбль, признаки, per-ticker отбор, ночное дообучение |
| [docs/trading.md](docs/trading.md) | Авто/ручная торговля, P&L, FIFO, дивидендная защита |
| [docs/settings.md](docs/settings.md) | Все переменные `.env` с описанием и дефолтами |
| [docs/docker.md](docs/docker.md) | Docker Compose: сервисы, команды, переменные окружения |

## Структура проекта

```
we_trust_in_people/
├── config/settings.py       # Pydantic Settings: все параметры из .env
├── db/                      # ORM-модели, async engine, CRUD (candles, trades)
├── tinkoff/                 # Async gRPC клиент, рыночные данные, портфель, дивиденды
├── ml/                      # Feature engineering, обучение, инференс, веса
├── trading/                 # Исполнение ордеров, scheduler, P&L, уведомления
├── bot/                     # Telegram-бот: handlers, keyboards
├── scripts/                 # Точки входа: collect_candles, train_model, run_bot
├── utils/                   # logger, redis_cache
├── docs/                    # Подробная документация
├── alembic/                 # Миграции БД
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Важные ограничения

- Веса ML-моделей (`*.pkl`) **не включены** в репозиторий
- Файл `.env` с реальными ключами **никогда не коммитится**
- Перед включением `TRADING_ENABLED=true` убедитесь в корректности всех параметров
- `TINKOFF_SANDBOX=false` — торгуете реальными деньгами
