# We Trust in People — Торговый бот

Автоматизированный торговый помощник для Tinkoff Инвестиции с ML-сигналами покупки/продажи.

## Возможности
- Интеграция с Tinkoff Invest API (асинхронная, боевой режим)
- PostgreSQL для хранения рыночных данных и сигналов
- Автоматический сбор исторических свечей по списку тикеров
- ML-модели для предсказания сигналов покупки/продажи
- Telegram-бот с инлайн-интерфейсом

## Технологический стек
- Python 3.13+
- aiogram 3.x (только inline-кнопки)
- SQLAlchemy 2.x async + asyncpg
- PostgreSQL 18
- Alembic (миграции)
- LightGBM, TA-Lib
- structlog

## Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/LehaDeev/we_trust_in_people.git
cd we_trust_in_people

# 2. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Установить зависимости
pip install -r requirements.txt --extra-index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple

# 4. Настроить переменные окружения
cp .env.example .env
# Отредактировать .env — вписать токены и данные для подключения к БД

# 5. Применить миграции БД
alembic upgrade head

# 6. Собрать исторические данные
python -m scripts.collect_candles
```

## Структура проекта

```
we_trust_in_people/
├── config/         # Настройки (Pydantic Settings)
├── db/             # Модели БД, подключение, репозитории
│   ├── models.py       # ORM: Asset, Candle, Signal
│   ├── database.py     # Async engine, session
│   └── candle_repo.py  # Операции с активами и свечами
├── tinkoff/        # Клиент Tinkoff Invest API
│   ├── client.py       # Async gRPC клиент
│   ├── market_data.py  # Свечи, цены, стакан
│   ├── instruments.py  # Поиск инструментов по тикеру
│   └── portfolio.py    # Портфель и ордера
├── ml/             # ML-модели (код обучения приватный)
│   └── weights/        # Веса моделей (не включены в репозиторий)
├── bot/            # Telegram-бот (в разработке)
├── scripts/        # Утилиты запуска
│   └── collect_candles.py  # Сбор исторических свечей
├── alembic/        # Миграции БД
└── utils/          # Логгер
```

## Сбор данных

Скрипт `scripts/collect_candles.py` загружает 365 дней часовых свечей для 10 голубых фишек MOEX:
`SBER, GAZP, LKOH, YDEX, NVTK, GMKN, MGNT, TATN, ROSN, MTSS`

При повторном запуске догружает только новые свечи (инкрементально).

## ML

Для предсказания сигналов BUY / SELL / HOLD используются:
- **LightGBM** — градиентный бустинг для классификации
- **TA-Lib** — технические индикаторы на основе OHLCV данных

Код обучения и веса моделей не публикуются.
