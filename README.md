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
- LightGBM + TA-Lib (технический анализ, 150+ индикаторов)
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
├── ml/             # ML-pipeline
│   ├── features.py     # Технические индикаторы (TA-Lib): RSI, MACD, BB, ATR...
│   ├── labels.py       # Генерация меток BUY/SELL/HOLD
│   ├── dataset.py      # Загрузка данных из БД для обучения
│   ├── train.py        # Обучение LightGBM модели
│   ├── predict.py      # Инференс: сигнал для тикера
│   └── weights/        # Веса моделей (не включены в репозиторий)
├── bot/            # Telegram-бот (в разработке)
├── scripts/        # Утилиты запуска
│   ├── collect_candles.py  # Сбор исторических свечей
│   └── train_model.py      # Запуск обучения ML модели
├── alembic/        # Миграции БД
└── utils/          # Логгер
```

## Сбор данных

Скрипт `scripts/collect_candles.py` загружает 365 дней часовых свечей для 10 голубых фишек MOEX:
`SBER, GAZP, LKOH, YDEX, NVTK, GMKN, MGNT, TATN, ROSN, MTSS`

При повторном запуске догружает только новые свечи (инкрементально).

## ML Pipeline

Технические индикаторы вычисляются с помощью **TA-Lib** (RSI, MACD, Bollinger Bands, ATR, ADX, EMA, SMA, OBV и др.).

### Метки (labels)
- Смотрим на `+4` свечи вперёд (4 часа)
- Рост цены `> 1%` → **BUY**
- Падение цены `> 1%` → **SELL**
- Иначе → **HOLD**

### Обучение модели

```bash
python -m scripts.train_model
```

Веса сохраняются в `ml/weights/lgbm_v1.pkl`.

## Важно: веса моделей
Веса ML-моделей **не включены** в этот репозиторий.
Они хранятся в `ml/weights/` (директория в `.gitignore`).
