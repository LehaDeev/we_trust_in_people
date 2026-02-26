# We Trust in People — Торговый бот

Автоматизированный торговый помощник для Tinkoff Инвестиции с ML-сигналами покупки/продажи.

## Возможности
- Интеграция с Tinkoff Invest API (асинхронная)
- PostgreSQL для хранения рыночных данных и сигналов
- ML-модели для предсказания сигналов покупки/продажи
- Telegram-бот с инлайн-интерфейсом

## Технологический стек
- Python 3.11+
- aiohttp, aiogram 3.x
- SQLAlchemy 2.x async + asyncpg
- PostgreSQL
- LightGBM / CatBoost

## Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/LehaDeev/we_trust_in_people.git
cd we_trust_in_people

# 2. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Настроить переменные окружения
cp .env.example .env
# Отредактировать .env — вписать токены и данные для подключения к БД

# 5. Применить миграции БД
alembic upgrade head
```

## Структура проекта

```
we_trust_in_people/
├── config/         # Настройки (Pydantic)
├── db/             # Модели БД и подключение
├── tinkoff/        # Клиент Tinkoff API
├── ml/             # ML-модели (веса не включены в публичный репозиторий)
├── bot/            # Telegram-бот
└── utils/          # Логгер и общие утилиты
```

## Важно: веса моделей
Веса ML-моделей **не включены** в этот репозиторий.
Они хранятся отдельно и подключаются через путь, указанный в `.env`.
