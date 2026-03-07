# Docker

## Сервисы

| Сервис | Образ | Роль |
|---|---|---|
| `postgres` | `postgres:15-alpine` | PostgreSQL (данные в named volume `postgres_data`) |
| `redis` | `redis:7-alpine` | Redis-кеш (данные в named volume `redis_data`) |
| `bot` | сборка из `Dockerfile` | Telegram-бот + автоторговля |

Сервисы `postgres` и `redis` имеют healthcheck; бот стартует только после их готовности.

## Переменные окружения

Все переменные берутся из `.env` через `env_file`. Два параметра переопределяются автоматически — внутри Docker сервисы доступны по имени контейнера, не `localhost`:

```yaml
POSTGRES_HOST: postgres
REDIS_HOST: redis
```

## Веса ML-моделей

Модели обучаются **вне Docker** — обучение ресурсоёмкое и выполняется один раз.
Директория `ml/weights/` смонтирована как bind-mount:

```
./ml/weights  →  /app/ml/weights
```

Веса не теряются при пересборке образа. После переобучения на хосте новые веса
доступны в контейнере немедленно — без перезапуска.

## Entrypoint

`docker-entrypoint.sh` при каждом старте контейнера:

1. Применяет миграции БД: `alembic upgrade head`
2. Запускает бота: `python -m scripts.run_bot`

## Полезные команды

```bash
# Поднять все сервисы
docker compose up -d

# Логи бота в реальном времени
docker compose logs -f bot

# Пересобрать образ бота после изменений кода
docker compose up -d --build

# Остановить все сервисы
docker compose down

# Остановить и удалить данные (postgres_data, redis_data)
docker compose down -v
```
