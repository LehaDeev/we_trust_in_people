FROM python:3.11-slim

WORKDIR /app

# Системные зависимости:
# gcc/g++ — нужен для компиляции некоторых Python-пакетов (asyncpg, aiohttp)
# wget/ca-certificates — для возможных загрузок при установке
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости отдельным слоем — кешируется при неизменном requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код (без .env, venv и весов — см. .dockerignore)
COPY . .

# ml/weights монтируется как volume из docker-compose
# Создаём директорию заранее чтобы volume смонтировался корректно
RUN mkdir -p ml/weights

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "-m", "scripts.run_bot"]
