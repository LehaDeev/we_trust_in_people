#!/bin/bash
set -e

echo "Applying DB migrations..."
alembic upgrade head

echo "Starting application..."
exec "$@"
