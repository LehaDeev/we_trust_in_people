"""
Экспорт свечей из PostgreSQL в CSV-файлы для обучения в Google Colab.

Файлы сохраняются в ml/data/{TICKER}.csv (включая USDRUB).
Загрузите папку ml/data/ на Google Drive перед запуском ноутбука.

Запуск:
    python -m scripts.export_candles_csv
"""
import asyncio
from pathlib import Path

from config.settings import data_settings
from db.database import close_db, init_db
from ml.dataset import load_ticker_data, load_usdrub_data
from utils.logger import logger

DATA_DIR = Path(__file__).parent.parent / "ml" / "data"


async def main() -> None:
    """Экспортировать свечи всех тикеров и USD/RUB в CSV."""
    await init_db()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tickers = data_settings.tickers
    interval = data_settings.candle_interval
    total = len(tickers) + 1  # +1 для USDRUB

    print(f"Экспорт {len(tickers)} тикеров + USDRUB → {DATA_DIR}")
    print("=" * 50)

    # Основные тикеры
    for i, ticker in enumerate(tickers, 1):
        df = await load_ticker_data(ticker, interval)
        if df.empty:
            print(f"  [{i}/{total}] {ticker:<8} — нет данных, пропущен")
            continue
        path = DATA_DIR / f"{ticker}.csv"
        df.to_csv(path, index=False)
        print(f"  [{i}/{total}] {ticker:<8} {len(df):>6} строк → {path.name}")

    # USD/RUB
    usdrub_df = await load_usdrub_data(interval)
    if usdrub_df.empty:
        print(f"  [{total}/{total}] USDRUB   — нет данных")
    else:
        path = DATA_DIR / "USDRUB.csv"
        usdrub_df.to_csv(path, index=False)
        print(f"  [{total}/{total}] USDRUB   {len(usdrub_df):>6} строк → {path.name}")

    await close_db()

    print("=" * 50)
    print(f"Готово. Загрузите папку ml/data/ на Google Drive.")
    logger.info("Экспорт CSV завершён", data_dir=str(DATA_DIR))


if __name__ == "__main__":
    asyncio.run(main())
