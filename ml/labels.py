"""
Генерация меток для обучения ML с учителем.

Преобразует ценовой ряд OHLCV в метки классификации BUY / HOLD / SELL
на основе будущей доходности за заданное окно прогноза.
"""
import numpy as np
import pandas as pd

# Числовое кодирование меток для LightGBM (обязательно с нуля)
LABEL_MAP: dict[str, int] = {
    "SELL": 0,
    "HOLD": 1,
    "BUY":  2,
}
LABEL_NAMES: list[str] = ["SELL", "HOLD", "BUY"]  # индекс → название


def create_labels(
    df: pd.DataFrame,
    lookahead: int = 4,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Сгенерировать целочисленные метки BUY/HOLD/SELL по будущей доходности цены.

    Логика меток:
        future_return = (close[t+lookahead] - close[t]) / close[t]
        > +threshold  →  BUY  (2)
        < -threshold  →  SELL (0)
        иначе         →  HOLD (1)

    Последние `lookahead` строк удаляются — будущая цена для них неизвестна.

    Аргументы:
        df: DataFrame с колонкой "close", отсортированный по времени ASC.
        lookahead: количество свечей вперёд для измерения доходности (по умолчанию 4 = 4 часа).
        threshold: минимальная абсолютная доходность для сигнала BUY/SELL (по умолчанию 1%).

    Возвращает:
        pd.Series с целыми метками (SELL=0, HOLD=1, BUY=2), тот же индекс что у df,
        но без последних `lookahead` строк.
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)

    labels = np.ones(n, dtype=np.int8)  # по умолчанию: HOLD

    # future_return определён только для индексов 0 .. n-lookahead-1
    valid = n - lookahead
    future_close = close[lookahead:lookahead + valid]
    current_close = close[:valid]

    # Защита от деления на ноль
    with np.errstate(invalid="ignore", divide="ignore"):
        future_return = np.where(
            current_close != 0,
            (future_close - current_close) / current_close,
            0.0,
        )

    labels[:valid][future_return > threshold] = LABEL_MAP["BUY"]
    labels[:valid][future_return < -threshold] = LABEL_MAP["SELL"]

    # Удаляем последние `lookahead` строк — будущих данных нет
    result = pd.Series(labels[:valid], index=df.index[:valid], name="label")
    return result
