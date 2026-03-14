"""
Генерация меток для обучения ML с учителем.

Два режима:
  create_labels()     — по сырой доходности за окно lookahead (устарел, оставлен для совместимости).
  create_labels_sim() — по результату SL/TP-симуляции (рекомендуется).

Симуляционные метки устраняют рассогласование между целью обучения и метрикой HPO:
модель учится предсказывать ровно то, что оценивает Sortino-симуляция — прибыльность
реальной сделки с учётом SL/TP, комиссии и налога.
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
    Сгенерировать метки BUY/HOLD/SELL по сырой доходности за окно lookahead.

    Логика:
        future_return = (close[t+lookahead] - close[t]) / close[t]
        > +threshold  →  BUY  (2)
        < -threshold  →  SELL (0)
        иначе         →  HOLD (1)

    Устарел — оставлен для внешних скриптов. В train.py используется create_labels_sim().

    Аргументы:
        df:        DataFrame с колонкой "close", отсортированный по времени ASC.
        lookahead: свечей вперёд.
        threshold: минимальная доходность для BUY/SELL.

    Возвращает:
        pd.Series с метками (SELL=0, HOLD=1, BUY=2), без последних lookahead строк.
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)
    labels = np.ones(n, dtype=np.int8)

    valid = n - lookahead
    current_close = close[:valid]
    future_close = close[lookahead:lookahead + valid]

    with np.errstate(invalid="ignore", divide="ignore"):
        future_return = np.where(
            current_close != 0,
            (future_close - current_close) / current_close,
            0.0,
        )

    labels[:valid][future_return > threshold] = LABEL_MAP["BUY"]
    labels[:valid][future_return < -threshold] = LABEL_MAP["SELL"]

    return pd.Series(labels[:valid], index=df.index[:valid], name="label")


def create_labels_sim(
    index: pd.Index,
    close_window: np.ndarray,
    lookahead: int,
    threshold: float,
    commission_pct: float,
    sl_pct: float,
    tp_pct: float,
    tax_pct: float,
) -> pd.Series:
    """
    Сгенерировать метки BUY/HOLD/SELL на основе SL/TP-симуляции каждой свечи.

    Для каждой строки симулируется «что было бы, если открыть BUY на этом баре»:
    выход по первому достижению SL или TP в пределах lookahead свечей, иначе close[t+lookahead].
    Чистый P&L (после комиссии и НДФЛ) определяет метку:

        gross > +threshold  →  BUY  (2) — цена выросла достаточно
        gross < -threshold  →  SELL (0) — цена упала достаточно
        иначе               →  HOLD (1) — движение в пределах порога

    Устраняет рассогласование между целью обучения (классификация по сырой доходности)
    и метрикой HPO (Sortino по симулированным сделкам): модель теперь учится предсказывать
    ровно то, что измеряет оптимизатор.

    Аргументы:
        index:          pd.Index строк (время), совпадает с осью 0 close_window.
        close_window:   numpy array (N, lookahead+1):
                        close_window[i, 0] = close[t_i] (цена входа),
                        close_window[i, j] = close[t_i + j], j=1..lookahead.
        lookahead:      максимальный горизонт удержания позиции (свечей).
        threshold:      минимальный |net_pnl| для BUY/SELL (доля; 0.005 = 0.5%).
        commission_pct: комиссия брокера (доля за одну сторону; 0.003 = 0.3%).
        sl_pct:         стоп-лосс от цены входа (доля; 0.03 = 3%).
        tp_pct:         тейк-профит от цены входа (доля; 0.05 = 5%).
        tax_pct:        ставка НДФЛ на прибыль (0.13 = 13%).

    Возвращает:
        pd.Series с метками (SELL=0, HOLD=1, BUY=2), индекс = index.
    """
    entry = close_window[:, 0]
    sl_prices = entry * (1.0 - sl_pct)
    tp_prices = entry * (1.0 + tp_pct)

    # По умолчанию выходим по цене закрытия через lookahead свечей
    exit_prices = close_window[:, lookahead].copy()
    still_open = np.ones(len(entry), dtype=bool)

    # Ищем первое касание SL или TP внутри окна
    for j in range(1, lookahead + 1):
        c = close_window[:, j]
        hit_sl = still_open & (c <= sl_prices)
        hit_tp = still_open & (c >= tp_prices)
        exit_prices[hit_sl] = sl_prices[hit_sl]
        exit_prices[hit_tp] = tp_prices[hit_tp]
        still_open &= ~(hit_sl | hit_tp)

    # Чистый P&L = gross - commission - НДФЛ (только на прибыль)
    with np.errstate(invalid="ignore", divide="ignore"):
        gross = np.where(entry > 0.0, (exit_prices - entry) / entry, 0.0)
    commission = 2.0 * commission_pct
    tax = np.maximum(0.0, (gross - commission) * tax_pct)
    net_pnl = gross - commission - tax

    # Метки определяем по gross (сырой доходности), а не по net_pnl.
    # Комиссия и налог уже учтены в Sortino-метрике HPO — включать их в границы
    # меток нельзя: commission=0.6% сдвигает SELL-зону вправо и любое падение
    # даже на 0.1% становится SELL → дисбаланс классов (>60% SELL).
    labels = np.ones(len(gross), dtype=np.int8)  # HOLD по умолчанию
    labels[gross > threshold] = LABEL_MAP["BUY"]
    labels[gross < -threshold] = LABEL_MAP["SELL"]
    # Нулевые/отрицательные entry — оставляем HOLD
    labels[entry <= 0.0] = LABEL_MAP["HOLD"]

    return pd.Series(labels, index=index, name="label")
