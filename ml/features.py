"""
Вычисление технических индикаторов для ML-моделей через TA-Lib.

Принимает DataFrame с колонками OHLCV и возвращает обогащённый DataFrame
с признаками технического анализа, готовыми для обучения LightGBM.

Все ценовые признаки нормализованы (отношения) — не зависят от абсолютного
уровня цены тикера. Это обеспечивает стационарность признаков и корректное
дообучение при росте или падении цены со временем.

Группы признаков (40 признаков):
    - Тренд: SMA, нормализованные EMA, ADX
    - Импульс: RSI, MACD, ROC, MFI, Stochastic %K/%D
    - Волатильность: Bollinger %B, ATR-ratio, ширина Bollinger, историческая волатильность
    - Объём: OBV, нормализованный объём, volume_ratio, CMF, изменения объёма
    - Ценовые отношения: close/SMA, high-low/close, SMA20/SMA50, VWAP-ratio
    - Структура свечи: body_ratio, upper/lower shadow, gap при открытии
    - Лаговые доходности: 1h, 4h, 8h, 24h (прямой сигнал импульса)
    - Временные: час дня и день недели (синус/косинус — циклическое кодирование)
"""
import numpy as np
import pandas as pd
import talib

# Имена колонок признаков (используются для согласованности между обучением и инференсом)
FEATURE_COLUMNS: list[str] = [
    # ── Тренд (нормализованные) ──────────────────────────────────────────────
    "SMA_20", "SMA_50", "ema12_ratio", "ema26_ratio", "ADX_14",
    # ── Импульс ──────────────────────────────────────────────────────────────
    "RSI_14", "MACD", "MACD_signal", "MACD_hist", "ROC_10",
    # MFI = RSI с учётом объёма; Stochastic %K/%D — перекупленность/перепроданность
    "MFI_14", "stoch_k", "stoch_d",
    # ── Волатильность (нормализованные) ──────────────────────────────────────
    "bb_pct_b", "atr_ratio", "BB_width",
    # Историческая волатильность: скользящее std доходностей за 20 баров
    "hist_vol_20",
    # ── Объём ────────────────────────────────────────────────────────────────
    "OBV", "VOLUME_SMA_20", "volume_ratio",
    # CMF — давление покупателей с учётом объёма; изменения объёма за 1h/4h
    "cmf_20", "volume_change_1h", "volume_change_4h",
    # ── Ценовые отношения (производные) ──────────────────────────────────────
    "close_sma20_ratio", "close_sma50_ratio", "high_low_ratio", "sma20_sma50_ratio",
    # VWAP-ratio: где цена относительно средневзвешенной по объёму цены дня
    "vwap_ratio",
    # ── Структура свечи ───────────────────────────────────────────────────────
    # body_ratio > 0 — бычья свеча, < 0 — медвежья; тени показывают отверженные уровни
    "body_ratio", "upper_shadow", "lower_shadow", "gap",
    # ── Лаговые доходности — прямой сигнал импульса ──────────────────────────
    # Показывают куда цена двигалась последние часы; один из сильнейших предикторов
    "return_1h", "return_4h", "return_8h", "return_24h",
    # ── Временные признаки — внутридневная сезонность MOEX ───────────────────
    # Синус/косинус для сохранения цикличности (23:00 и 00:00 — соседние точки)
    # Первый час (10–11 МСК): высокая волатильность; обед (13–14): боковик;
    # последний час (17–18): направленное движение с объёмом
    "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычислить признаки технических индикаторов для одного тикера.

    Аргументы:
        df: DataFrame с колонками [time, open, high, low, close, volume].
            Должен быть отсортирован по времени ASC. Значения — float64.

    Возвращает:
        DataFrame с исходными колонками + FEATURE_COLUMNS.
        Первые ~50 строк удаляются (период прогрева индикаторов).
        Индекс сбрасывается к целым числам начиная с 0.
    """
    df = df.copy()

    # TA-Lib требует numpy-массивы типа float64
    close = df["close"].values.astype(np.float64)
    open_prices = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # ── Индикаторы тренда ────────────────────────────────────────────────────
    sma20 = talib.SMA(close, timeperiod=20)
    sma50 = talib.SMA(close, timeperiod=50)
    ema12 = talib.EMA(close, timeperiod=12)
    ema26 = talib.EMA(close, timeperiod=26)

    df["SMA_20"] = sma20
    df["SMA_50"] = sma50
    # Нормализованные EMA: отношение цены закрытия к скользящей средней
    # close/EMA > 1 → цена выше EMA (бычий сигнал), < 1 → ниже (медвежий)
    df["ema12_ratio"] = np.where(ema12 != 0, close / ema12, np.nan)
    df["ema26_ratio"] = np.where(ema26 != 0, close / ema26, np.nan)
    df["ADX_14"] = talib.ADX(high, low, close, timeperiod=14)

    # ── Индикаторы импульса ──────────────────────────────────────────────────
    df["RSI_14"] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    df["ROC_10"] = talib.ROC(close, timeperiod=10)
    # MFI: Money Flow Index — RSI с учётом объёма [0–100]
    df["MFI_14"] = talib.MFI(high, low, close, volume, timeperiod=14)
    # Stochastic: медленный %K и %D — позиция цены в диапазоне за 14 баров [0–100]
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0,
    )
    df["stoch_k"] = slowk
    df["stoch_d"] = slowd

    # ── Индикаторы волатильности ─────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2
    )
    atr = talib.ATR(high, low, close, timeperiod=14)

    # Bollinger %B: позиция цены внутри полос [0 = нижняя полоса, 1 = верхняя]
    # < 0 → цена ниже нижней полосы (перепродан), > 1 → выше верхней (перекуплен)
    bb_band_width = bb_upper - bb_lower
    df["bb_pct_b"] = np.where(bb_band_width != 0, (close - bb_lower) / bb_band_width, np.nan)

    # ATR как доля от цены закрытия: мера волатильности независимо от уровня цены
    df["atr_ratio"] = np.where(close != 0, atr / close, np.nan)

    # Ширина полос Боллинджера: нормализованная ширина (защита от деления на ноль)
    df["BB_width"] = np.where(bb_mid != 0, (bb_upper - bb_lower) / bb_mid, np.nan)
    # Историческая волатильность: скользящее std однопериодных доходностей за 20 баров
    # Нормализованная мера риска, не зависящая от абсолютного уровня цены
    returns_series = pd.Series(close).pct_change()
    df["hist_vol_20"] = returns_series.rolling(20).std().values

    # ── Индикаторы объёма ────────────────────────────────────────────────────
    volume_sma20 = talib.SMA(volume, timeperiod=20)
    df["OBV"] = talib.OBV(close, volume)
    df["VOLUME_SMA_20"] = volume_sma20
    # Относительный объём: текущий бар vs средний (>1 = повышенный объём)
    df["volume_ratio"] = np.where(volume_sma20 != 0, volume / volume_sma20, np.nan)
    # Chaikin Money Flow: давление покупателей/продавцов через объём за 20 баров
    # > 0 — накопление (покупки), < 0 — распределение (продажи)
    hl_range = high - low
    hl_range_safe = np.where(hl_range != 0, hl_range, np.nan)
    mfv = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range_safe * volume, 0.0)
    mfv_series = pd.Series(mfv)
    vol_series = pd.Series(volume)
    df["cmf_20"] = (mfv_series.rolling(20).sum() / vol_series.rolling(20).sum()).values
    # Изменения объёма: резкий рост объёма часто предшествует ценовому движению
    df["volume_change_1h"] = vol_series.pct_change(periods=1).values
    df["volume_change_4h"] = vol_series.pct_change(periods=4).values

    # ── Производные ценовые отношения ────────────────────────────────────────
    df["close_sma20_ratio"] = np.where(sma20 != 0, close / sma20, np.nan)
    df["close_sma50_ratio"] = np.where(sma50 != 0, close / sma50, np.nan)
    df["high_low_ratio"] = np.where(close != 0, (high - low) / close, np.nan)
    # Отношение SMA_20 к SMA_50: > 1 = золотой крест (бычий тренд), < 1 = мёртвый крест
    df["sma20_sma50_ratio"] = np.where(sma50 != 0, sma20 / sma50, np.nan)
    # VWAP-ratio: где цена относительно средневзвешенной по объёму цены за торговый день.
    # > 1 — цена выше VWAP (бычий сигнал внутри дня), < 1 — ниже (медвежий).
    # VWAP сбрасывается каждый торговый день — используем группировку по дате.
    time_col_vwap = pd.to_datetime(df["time"])
    if time_col_vwap.dt.tz is not None:
        date_key = time_col_vwap.dt.tz_convert("Europe/Moscow").dt.date
    else:
        date_key = (time_col_vwap + pd.Timedelta(hours=3)).dt.date
    typical_price = (high + low + close) / 3
    df["_tp_vol"] = typical_price * volume
    df["_vol"] = volume
    df["_date"] = date_key.values
    df["_cumtp"] = df.groupby("_date")["_tp_vol"].cumsum()
    df["_cumvol"] = df.groupby("_date")["_vol"].cumsum()
    vwap = df["_cumtp"] / df["_cumvol"]
    df["vwap_ratio"] = np.where(vwap != 0, close / vwap, np.nan)
    df.drop(columns=["_tp_vol", "_vol", "_date", "_cumtp", "_cumvol"], inplace=True)

    # ── Структура свечи ───────────────────────────────────────────────────────
    # Нормализованы относительно диапазона high-low свечи.
    # body_ratio > 0 — бычья свеча (close > open), < 0 — медвежья.
    # upper/lower shadow показывают отверженные уровни (хвосты свечи).
    # gap — разрыв между open текущего бара и close предыдущего.
    prev_close = pd.Series(close).shift(1).values
    df["body_ratio"]   = (close - open_prices) / hl_range_safe
    df["upper_shadow"] = (high - np.maximum(open_prices, close)) / hl_range_safe
    df["lower_shadow"] = (np.minimum(open_prices, close) - low) / hl_range_safe
    df["gap"]          = np.where(prev_close != 0, (open_prices - prev_close) / prev_close, np.nan)

    # ── Лаговые доходности ───────────────────────────────────────────────────
    # Прямой сигнал импульса: куда двигалась цена за последние N баров.
    # Один из наиболее предсказательных признаков для краткосрочного прогноза.
    close_series = pd.Series(close, index=df.index)
    df["return_1h"]  = close_series.pct_change(periods=1)
    df["return_4h"]  = close_series.pct_change(periods=4)
    df["return_8h"]  = close_series.pct_change(periods=8)
    df["return_24h"] = close_series.pct_change(periods=24)

    # ── Временные признаки ───────────────────────────────────────────────────
    # Внутридневные паттерны MOEX: первый/последний час, обед — разная динамика.
    # Синус/косинус сохраняет цикличность: 23:00 и 00:00 считаются соседними.
    time_col = pd.to_datetime(df["time"])
    # Конвертируем в московское время (UTC+3) — биржа работает в МСК
    if time_col.dt.tz is not None:
        time_col = time_col.dt.tz_convert("Europe/Moscow")
    else:
        # Нет timezone-инфо → предполагаем UTC, сдвигаем на +3
        time_col = time_col + pd.Timedelta(hours=3)

    hour = time_col.dt.hour.values.astype(np.float64)
    dow  = time_col.dt.dayofweek.values.astype(np.float64)  # 0=Пн, 4=Пт

    df["hour_sin"]      = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * hour / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * dow / 5)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dow / 5)

    # Удаляем строки где хотя бы один признак NaN (период прогрева, ~50 строк на тикер)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return df
