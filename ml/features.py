"""
Вычисление технических индикаторов для ML-моделей через TA-Lib.

Принимает DataFrame с колонками OHLCV и возвращает обогащённый DataFrame
с признаками технического анализа, готовыми для обучения LightGBM.

Все ценовые признаки нормализованы (отношения) — не зависят от абсолютного
уровня цены тикера. Это обеспечивает стационарность признаков и корректное
дообучение при росте или падении цены со временем.

Группы признаков (58 признаков):
    - Тренд: SMA, нормализованные EMA, ADX
    - Импульс: RSI, MACD, ROC, MFI, Stochastic %K/%D, CCI, Aroon Up/Down, Williams %R
    - Импульс (дельты): изменение RSI/Stoch/MACD_hist/CCI за 4 бара — направление индикатора
    - Волатильность: Bollinger %B, ATR-ratio, ширина Bollinger, историческая волатильность
    - Объём: OBV, нормализованный объём, volume_ratio, CMF, изменения объёма
    - Ценовые отношения: close/SMA, high-low/close, SMA20/SMA50, VWAP-ratio
    - Экстремумы: позиция цены относительно 52-недельных max/min
    - Donchian: позиция и ширина канала за 20 баров (breakout детектор)
    - Роллинг: среднее и std доходностей за 8 баров (краткосрочный режим)
    - Структура свечи: body_ratio, upper/lower shadow, gap при открытии
    - Лаговые доходности: 1h, 4h, 8h, 24h (прямой сигнал импульса)
    - Временные: час дня и день недели (синус/косинус — циклическое кодирование)
    - Режим рынка: автокорреляция доходностей, корреляция цена×объём
"""
import numpy as np
import pandas as pd
import talib

from config.settings import ml_settings

# Имена колонок признаков (используются для согласованности между обучением и инференсом)
FEATURE_COLUMNS: list[str] = [
    # ── Тренд (нормализованные) ──────────────────────────────────────────────
    "SMA_20", "SMA_50", "ema12_ratio", "ema26_ratio", "ADX_14",
    # ── Импульс ──────────────────────────────────────────────────────────────
    "RSI_14", "MACD", "MACD_signal", "MACD_hist", "ROC_10",
    # MFI = RSI с учётом объёма; Stochastic %K/%D — перекупленность/перепроданность
    "MFI_14", "stoch_k", "stoch_d",
    # CCI: отклонение цены от статистической нормы (дополняет RSI другой формулой)
    "CCI_14",
    # Aroon: сколько баров назад был максимум/минимум → сила и направление тренда
    "aroon_up", "aroon_down",
    # Дельты индикаторов за 4 бара: направление изменения важнее абсолютного значения
    # RSI пробивает 50 снизу вверх → сильнее чем просто RSI=52
    "rsi_delta_4h", "stoch_k_delta_4h", "macd_hist_delta_4h",
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
    # 52-недельные экстремумы: режим рынка (у максимума = импульс, у минимума = перепродан)
    "high_252_ratio", "low_252_ratio",
    # Donchian channel: позиция цены в 20-барном диапазоне + ширина канала
    # donchian_pct=1.0 → пробой вверх (breakout), 0.0 → пробой вниз
    "donchian_pct", "donchian_width",
    # Роллинговые статистики доходностей за 8 баров
    # mean > 0 = краткосрочный восходящий тренд; std = мини-волатильность
    "return_mean_8h", "return_std_8h",
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
    # ── USD/RUB курс ─────────────────────────────────────────────────────────
    # Сильно влияет на экспортёров (GAZP, LKOH, ROSN, NVTK, GMKN).
    # usdrub_ratio: нормализованный курс (close / SMA20) — без зависимости от уровня
    # usdrub_change_1h: изменение курса за 1 час — прямой сигнал движения рубля
    "usdrub_ratio", "usdrub_change_1h",
    # ── Режим рынка и подтверждение объёма ───────────────────────────────────
    # autocorr_returns: скользящая автокорреляция (лаг=1) доходностей за N баров.
    # > 0 → импульсный режим (тренд продолжается), < 0 → возврат к среднему.
    # Одна из ключевых идей WorldQuant 101 alphas и академических работ по momentum.
    "autocorr_returns",
    # price_vol_corr: скользящая корреляция доходность × изменение объёма за N баров.
    # > 0 → объём подтверждает направление цены (сильный сигнал),
    # < 0 → объём расходится с ценой (дивергенция, возможный разворот).
    "price_vol_corr",
    # williams_r: Williams %R — осциллятор перекупленности/перепроданности [-100, 0].
    # Отличается от Stochastic знаком и формулой: фокус на расстоянии от максимума.
    # -20..0 = перекуплен, -80..-100 = перепродан. Дополняет RSI и stoch иным взглядом.
    "williams_r",
    # cci_delta: изменение CCI за N баров — направление отклонения типичной цены от нормы.
    # Аналогично rsi_delta_4h / stoch_k_delta_4h / macd_hist_delta_4h: направление важнее
    # абсолютного значения. CCI пробивает +100 снизу → сильнее чем просто CCI=110.
    "cci_delta",
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
    # CCI: отклонение типичной цены от скользящего среднего, нормализованное на MAD
    # Выше +100 = перекуплен, ниже -100 = перепродан; дополняет RSI другой формулой
    df["CCI_14"] = talib.CCI(high, low, close, timeperiod=14)
    # Aroon: Up показывает сколько баров назад был максимум (0–100), Down — минимум
    # Aroon Up > 70 и Down < 30 → сильный восходящий тренд
    aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
    df["aroon_up"]   = aroon_up
    df["aroon_down"] = aroon_down
    # Дельты индикаторов: направление изменения за 4 бара важнее абсолютного значения.
    # RSI пробивает 50 снизу вверх → сигнал смены тренда сильнее чем просто RSI=52.
    rsi_series        = pd.Series(talib.RSI(close, timeperiod=14))
    stoch_k_series    = pd.Series(slowk)
    macd_hist_series  = pd.Series(macd_hist)
    df["rsi_delta_4h"]       = (rsi_series - rsi_series.shift(4)).values
    df["stoch_k_delta_4h"]   = (stoch_k_series - stoch_k_series.shift(4)).values
    df["macd_hist_delta_4h"] = (macd_hist_series - macd_hist_series.shift(4)).values

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
    vol_sum_20 = vol_series.rolling(20).sum().replace(0, np.nan)
    df["cmf_20"] = (mfv_series.rolling(20).sum() / vol_sum_20).fillna(0.0).values
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

    # ── 52-недельные экстремумы ───────────────────────────────────────────────
    # Показывают режим рынка: у исторического максимума → импульс/перекупленность,
    # у минимума → перепроданность/разворот.
    # 252 бара ≈ 52 недели при 1h-свечах (250 торговых дней × ~7 часов / 7 ≈ 250 баров).
    # Для полноценного 52w нужно 252*7≈1764 баров — используем доступные данные rolling.
    close_s = pd.Series(close)
    high_252 = close_s.rolling(252, min_periods=50).max()
    low_252  = close_s.rolling(252, min_periods=50).min()
    df["high_252_ratio"] = np.where(high_252 != 0, close / high_252, np.nan)
    df["low_252_ratio"]  = np.where(low_252  != 0, close / low_252,  np.nan)

    # ── Donchian channel ──────────────────────────────────────────────────────
    # Breakout детектор: пробой 20-барного максимума/минимума.
    # donchian_pct = 1.0 → цена у верхней границы канала (бычий пробой),
    # donchian_pct = 0.0 → цена у нижней границы (медвежий пробой).
    high_s = pd.Series(high)
    low_s  = pd.Series(low)
    don_high  = high_s.rolling(20).max().values
    don_low   = low_s.rolling(20).min().values
    don_range = don_high - don_low
    df["donchian_pct"]   = np.where(don_range != 0, (close - don_low) / don_range, np.nan)
    df["donchian_width"] = np.where(close != 0, don_range / close, np.nan)

    # ── Роллинговые статистики доходностей ───────────────────────────────────
    # return_mean_8h > 0 = краткосрочный восходящий тренд доходностей;
    # return_std_8h = мини-волатильность (дополняет hist_vol_20 на коротком горизонте).
    returns_8h = pd.Series(close).pct_change()
    df["return_mean_8h"] = returns_8h.rolling(8).mean().values
    df["return_std_8h"]  = returns_8h.rolling(8).std().values

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

    # ── USD/RUB курс ─────────────────────────────────────────────────────────
    # Если данные USD/RUB не загружены — graceful degradation: признаки = 0.0.
    # Нормализация через SMA20 убирает долгосрочный тренд и делает признак стационарным.
    if "usdrub_close" in df.columns:
        usdrub = df["usdrub_close"].values.astype(np.float64)
        usdrub_sma20 = talib.SMA(usdrub, timeperiod=20)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = usdrub / usdrub_sma20
        df["usdrub_ratio"] = np.where(usdrub_sma20 > 0, ratio, 1.0)
        df["usdrub_change_1h"] = pd.Series(usdrub).pct_change(1).fillna(0.0).values
    else:
        df["usdrub_ratio"] = 0.0
        df["usdrub_change_1h"] = 0.0

    # ── Режим рынка и подтверждение объёма ───────────────────────────────────
    # Все окна берутся из config/settings.py — не хардкодятся.
    autocorr_window      = ml_settings.autocorr_window
    price_vol_corr_window = ml_settings.price_vol_corr_window
    williams_r_period    = ml_settings.williams_r_period
    cci_delta_period     = ml_settings.cci_delta_period

    # Автокорреляция доходностей (лаг 1): скользящее окно autocorr_window баров.
    # Pandas Series.rolling().corr(shift(1)) вычисляет Pearson r(x_t, x_{t-1}).
    # > 0 → последовательные доходности схожи (импульс), < 0 → чередуются (mean-reversion).
    # NaN в первых autocorr_window+1 строках — удаляются при dropna.
    returns_autocorr = pd.Series(close).pct_change()
    df["autocorr_returns"] = (
        returns_autocorr.rolling(autocorr_window)
        .corr(returns_autocorr.shift(1))
        .values
    )

    # Корреляция доходность × изменение объёма за price_vol_corr_window баров.
    # Pearson r(return_1h, volume_change_1h) по скользящему окну.
    # > 0 → объём растёт когда цена растёт (подтверждение), < 0 → дивергенция (слабый сигнал).
    vol_change_for_corr = pd.Series(volume).pct_change()
    df["price_vol_corr"] = (
        returns_autocorr.rolling(price_vol_corr_window)
        .corr(vol_change_for_corr)
        .values
    )

    # Williams %R через TA-Lib: диапазон [-100, 0].
    # -20..0 = перекуплен (цена у максимума диапазона), -80..-100 = перепродан.
    # Отличается от Stochastic %K знаком: WILLR = -100 * (high_max - close) / (high_max - low_min).
    df["williams_r"] = talib.WILLR(high, low, close, timeperiod=williams_r_period)

    # Дельта CCI за cci_delta_period баров: направление изменения CCI важнее абсолютного значения.
    # CCI пробивает +100 снизу вверх → вход в зону перекупленности → сильный бычий сигнал.
    cci_series = pd.Series(talib.CCI(high, low, close, timeperiod=14))
    df["cci_delta"] = (cci_series - cci_series.shift(cci_delta_period)).values

    # Удаляем строки где хотя бы один признак NaN (период прогрева, ~50 строк на тикер)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return df
