"""
Модуль расчёта размера позиции (Position Sizing).

Поддерживаемые методы (TRADING_POSITION_SIZING):
    fixed_risk  — Fixed Fractional Risk: лоты масштабируются по формуле
                  lots = floor(balance × risk_pct / (sl_pct × price × lot_size)).
                  Гарантирует постоянный рублёвый риск на сделку независимо
                  от цены тикера и волатильности. Рекомендуется.
    fixed_lots  — Всегда lots_per_ticker лотов из настроек.
                  Обратная совместимость со старым поведением.

Три слоя position sizing (порядок применения в scheduler.py):
    1. compute_lots()              — базовый расчёт по методу fixed_risk / fixed_lots
    2. _apply_regime_filter()      — корректировка на режим рынка (аптренд / флет / даунтренд)
    3. apply_confidence_scaling()  — масштабирование по уверенности ML-сигнала (опционально)
"""
from decimal import Decimal

from config.settings import trading_settings


def compute_lots(
    balance: Decimal,
    price: Decimal,
    lot_size: int,
    sl_pct: float,
) -> int:
    """
    Рассчитать число лотов по методу Fixed Fractional Risk (фиксированный % риска от депо).

    При TRADING_POSITION_SIZING='fixed_risk':
        риск_на_сделку   = balance × TRADING_RISK_PCT_PER_TRADE
        стоимость_позиции = риск_на_сделку / sl_pct
        лотов             = floor(стоимость_позиции / (price × lot_size))

    Результат зажат в диапазон [1, TRADING_MAX_LOTS_PER_TRADE].
    Метод гарантирует: при срабатывании SL потеря ≈ risk_pct_per_trade × balance,
    независимо от цены тикера и текущей волатильности.

    При TRADING_POSITION_SIZING='fixed_lots' или sl_pct <= 0 возвращает
    TRADING_LOTS_PER_TICKER (обратная совместимость).

    Аргументы:
        balance:  доступный рублёвый баланс (Decimal)
        price:    текущая цена одной бумаги (Decimal)
        lot_size: количество бумаг в одном лоте
        sl_pct:   стоп-лосс как доля от цены входа (например 0.025 = 2.5%)

    Возвращает:
        целое число лотов >= 1
    """
    ts = trading_settings
    if ts.position_sizing == "fixed_lots" or sl_pct <= 0:
        return ts.lots_per_ticker

    lot_value = price * Decimal(lot_size)
    if lot_value <= 0:
        return 1

    risk_rub = balance * Decimal(str(ts.risk_pct_per_trade))
    position_value = risk_rub / Decimal(str(sl_pct))
    raw_lots = int(position_value / lot_value)
    return max(1, min(raw_lots, ts.max_lots_per_trade))


def apply_confidence_scaling(lots: int, confidence: float) -> int:
    """
    Масштабировать размер позиции на основе уверенности ML-сигнала.

    Третий слой position sizing. Применяется ПОСЛЕ compute_lots() и _apply_regime_filter().
    При TRADING_CONFIDENCE_SCALING_ENABLED=false возвращает lots без изменений
    (полная обратная совместимость).

    Значение confidence — предсказанный net P&L из predict_signal() (доля от вложений,
    например 0.008 = 0.8%). Это НЕ вероятность: регрессор выдаёт rank-оценку ожидаемой
    доходности. Типичный диапазон BUY-сигналов: 0.003–0.025.

    Методы (TRADING_CONFIDENCE_SCALING_METHOD):
        tiered — три уровня уверенности → три фиксированных множителя (рекомендуется):
                 confidence < tier_low  → lots × mult_low  (слабый сигнал, меньше лотов)
                 tier_low <= c < tier_high → lots × mult_mid  (средний сигнал)
                 confidence >= tier_high → lots × mult_high (сильный сигнал, больше лотов)

        linear — линейное масштабирование пропорционально confidence:
                 при confidence = tier_low → mult = 1.0 (baseline);
                 ниже tier_low → < 1.0; выше tier_high → > 1.0.
                 Результат зажат в [mult_low, mult_high].

        kelly  — Half-Kelly: f = 0.5 × (confidence / tier_low).
                 tier_low выступает proxy для baseline expected return (edge).
                 Зажат в [mult_low, mult_high] для стабильности при неверном edge.

    Аргументы:
        lots:       число лотов после compute_lots() и _apply_regime_filter()
        confidence: predicted net P&L из predict_signal() (доля, 0.008 = 0.8%)

    Возвращает:
        скорректированное число лотов, зажатое в [1, TRADING_MAX_LOTS_PER_TRADE].
        При lots <= 0 возвращает lots без изменений (не вмешивается в блокировки режим-фильтра).
    """
    ts = trading_settings
    if not ts.confidence_scaling_enabled or lots <= 0:
        return lots

    method = ts.confidence_scaling_method

    if method == "tiered":
        if confidence < ts.confidence_tier_low:
            mult = ts.confidence_mult_low
        elif confidence < ts.confidence_tier_high:
            mult = ts.confidence_mult_mid
        else:
            mult = ts.confidence_mult_high

    elif method == "linear":
        # Линейное масштабирование: при confidence = tier_low → mult = 1.0.
        # Ниже tier_low → mult < 1.0 (уменьшаем лоты), выше tier_high → mult > 1.0.
        # Зажимаем в [mult_low, mult_high] для защиты от экстремальных значений.
        base = ts.confidence_tier_low if ts.confidence_tier_low > 0 else 1e-6
        raw_mult = confidence / base
        mult = max(ts.confidence_mult_low, min(ts.confidence_mult_high, raw_mult))

    elif method == "kelly":
        # Half-Kelly: f = 0.5 × (confidence / tier_low).
        # confidence ≈ edge (ожидаемый доход); tier_low ≈ минимальный приемлемый edge.
        # Коэффициент 0.5 снижает агрессивность Kelly при неточной оценке edge
        # (стандартная практика: Half-Kelly безопаснее Full-Kelly при estimation error).
        # Зажимаем в [mult_low, mult_high] для стабильности при выбросах confidence.
        base = ts.confidence_tier_low if ts.confidence_tier_low > 0 else 1e-6
        raw_mult = 0.5 * (confidence / base)
        mult = max(ts.confidence_mult_low, min(ts.confidence_mult_high, raw_mult))

    else:
        # Неизвестный метод — возвращаем лоты без изменений, не блокируем сделку
        return lots

    scaled = max(1, int(lots * mult))
    return min(scaled, ts.max_lots_per_trade)
