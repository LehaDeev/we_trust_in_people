"""
Хендлер ML-сигналов: запрос сигнала по тикеру, обновление.
"""
import json
from pathlib import Path

from aiogram import Router
from aiogram.types import CallbackQuery

from bot.keyboards import signal_actions
from config.settings import ml_settings, trading_settings
from ml.predict import predict_signal
from utils.logger import logger

router = Router(name="signals")

_SIGNAL_EMOJI = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
_WEIGHTS_DIR = Path("ml/weights")


def _ticker_threshold(ticker: str) -> float:
    """Вернуть per-ticker порог уверенности (или глобальный если файл не найден)."""
    path = _WEIGHTS_DIR / f"best_threshold_{ticker}_{ml_settings.model_version}.json"
    try:
        with open(path) as f:
            return json.load(f)["threshold"]
    except Exception:
        return trading_settings.confidence_threshold


def _format_signal(result: dict) -> str:
    """
    Форматировать результат predict_signal в читаемый текст.

    Пример вывода:
        📈 SBER
        Сигнал:   🟢 BUY
        P&L пред: +0.83%  |  порог: +0.30%  ✅
        Объём:    1.24× среднего
    """
    ticker = result["ticker"]
    signal = result["signal"]
    # confidence — предсказанный net P&L (доля), например 0.0083 = 0.83%
    confidence: float = result.get("confidence", 0.0)
    volume_ratio: float = result.get("volume_ratio", 1.0)

    threshold = _ticker_threshold(ticker)
    passes_threshold = confidence >= threshold
    threshold_mark = "✅" if passes_threshold else "❌"

    volume_min = trading_settings.volume_min_ratio
    passes_volume = volume_ratio >= volume_min
    volume_mark = "✅" if passes_volume else "❌"
    # Если фильтр объёма отключён (=1.0) — не показываем метку
    volume_suffix = f"  {volume_mark}" if volume_min > 1.0 else ""

    emoji = _SIGNAL_EMOJI.get(signal, "⚪")

    # Процент от порога: насколько текущий сигнал близок к срабатыванию
    if threshold != 0:
        pct_of_threshold = confidence / threshold * 100
    else:
        pct_of_threshold = 100.0 if confidence >= 0 else 0.0
    gap = confidence - threshold
    gap_sign = "+" if gap >= 0 else ""
    gap_line = "Порог пройден" if passes_threshold else f"До порога: <b>{gap * 100:.2f}%</b>"

    lines = [
        f"📈 <b>{ticker}</b>",
        f"Сигнал:   {emoji} <b>{signal}</b>",
        f"Ранг:     <b>{pct_of_threshold:.0f}%</b> от порога  {threshold_mark}",
        gap_line,
        f"Объём:    <b>{volume_ratio:.2f}×</b> среднего{volume_suffix}",
    ]
    return "\n".join(lines)


async def _get_and_send_signal(callback: CallbackQuery, ticker: str) -> None:
    """Запросить сигнал и обновить сообщение."""
    await callback.answer("⏳ Считаю сигнал...")
    try:
        result = await predict_signal(ticker)
        text = _format_signal(result)
        markup = signal_actions(ticker)
    except FileNotFoundError:
        text = (
            f"⚠️ <b>{ticker}</b>\n\n"
            "Веса модели не найдены.\n"
            "Запустите: <code>python -m scripts.train_model</code>"
        )
        markup = signal_actions(ticker)
    except ValueError as e:
        text = f"⚠️ <b>{ticker}</b>\n\n{e}"
        markup = signal_actions(ticker)
    except Exception as e:
        logger.error("Signal prediction error", ticker=ticker, error=str(e))
        text = f"❌ Ошибка при получении сигнала для <b>{ticker}</b>."
        markup = signal_actions(ticker)

    await callback.message.edit_text(text, reply_markup=markup, parse_mode="HTML")


@router.callback_query(lambda c: c.data and c.data.startswith("signal:"))
async def cb_signal(callback: CallbackQuery) -> None:
    """Показать сигнал для выбранного тикера."""
    ticker = callback.data.split(":", 1)[1]
    await _get_and_send_signal(callback, ticker)


@router.callback_query(lambda c: c.data and c.data.startswith("signal_refresh:"))
async def cb_signal_refresh(callback: CallbackQuery) -> None:
    """Обновить сигнал для текущего тикера."""
    ticker = callback.data.split(":", 1)[1]
    await _get_and_send_signal(callback, ticker)
