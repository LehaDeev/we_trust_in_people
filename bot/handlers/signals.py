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
        Сигнал: 🟢 BUY
        P(BUY):  72.4%  |  порог: 58.0%  ✅
        Объём:   1.24× среднего

        SELL: 12.3% | HOLD: 15.3% | BUY: 72.4%
    """
    ticker = result["ticker"]
    signal = result["signal"]
    proba = result["probabilities"]
    buy_proba = proba["BUY"] * 100
    volume_ratio: float = result.get("volume_ratio", 1.0)

    threshold = _ticker_threshold(ticker)
    threshold_pct = threshold * 100
    passes_threshold = proba["BUY"] >= threshold
    threshold_mark = "✅" if passes_threshold else "❌"

    volume_min = trading_settings.volume_min_ratio
    passes_volume = volume_ratio >= volume_min
    volume_mark = "✅" if passes_volume else "❌"
    # Если фильтр объёма отключён (=1.0) — не показываем метку
    volume_suffix = f"  {volume_mark}" if volume_min > 1.0 else ""

    emoji = _SIGNAL_EMOJI.get(signal, "⚪")

    lines = [
        f"📈 <b>{ticker}</b>",
        f"Сигнал:  {emoji} <b>{signal}</b>",
        f"P(BUY):  <b>{buy_proba:.1f}%</b>  |  порог: {threshold_pct:.1f}%  {threshold_mark}",
        f"Объём:   <b>{volume_ratio:.2f}×</b> среднего{volume_suffix}",
        "",
        f"SELL: {proba['SELL']*100:.1f}% | HOLD: {proba['HOLD']*100:.1f}% | BUY: {proba['BUY']*100:.1f}%",
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
