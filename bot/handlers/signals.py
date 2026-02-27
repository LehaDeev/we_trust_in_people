"""
Хендлер ML-сигналов: запрос сигнала по тикеру, обновление.
"""
from aiogram import Router
from aiogram.types import CallbackQuery

from bot.keyboards import signal_actions
from ml.predict import predict_signal
from utils.logger import logger

router = Router(name="signals")

_SIGNAL_EMOJI = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}


def _format_signal(result: dict) -> str:
    """
    Форматировать результат predict_signal в читаемый текст.

    Пример вывода:
        📈 SBER
        Сигнал: 🟢 BUY
        Уверенность: 72.4%

        SELL: 12.3% | HOLD: 15.3% | BUY: 72.4%
    """
    ticker = result["ticker"]
    signal = result["signal"]
    confidence = result["confidence"] * 100
    proba = result["probabilities"]

    emoji = _SIGNAL_EMOJI.get(signal, "⚪")

    lines = [
        f"📈 <b>{ticker}</b>",
        f"Сигнал: {emoji} <b>{signal}</b>",
        f"Уверенность: <b>{confidence:.1f}%</b>",
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
