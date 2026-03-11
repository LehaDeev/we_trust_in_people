"""
Вывод метрик последнего обучения ML-моделей.

Запуск:
    python -m scripts.show_results
"""
import json
import zoneinfo
from datetime import datetime
from pathlib import Path

RESULTS_PATH = Path("ml/weights/last_results.json")
RETRAIN_TZ = zoneinfo.ZoneInfo("Europe/Moscow")


def _nightly_status(trained_at_str: str, force_tune: bool) -> str:
    """Определить статус ночного дообучения относительно сегодняшнего дня."""
    try:
        trained_at = datetime.fromisoformat(trained_at_str)
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
        trained_msk = trained_at.astimezone(RETRAIN_TZ)
        today_msk = datetime.now(RETRAIN_TZ).date()

        if trained_msk.date() == today_msk and not force_tune:
            return "ДА ✓  (ночной, сегодня)"
        elif trained_msk.date() == today_msk and force_tune:
            return "ДА ✓  (ручной с Optuna, сегодня)"
        else:
            hours_ago = (datetime.now(RETRAIN_TZ) - trained_msk).total_seconds() / 3600
            return f"НЕТ ⚠  (последнее {trained_msk.strftime('%d.%m %H:%M')} МСК, {hours_ago:.0f}ч назад)"
    except Exception:
        return "неизвестно"


def main() -> None:
    """Прочитать и вывести таблицу F1 из последнего обучения."""
    if not RESULTS_PATH.exists():
        print("Файл метрик не найден. Запустите train_model хотя бы один раз.")
        return

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    trained_at: str = data["trained_at"]
    force_tune: bool = data["force_tune"]

    print(f"\nДата обучения   : {trained_at}")
    print(f"Ночное сегодня  : {_nightly_status(trained_at, force_tune)}")
    print(f"force_tune      : {force_tune}")
    print()
    print(f"{'Тикер':<8} {'F1':>8}")
    print("-" * 18)

    f1_scores: dict[str, float] = data["f1_scores"]
    for ticker, f1 in sorted(f1_scores.items()):
        print(f"{ticker:<8} {f1:>8.4f}")

    if f1_scores:
        avg = sum(f1_scores.values()) / len(f1_scores)
        print("-" * 18)
        print(f"{'Среднее':<8} {avg:>8.4f}")

    if data["failed"]:
        print(f"\nОшибки: {', '.join(data['failed'])}")


if __name__ == "__main__":
    main()
