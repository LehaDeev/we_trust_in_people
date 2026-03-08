"""
Вывод метрик последнего обучения ML-моделей.

Запуск:
    python -m scripts.show_results
"""
import json
from pathlib import Path

RESULTS_PATH = Path("ml/weights/last_results.json")


def main() -> None:
    """Прочитать и вывести таблицу F1 из последнего обучения."""
    if not RESULTS_PATH.exists():
        print("Файл метрик не найден. Запустите train_model хотя бы один раз.")
        return

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    print(f"\nДата обучения : {data['trained_at']}")
    print(f"force_tune    : {data['force_tune']}")
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
