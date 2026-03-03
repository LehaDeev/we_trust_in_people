"""
CLI-точка входа для запуска 3-агентного конвейера.

Использование:
    python -m scripts.run_agents --task "Описание задачи"
    python -m scripts.run_agents --task-file task.txt
    python -m scripts.run_agents --task "..." --context "контекст"
    python -m scripts.run_agents --task "..." --context-file context.py

Результат:
    Выводит финальный код, результаты ревью и архитектурной проверки.
    Если архитектор предлагает обновление CLAUDE.md — выводит его отдельно.
"""
import argparse
import asyncio
import sys

from agents.pipeline import AgentPipeline
from utils.logger import logger


def _parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Запустить 3-агентный конвейер: Coder → Reviewer → Architect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python -m scripts.run_agents --task "Написать async функцию get_balance() -> Decimal"
  python -m scripts.run_agents --task-file task.txt --context-file tinkoff/portfolio.py
        """,
    )

    # Источник задания
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task",
        type=str,
        help="Описание задачи напрямую в строке",
    )
    task_group.add_argument(
        "--task-file",
        type=str,
        metavar="FILE",
        help="Путь к файлу с описанием задачи",
    )

    # Дополнительный контекст (опционально)
    context_group = parser.add_mutually_exclusive_group()
    context_group.add_argument(
        "--context",
        type=str,
        help="Контекст напрямую в строке (существующий код, требования)",
    )
    context_group.add_argument(
        "--context-file",
        type=str,
        metavar="FILE",
        help="Путь к файлу с контекстом",
    )

    return parser.parse_args()


def _read_file(path: str) -> str:
    """Прочитать содержимое файла."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Ошибка: файл не найден: {path}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Ошибка чтения файла {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _print_separator(title: str) -> None:
    """Вывести разделитель с заголовком."""
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


async def main() -> None:
    """Точка входа: разобрать аргументы и запустить конвейер."""
    args = _parse_args()

    # Получаем задание
    task: str = args.task if args.task else _read_file(args.task_file)

    # Получаем контекст (опционально)
    context: str | None = None
    if args.context:
        context = args.context
    elif args.context_file:
        context = _read_file(args.context_file)

    print(f"\nЗадание: {task[:200]}{'...' if len(task) > 200 else ''}")
    if context:
        print(f"Контекст: {len(context)} символов")

    # Запускаем конвейер
    pipeline = AgentPipeline()
    try:
        result = await pipeline.run(task, context)
    except ValueError as e:
        print(f"\nОшибка конфигурации: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error("Ошибка конвейера", error=str(e))
        print(f"\nОшибка: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Вывод результатов ─────────────────────────────────────────────────────

    _print_separator("ФИНАЛЬНЫЙ КОД")
    print(result.final_code)

    _print_separator("РЕЗУЛЬТАТЫ РЕВЬЮ")
    status = "✓ ОДОБРЕН" if result.approved_by_reviewer else "✗ НЕ ОДОБРЕН"
    print(f"Статус: {status} (раундов: {result.review_rounds})")
    if result.last_review.issues:
        print("\nОставшиеся замечания:")
        for issue in result.last_review.issues:
            print(f"  • {issue}")
    else:
        print("Замечаний нет.")

    _print_separator("АРХИТЕКТУРНАЯ ПРОВЕРКА")
    arch_status = "✓ СООТВЕТСТВУЕТ" if result.arch_valid else "✗ НАРУШЕНИЯ"
    print(f"Статус: {arch_status}")
    if result.arch_violations:
        print("\nНарушения:")
        for violation in result.arch_violations:
            print(f"  • {violation}")
    else:
        print("Нарушений нет.")

    if result.doc_update:
        _print_separator("ПРЕДЛОЖЕНИЕ ПО ОБНОВЛЕНИЮ CLAUDE.md")
        print(result.doc_update)

    print()

    # Возвращаем код выхода: 0 если всё ОК, 1 если есть проблемы
    if not result.approved_by_reviewer or not result.arch_valid:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
