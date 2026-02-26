"""
Structured logging setup using structlog.
Import `logger` from this module everywhere in the project.
"""
import logging
import structlog
from config.settings import app_settings


def setup_logging() -> None:
    """Configure structlog with JSON output for production, colored for dev."""
    log_level = getattr(logging, app_settings.log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if app_settings.debug:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )


# Module-level logger — import this in other files
logger = structlog.get_logger(__name__)
