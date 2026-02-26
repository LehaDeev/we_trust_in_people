"""
Project settings loaded from .env file.
Uses pydantic-settings for validation and type safety.
"""
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TinkoffSettings(BaseSettings):
    token: str = Field(..., alias="TINKOFF_TOKEN")
    account_id: str = Field(..., alias="TINKOFF_ACCOUNT_ID")
    sandbox: bool = Field(True, alias="TINKOFF_SANDBOX")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class TelegramSettings(BaseSettings):
    bot_token: str = Field(..., alias="TELEGRAM_BOT_TOKEN")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class PostgresSettings(BaseSettings):
    host: str = Field("localhost", alias="POSTGRES_HOST")
    port: int = Field(5432, alias="POSTGRES_PORT")
    db: str = Field("we_trust_db", alias="POSTGRES_DB")
    user: str = Field("postgres", alias="POSTGRES_USER")
    password: str = Field(..., alias="POSTGRES_PASSWORD")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def dsn(self) -> str:
        """Async DSN for SQLAlchemy + asyncpg (special chars URL-encoded)."""
        pwd = quote_plus(self.password)
        return (
            f"postgresql+asyncpg://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def asyncpg_dsn(self) -> str:
        """Raw asyncpg DSN (without SQLAlchemy prefix, special chars URL-encoded)."""
        pwd = quote_plus(self.password)
        return (
            f"postgresql://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class AppSettings(BaseSettings):
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    debug: bool = Field(False, alias="DEBUG")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Singleton instances — import these in other modules
tinkoff_settings = TinkoffSettings()
telegram_settings = TelegramSettings()
postgres_settings = PostgresSettings()
app_settings = AppSettings()
