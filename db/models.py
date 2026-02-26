"""
SQLAlchemy ORM models for the project.
Add new models here as the project grows.
"""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base


class Asset(Base):
    """Tracked financial instrument (stock, bond, ETF)."""
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(primary_key=True)
    figi: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    name: Mapped[str] = mapped_column(String(255))
    currency: Mapped[str] = mapped_column(String(10), default="RUB")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    candles: Mapped[list["Candle"]] = relationship(back_populates="asset")
    signals: Mapped[list["Signal"]] = relationship(back_populates="asset")


class Candle(Base):
    """OHLCV candle data for an asset."""
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), index=True)
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    high: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    low: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    close: Mapped[Decimal] = mapped_column(Numeric(18, 6))
    volume: Mapped[int]
    interval: Mapped[str] = mapped_column(String(10))  # e.g. "1h", "1d"

    asset: Mapped["Asset"] = relationship(back_populates="candles")


class Signal(Base):
    """ML model prediction signal for an asset."""
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    signal_type: Mapped[str] = mapped_column(String(10))  # "BUY", "SELL", "HOLD"
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4))  # 0.0000 - 1.0000
    model_version: Mapped[str] = mapped_column(String(50))
    price_at_signal: Mapped[Decimal] = mapped_column(Numeric(18, 6))

    asset: Mapped["Asset"] = relationship(back_populates="signals")
