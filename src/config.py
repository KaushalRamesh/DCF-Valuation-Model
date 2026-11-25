"""Configuration, environment variable loading, and currency formatting utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppSettings:
    """Application default parameters loaded from environment variables."""

    us_risk_free_rate: float = _get_float_env("DEFAULT_US_RISK_FREE_RATE", 0.042)
    in_risk_free_rate: float = _get_float_env("DEFAULT_IN_RISK_FREE_RATE", 0.070)
    us_erp: float = _get_float_env("DEFAULT_US_ERP", 0.045)
    in_erp: float = _get_float_env("DEFAULT_IN_ERP", 0.065)
    us_terminal_growth: float = _get_float_env("DEFAULT_US_TERMINAL_GROWTH", 0.025)
    in_terminal_growth: float = _get_float_env("DEFAULT_IN_TERMINAL_GROWTH", 0.040)
    cache_ttl_seconds: int = _get_int_env("DATA_CACHE_TTL_SECONDS", 900)


SETTINGS = AppSettings()

CURRENCY_SYMBOLS: dict[str, str] = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CAD": "C$",
    "AUD": "A$",
    "CHF": "CHF ",
    "HKD": "HK$",
    "SGD": "S$",
}


def get_currency_symbol(currency: Optional[str]) -> str:
    """Return display symbol for an ISO currency code."""
    if not currency:
        return ""
    code = currency.strip().upper()
    return CURRENCY_SYMBOLS.get(code, f"{code} ")


def is_indian_market(ticker: str, currency: Optional[str] = None, exchange: Optional[str] = None) -> bool:
    """Detect whether a security trades in India (NSE/BSE or INR)."""
    t = (ticker or "").strip().upper()
    c = (currency or "").strip().upper()
    e = (exchange or "").strip().upper()
    return t.endswith(".NS") or t.endswith(".BO") or c == "INR" or e in {"NSI", "NSE", "BSE", "BOM"}


def get_display_scale(currency: Optional[str], unit_mode: str = "auto") -> tuple[float, str]:
    """Return (divisor, unit_label) for financial statement scaling.

    For INR: 1e7 ('Cr' - Crore) when unit_mode is 'auto' or 'crore'.
    For USD/other: 1e6 ('M' - Millions) when unit_mode is 'auto' or 'millions'.
    """
    code = (currency or "USD").strip().upper()
    if unit_mode == "raw":
        return 1.0, ""
    if unit_mode == "crore" or (unit_mode == "auto" and code == "INR"):
        return 1e7, "Cr"
    if unit_mode == "billions":
        return 1e9, "B"
    return 1e6, "M"


def format_per_share(value: Optional[float], currency: Optional[str] = None, decimals: int = 2) -> str:
    """Format a per-share monetary amount (never scaled to millions/crores)."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "Unavailable"
    symbol = get_currency_symbol(currency)
    sign = "-" if value < 0 else ""
    return f"{sign}{symbol}{abs(value):,.{decimals}f}"


def format_large_currency(
    value: Optional[float],
    currency: Optional[str] = None,
    unit_mode: str = "auto",
    decimals: int = 2,
) -> str:
    """Format large financial statement numbers in Crore (INR) or Millions (USD/other)."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "Unavailable"
    symbol = get_currency_symbol(currency)
    divisor, suffix = get_display_scale(currency, unit_mode)
    scaled = value / divisor
    sign = "-" if scaled < 0 else ""
    unit_str = f" {suffix}" if suffix else ""
    return f"{sign}{symbol}{abs(scaled):,.{decimals}f}{unit_str}"


def format_percent(value: Optional[float], decimals: int = 2, is_decimal: bool = True) -> str:
    """Format a percentage value cleanly, returning 'Unavailable' for NaN/None."""
    if value is None or (isinstance(value, float) and (value != value)):
        return "Unavailable"
    pct = value * 100.0 if is_decimal else value
    return f"{pct:,.{decimals}f}%"
