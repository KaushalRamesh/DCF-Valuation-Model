"""Industry classification and valuation methodology selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

FINANCIAL_SECTOR_KEYWORDS = (
    "financial services",
    "financial",
    "banking",
    "insurance",
)

FINANCIAL_INDUSTRY_KEYWORDS = (
    "bank",
    "banks",
    "insurance",
    "capital markets",
    "credit services",
    "financial",
    "mortgage",
    "asset management",
    "consumer finance",
    "brokerage",
    "diversified financials",
    "thrifts",
)


@dataclass(frozen=True)
class IndustryProfile:
    """Classification of a company's sector/industry for DCF methodology guidance."""

    sector: str
    industry: str
    is_financial_institution: bool
    recommended_method: str  # "FCFF" or "FCFE"
    warning_message: Optional[str]
    methodology_explanation: str


def get_methodology_explanation(methodology: str) -> str:
    """Return accurate explanation text for the given active valuation methodology."""
    if methodology == "FCFE":
        return (
            "FCFE Valuation Mode: Discounts projected Free Cash Flow to Equity (FCFE) and Terminal Value "
            "directly at Cost of Equity (CAPM) to estimate Equity Value without forcing an Enterprise Value "
            "bridge or subtracting debt again."
        )
    return (
        "FCFF Valuation Mode: Discounts projected Unlevered Free Cash Flow (UFCF) and Terminal Value at WACC "
        "to compute Enterprise Value, then bridges to Equity Value (+ Cash - Debt)."
    )


def detect_industry_profile(sector: Optional[str], industry: Optional[str]) -> IndustryProfile:
    """Determine whether a company requires standard FCFF DCF or financial-institution FCFE guidance."""
    sec = (sector or "Unavailable").strip()
    ind = (industry or "Unavailable").strip()
    sec_lower = sec.lower()
    ind_lower = ind.lower()

    is_fin = any(k in sec_lower for k in FINANCIAL_SECTOR_KEYWORDS) or any(
        k in ind_lower for k in FINANCIAL_INDUSTRY_KEYWORDS
    )

    if is_fin:
        return IndustryProfile(
            sector=sec,
            industry=ind,
            is_financial_institution=True,
            recommended_method="FCFE",
            warning_message=(
                f"Detected financial sector/industry ({sec} / {ind}). Standard Enterprise Value / FCFF DCF "
                "is typically unsuitable for banks, insurers, and financial institutions because debt is "
                "part of operating funding (deposits/borrowings) and CapEx/working capital definitions differ. "
                "FCFE (Free Cash Flow to Equity) discounted at the Cost of Equity is recommended."
            ),
            methodology_explanation=get_methodology_explanation("FCFE"),
        )

    return IndustryProfile(
        sector=sec,
        industry=ind,
        is_financial_institution=False,
        recommended_method="FCFF",
        warning_message=None,
        methodology_explanation=get_methodology_explanation("FCFF"),
    )
