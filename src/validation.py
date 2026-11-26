"""Data and assumption validation layer for financial statements and DCF inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding for financial data or model assumptions."""

    severity: str  # "error", "warning", "info"
    category: str
    field: str
    message: str


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    try:
        return bool(np.isnan(float(val)))
    except (TypeError, ValueError):
        return True


def validate_company_data(
    market_data: dict[str, Any],
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    is_financial_institution: bool = False,
) -> list[ValidationIssue]:
    """Validate retrieved market data and historical financial statements.

    Checks for missing values, currency mismatches, suspicious magnitudes,
    negative cash flows, and insufficient historical coverage.
    """
    issues: list[ValidationIssue] = []

    # 1. Market data checks
    price = market_data.get("current_price")
    shares = market_data.get("shares_outstanding")
    market_cap = market_data.get("market_cap")
    beta = market_data.get("beta")
    trade_ccy = market_data.get("currency")
    fin_ccy = market_data.get("financial_currency")

    if _is_missing(price) or float(price) <= 0:
        issues.append(
            ValidationIssue(
                severity="error",
                category="Market Data",
                field="current_price",
                message="Current stock price is unavailable or non-positive. Enter it manually to compare valuation.",
            )
        )

    if _is_missing(shares) or float(shares) <= 0:
        issues.append(
            ValidationIssue(
                severity="error",
                category="Market Data",
                field="shares_outstanding",
                message="Shares outstanding is missing or non-positive. Intrinsic value per share requires a positive share count.",
            )
        )

    if _is_missing(market_cap) or float(market_cap) <= 0:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Market Data",
                field="market_cap",
                message="Market capitalization is unavailable; WACC equity weight will rely on Price × Shares if available.",
            )
        )
    elif not _is_missing(price) and float(price) > 0 and not _is_missing(shares) and float(shares) > 0:
        implied_mcap = float(price) * float(shares)
        rel_diff = abs(float(market_cap) - implied_mcap) / float(market_cap)
        if rel_diff > 0.15:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="Unit Consistency",
                    field="market_cap",
                    message=(
                        f"Reported Market Cap ({float(market_cap):,.0f}) differs from Price × Shares "
                        f"({implied_mcap:,.0f}) by {rel_diff:.1%}. Verify share count and currency units."
                    ),
                )
            )

    if _is_missing(beta):
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Market Data",
                field="beta",
                message="Reported Beta is unavailable. Please supply an appropriate Beta in the WACC panel.",
            )
        )
    elif float(beta) < 0 or float(beta) > 4.0:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Market Data",
                field="beta",
                message=f"Reported Beta ({float(beta):.2f}) is outside typical [0, 4.0] range; verify before relying on CAPM.",
            )
        )
    elif float(beta) < 0.4:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Market Data",
                field="beta",
                message=(
                    f"Reported Beta ({float(beta):.2f}) is unusually low (< 0.40), which yields a very low CAPM "
                    "Cost of Equity. Consider overriding Beta with a sector/peer beta in the sidebar."
                ),
            )
        )

    if trade_ccy and fin_ccy and trade_ccy.upper() != fin_ccy.upper():
        issues.append(
            ValidationIssue(
                severity="error",
                category="Currency",
                field="financial_currency",
                message=(
                    f"Currency mismatch detected: stock trades in {trade_ccy} while financial statements are "
                    f"reported in {fin_ccy}. Do not mix currencies without manual conversion."
                ),
            )
        )

    # 2. Historical depth check
    n_years = len(income_df.columns) if not income_df.empty else 0
    if n_years == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                category="Financial Statements",
                field="income_statement",
                message="No historical income statement periods were retrieved.",
            )
        )
        return issues
    elif n_years < 3:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Financial Statements",
                field="history_length",
                message=f"Only {n_years} historical period(s) available (3–5 years preferred).",
            )
        )

    # Latest column (assumed chronologically sorted oldest -> newest)
    latest_col = income_df.columns[-1]

    def _latest(df: pd.DataFrame, row: str) -> Optional[float]:
        if df.empty or row not in df.index:
            return None
        col = latest_col if latest_col in df.columns else df.columns[-1]
        val = df.loc[row, col]
        return None if _is_missing(val) else float(val)

    rev = _latest(income_df, "Revenue")
    ebit = _latest(income_df, "Operating Income / EBIT")
    net_inc = _latest(income_df, "Net Income")
    tax_exp = _latest(income_df, "Income Tax Expense")

    da = _latest(cashflow_df, "Depreciation & Amortization")
    capex = _latest(cashflow_df, "Capital Expenditure")
    ocf = _latest(cashflow_df, "Operating Cash Flow")
    fcf = _latest(cashflow_df, "Free Cash Flow")

    cash = _latest(balance_df, "Total Cash & Equivalents")
    debt = _latest(balance_df, "Total Debt")
    equity = _latest(balance_df, "Total Equity")

    required_checks = [
        ("Revenue", rev, "error"),
        ("Net Income", net_inc, "warning"),
        ("Total Cash & Equivalents", cash, "warning"),
        ("Total Debt", debt, "warning"),
    ]
    if not is_financial_institution:
        required_checks.extend(
            [
                ("Operating Income / EBIT", ebit, "error"),
                ("Depreciation & Amortization", da, "warning"),
                ("Capital Expenditure", capex, "warning"),
            ]
        )

    for label, val, sev in required_checks:
        if val is None:
            issues.append(
                ValidationIssue(
                    severity=sev,
                    category="Missing Metric",
                    field=label,
                    message=f"Latest reported '{label}' is unavailable. Enter a manual value if needed.",
                )
            )

    # 3. Consistency & suspicious data checks
    if rev is not None and rev <= 0:
        issues.append(
            ValidationIssue(
                severity="error",
                category="Consistency",
                field="Revenue",
                message="Latest reported Revenue is zero or negative.",
            )
        )

    if ebit is not None and ebit < 0:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Profitability",
                field="Operating Income / EBIT",
                message="Latest Operating Income / EBIT is negative. Forecast margin assumptions should reflect expected recovery or loss trajectory.",
            )
        )

    if fcf is not None and fcf < 0:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Cash Flow",
                field="Free Cash Flow",
                message="Latest historical Free Cash Flow is negative.",
            )
        )

    if rev is not None and rev > 0 and capex is not None and capex > rev:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Suspicious Value",
                field="Capital Expenditure",
                message="Latest CapEx exceeds 100% of Revenue; verify whether this reflects a one-time expansion cycle.",
            )
        )

    if cash is not None and cash < 0:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Suspicious Value",
                field="Total Cash & Equivalents",
                message="Reported Cash & Equivalents is negative; verify balance sheet data.",
            )
        )

    if debt is not None and debt < 0:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Suspicious Value",
                field="Total Debt",
                message="Reported Total Debt is negative; verify balance sheet data.",
            )
        )

    if equity is not None and equity < 0:
        issues.append(
            ValidationIssue(
                severity="info",
                category="Balance Sheet",
                field="Total Equity",
                message="Book Stockholders' Equity is negative (often due to share buybacks or accumulated losses).",
            )
        )

    # Check Revenue growth swings across historical series
    if "Revenue" in income_df.index and len(income_df.columns) >= 2:
        rev_series = pd.to_numeric(income_df.loc["Revenue"], errors="coerce")
        pct_changes = rev_series.pct_change().dropna()
        if (pct_changes < 0).any():
            issues.append(
                ValidationIssue(
                    severity="info",
                    category="Historical Trend",
                    field="Revenue Growth",
                    message="Company experienced negative revenue growth in at least one historical year.",
                )
            )
        if (pct_changes > 2.0).any() or (pct_changes < -0.7).any():
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="Suspicious Value",
                    field="Revenue Growth",
                    message="Extreme YoY revenue swing (> +200% or < -70%) detected; default growth forecast trims outliers.",
                )
            )

    if ocf is not None and capex is not None and fcf is not None:
        implied_fcf = ocf - capex
        if abs(implied_fcf) > 0 and abs(fcf - implied_fcf) / max(abs(implied_fcf), 1.0) > 0.15:
            issues.append(
                ValidationIssue(
                    severity="info",
                    category="Cash Flow",
                    field="Free Cash Flow",
                    message="Reported FCF differs slightly from OCF - CapEx (possibly due to CapEx sign or lease/other adjustments).",
                )
            )

    if tax_exp is not None and ebit is not None and ebit > 0 and tax_exp < 0:
        issues.append(
            ValidationIssue(
                severity="info",
                category="Tax",
                field="Income Tax Expense",
                message="Latest reported tax expense is negative (tax benefit); verify effective tax rate assumption.",
            )
        )

    return issues


def validate_dcf_inputs(
    discount_rate: float,
    terminal_growth: float,
    shares_outstanding: Optional[float],
    terminal_value: Optional[float] = None,
    pv_terminal_value: Optional[float] = None,
    total_value: Optional[float] = None,
) -> list[ValidationIssue]:
    """Validate WACC/Ke, terminal growth rate, share count, and terminal value share."""
    issues: list[ValidationIssue] = []

    if shares_outstanding is None or _is_missing(shares_outstanding) or shares_outstanding <= 0:
        issues.append(
            ValidationIssue(
                severity="error",
                category="Valuation Inputs",
                field="shares_outstanding",
                message="Shares outstanding must be strictly positive (> 0) to compute Intrinsic Value Per Share.",
            )
        )

    if discount_rate <= 0:
        issues.append(
            ValidationIssue(
                severity="error",
                category="Discount Rate",
                field="discount_rate",
                message="Discount rate (WACC / Cost of Equity) must be positive.",
            )
        )

    if discount_rate <= terminal_growth:
        issues.append(
            ValidationIssue(
                severity="error",
                category="Terminal Value",
                field="terminal_growth",
                message=(
                    f"Discount rate ({discount_rate:.2%}) must be strictly greater than Terminal Growth Rate "
                    f"({terminal_growth:.2%}) for the Gordon Growth Model to be valid."
                ),
            )
        )

    if terminal_growth > 0.06:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Terminal Value",
                field="terminal_growth",
                message=(
                    f"Terminal growth rate ({terminal_growth:.2%}) exceeds 6.0%. Perpetual growth above long-term "
                    "nominal GDP growth can overstate Terminal Value."
                ),
            )
        )
    elif terminal_growth < -0.02:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Terminal Value",
                field="terminal_growth",
                message=f"Terminal growth rate ({terminal_growth:.2%}) is below -2.0%; confirm perpetual contraction is intended.",
            )
        )

    if terminal_value is not None and terminal_value < 0:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Terminal Value",
                field="terminal_value",
                message="Terminal Value is negative because Final Year FCF is negative.",
            )
        )

    if (
        pv_terminal_value is not None
        and total_value is not None
        and total_value > 0
        and (pv_terminal_value / total_value) > 0.75
    ):
        share_pct = (pv_terminal_value / total_value) * 100.0
        issues.append(
            ValidationIssue(
                severity="warning",
                category="Terminal Value",
                field="tv_share",
                message=(
                    f"Terminal Value accounts for {share_pct:.1f}% of total estimated value (>75%). "
                    "Valuation is highly sensitive to long-term growth and discount rate assumptions."
                ),
            )
        )

    return issues
