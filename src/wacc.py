"""Weighted Average Cost of Capital (WACC) and CAPM Cost of Equity calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import SETTINGS


@dataclass(frozen=True)
class CountryWACCPreset:
    """Default risk-free rate and equity risk premium presets by market."""

    name: str
    risk_free_rate: float
    equity_risk_premium: float
    default_cost_of_debt: float
    default_tax_rate: float
    default_terminal_growth: float


COUNTRY_PRESETS: dict[str, CountryWACCPreset] = {
    "US": CountryWACCPreset(
        name="United States (USD)",
        risk_free_rate=SETTINGS.us_risk_free_rate,
        equity_risk_premium=SETTINGS.us_erp,
        default_cost_of_debt=0.055,
        default_tax_rate=0.21,
        default_terminal_growth=SETTINGS.us_terminal_growth,
    ),
    "IN": CountryWACCPreset(
        name="India (INR)",
        risk_free_rate=SETTINGS.in_risk_free_rate,
        equity_risk_premium=SETTINGS.in_erp,
        default_cost_of_debt=0.085,
        default_tax_rate=0.25,
        default_terminal_growth=SETTINGS.in_terminal_growth,
    ),
}


@dataclass(frozen=True)
class WACCInputs:
    """Explicit inputs for CAPM and WACC calculation."""

    risk_free_rate: float
    beta: float
    equity_risk_premium: float
    pre_tax_cost_of_debt: float
    tax_rate: float
    market_cap_equity: float  # Must be market value of equity, NOT book equity
    total_debt: float


@dataclass(frozen=True)
class WACCResult:
    """Transparent breakdown of Cost of Equity, Cost of Debt, Capital Weights, and WACC."""

    cost_of_equity: float
    after_tax_cost_of_debt: float
    equity_value: float
    debt_value: float
    total_capital: float
    equity_weight: float
    debt_weight: float
    wacc: float
    formula_summary: str


def calculate_cost_of_equity(risk_free_rate: float, beta: float, equity_risk_premium: float) -> float:
    """Calculate Cost of Equity using CAPM: Ke = Rf + Beta * ERP."""
    return risk_free_rate + beta * equity_risk_premium


def calculate_wacc(inputs: WACCInputs) -> WACCResult:
    """Calculate Weighted Average Cost of Capital using market value capital weights.

    Formula:
        WACC = (E / (D + E)) * Ke + (D / (D + E)) * Kd * (1 - Tax Rate)
    where E is Market Capitalization (never book equity) and D is Total Debt.
    """
    if inputs.market_cap_equity < 0 or inputs.total_debt < 0:
        raise ValueError("Market value of equity and total debt must be non-negative.")

    total_capital = inputs.market_cap_equity + inputs.total_debt
    if total_capital <= 0:
        raise ValueError("Total capital (Market Value of Equity + Total Debt) must be strictly positive.")

    if not (0.0 <= inputs.tax_rate <= 1.0):
        raise ValueError("Tax rate must be between 0% and 100% (0.0 to 1.0).")

    ke = calculate_cost_of_equity(inputs.risk_free_rate, inputs.beta, inputs.equity_risk_premium)
    kd_after = inputs.pre_tax_cost_of_debt * (1.0 - inputs.tax_rate)

    equity_weight = inputs.market_cap_equity / total_capital
    debt_weight = inputs.total_debt / total_capital

    wacc = equity_weight * ke + debt_weight * kd_after

    formula_summary = (
        f"Ke = {inputs.risk_free_rate:.2%} + {inputs.beta:.2f} × {inputs.equity_risk_premium:.2%} = {ke:.2%}\n"
        f"Kd (after-tax) = {inputs.pre_tax_cost_of_debt:.2%} × (1 - {inputs.tax_rate:.2%}) = {kd_after:.2%}\n"
        f"WACC = ({equity_weight:.1%} × {ke:.2%}) + ({debt_weight:.1%} × {kd_after:.2%}) = {wacc:.2%}"
    )

    return WACCResult(
        cost_of_equity=ke,
        after_tax_cost_of_debt=kd_after,
        equity_value=inputs.market_cap_equity,
        debt_value=inputs.total_debt,
        total_capital=total_capital,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        wacc=wacc,
        formula_summary=formula_summary,
    )


def estimate_pre_tax_cost_of_debt(
    interest_expense: Optional[float],
    total_debt: Optional[float],
    fallback_rate: float,
) -> tuple[float, str]:
    """Estimate pre-tax cost of debt from Interest Expense / Total Debt when reasonable, else fallback."""
    if (
        interest_expense is not None
        and total_debt is not None
        and total_debt > 0
        and interest_expense > 0
    ):
        implied = abs(interest_expense) / total_debt
        if 0.01 <= implied <= 0.25:
            return float(implied), "Estimated from Interest Expense / Total Debt"
    return fallback_rate, "Country preset default (editable)"
