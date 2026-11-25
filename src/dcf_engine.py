"""Core DCF valuation engine: discounting, Gordon Growth Terminal Value, EV/Equity bridge, and per-share value."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DCFValuationResult:
    """Complete, auditable output of a DCF valuation (FCFF or FCFE)."""

    methodology: str  # "FCFF" or "FCFE"
    discount_rate: float  # WACC for FCFF, Cost of Equity for FCFE
    terminal_growth_rate: float
    pv_forecast_cashflows: float
    terminal_value: float
    pv_terminal_value: float
    tv_pct_of_total_value: Optional[float]
    enterprise_value: Optional[float]  # None in FCFE mode (banks/financials)
    cash: Optional[float]
    total_debt: Optional[float]
    net_debt: Optional[float]
    non_operating_adjustments: float
    equity_value: float
    shares_outstanding: float
    intrinsic_value_per_share: float
    current_market_price: Optional[float]
    upside_absolute: Optional[float]
    upside_percent: Optional[float]
    margin_of_safety_pct: float
    suggested_buy_price: float
    valuation_label: str
    forecast_pv_table: pd.DataFrame


def classify_valuation_label(
    intrinsic_value: float,
    market_price: Optional[float],
    near_band_pct: float = 0.10,
) -> str:
    """Assign a non-absolute valuation comparison label relative to estimated intrinsic value.

    Never labels a stock as definitively 'undervalued' or 'overvalued'.
    """
    if market_price is None or market_price <= 0 or np.isnan(market_price):
        return "Market price unavailable"
    diff_ratio = (intrinsic_value - market_price) / market_price
    if diff_ratio > near_band_pct:
        return "Below estimated value"
    if diff_ratio < -near_band_pct:
        return "Above estimated value"
    return "Near estimated value"


def calculate_terminal_value_gordon(
    final_year_fcf: float,
    discount_rate: float,
    terminal_growth_rate: float,
) -> float:
    """Calculate Terminal Value via Gordon Growth Model.

    Formula:
        TV = Final Year FCF * (1 + g) / (r - g)
    """
    if discount_rate <= terminal_growth_rate:
        raise ValueError(
            f"Discount rate ({discount_rate:.2%}) must be strictly greater than "
            f"terminal growth rate ({terminal_growth_rate:.2%})."
        )
    return final_year_fcf * (1.0 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)


def run_fcff_dcf(
    forecast_df: pd.DataFrame,
    wacc: float,
    terminal_growth_rate: float,
    cash: float,
    total_debt: float,
    shares_outstanding: float,
    current_market_price: Optional[float] = None,
    margin_of_safety_pct: float = 0.20,
    non_operating_adjustments: float = 0.0,
) -> DCFValuationResult:
    """Execute standard Enterprise Value -> Equity Value FCFF DCF valuation.

    Formulas:
        PV(UFCF_t) = UFCF_t / (1 + WACC)^t
        TV = UFCF_N * (1 + g) / (WACC - g)
        PV(TV) = TV / (1 + WACC)^N
        Enterprise Value = sum(PV(UFCF_t)) + PV(TV)
        Equity Value = Enterprise Value + Cash - Total Debt + Non-Operating Adjustments
        Intrinsic Value / Share = Equity Value / Shares Outstanding
    """
    if forecast_df.empty or "UFCF" not in forecast_df.columns:
        raise ValueError("Forecast DataFrame must contain 'UFCF' values.")
    if shares_outstanding is None or shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be strictly positive (> 0).")
    if wacc <= 0:
        raise ValueError("WACC must be strictly positive (> 0).")
    if wacc <= terminal_growth_rate:
        raise ValueError("WACC must be strictly greater than Terminal Growth Rate.")
    if not (0.0 <= margin_of_safety_pct < 1.0):
        raise ValueError("Margin of safety must be between 0% and <100%.")

    df = forecast_df.copy()
    periods = df["Period"].astype(float).values if "Period" in df.columns else np.arange(1, len(df) + 1, dtype=float)
    ufcf_values = df["UFCF"].astype(float).values

    discount_factors = (1.0 + wacc) ** periods
    pv_ufcf = ufcf_values / discount_factors

    df["Discount Factor"] = discount_factors
    df["PV of FCF"] = pv_ufcf

    pv_forecast_total = float(np.sum(pv_ufcf))
    final_ufcf = float(ufcf_values[-1])
    n_years = float(periods[-1])

    terminal_value = calculate_terminal_value_gordon(final_ufcf, wacc, terminal_growth_rate)
    pv_terminal_value = terminal_value / ((1.0 + wacc) ** n_years)

    enterprise_value = pv_forecast_total + pv_terminal_value
    tv_pct = (pv_terminal_value / enterprise_value * 100.0) if enterprise_value > 0 else None

    net_debt = total_debt - cash
    equity_value = enterprise_value + cash - total_debt + non_operating_adjustments
    intrinsic_per_share = equity_value / shares_outstanding

    suggested_buy = intrinsic_per_share * (1.0 - margin_of_safety_pct)

    upside_abs: Optional[float] = None
    upside_pct: Optional[float] = None
    if current_market_price is not None and current_market_price > 0 and not np.isnan(current_market_price):
        upside_abs = intrinsic_per_share - current_market_price
        upside_pct = (upside_abs / current_market_price) * 100.0

    label = classify_valuation_label(intrinsic_per_share, current_market_price)

    return DCFValuationResult(
        methodology="FCFF",
        discount_rate=wacc,
        terminal_growth_rate=terminal_growth_rate,
        pv_forecast_cashflows=pv_forecast_total,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        tv_pct_of_total_value=tv_pct,
        enterprise_value=enterprise_value,
        cash=cash,
        total_debt=total_debt,
        net_debt=net_debt,
        non_operating_adjustments=non_operating_adjustments,
        equity_value=equity_value,
        shares_outstanding=shares_outstanding,
        intrinsic_value_per_share=intrinsic_per_share,
        current_market_price=current_market_price,
        upside_absolute=upside_abs,
        upside_percent=upside_pct,
        margin_of_safety_pct=margin_of_safety_pct,
        suggested_buy_price=suggested_buy,
        valuation_label=label,
        forecast_pv_table=df,
    )


def run_fcfe_dcf(
    forecast_df: pd.DataFrame,
    cost_of_equity: float,
    terminal_growth_rate: float,
    shares_outstanding: float,
    current_market_price: Optional[float] = None,
    margin_of_safety_pct: float = 0.20,
) -> DCFValuationResult:
    """Execute direct Equity Value FCFE DCF for banks and financial institutions.

    Does not force an Enterprise Value calculation or subtract operating funding debt.
    """
    if forecast_df.empty or "FCFE" not in forecast_df.columns:
        raise ValueError("Forecast DataFrame must contain 'FCFE' values.")
    if shares_outstanding is None or shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be strictly positive (> 0).")
    if cost_of_equity <= 0:
        raise ValueError("Cost of Equity must be strictly positive (> 0).")
    if cost_of_equity <= terminal_growth_rate:
        raise ValueError("Cost of Equity must be strictly greater than Terminal Growth Rate.")
    if not (0.0 <= margin_of_safety_pct < 1.0):
        raise ValueError("Margin of safety must be between 0% and <100%.")

    df = forecast_df.copy()
    periods = df["Period"].astype(float).values if "Period" in df.columns else np.arange(1, len(df) + 1, dtype=float)
    fcfe_values = df["FCFE"].astype(float).values

    discount_factors = (1.0 + cost_of_equity) ** periods
    pv_fcfe = fcfe_values / discount_factors

    df["Discount Factor"] = discount_factors
    df["PV of FCF"] = pv_fcfe

    pv_forecast_total = float(np.sum(pv_fcfe))
    final_fcfe = float(fcfe_values[-1])
    n_years = float(periods[-1])

    terminal_value = calculate_terminal_value_gordon(final_fcfe, cost_of_equity, terminal_growth_rate)
    pv_terminal_value = terminal_value / ((1.0 + cost_of_equity) ** n_years)

    equity_value = pv_forecast_total + pv_terminal_value
    tv_pct = (pv_terminal_value / equity_value * 100.0) if equity_value > 0 else None
    intrinsic_per_share = equity_value / shares_outstanding
    suggested_buy = intrinsic_per_share * (1.0 - margin_of_safety_pct)

    upside_abs: Optional[float] = None
    upside_pct: Optional[float] = None
    if current_market_price is not None and current_market_price > 0 and not np.isnan(current_market_price):
        upside_abs = intrinsic_per_share - current_market_price
        upside_pct = (upside_abs / current_market_price) * 100.0

    label = classify_valuation_label(intrinsic_per_share, current_market_price)

    return DCFValuationResult(
        methodology="FCFE",
        discount_rate=cost_of_equity,
        terminal_growth_rate=terminal_growth_rate,
        pv_forecast_cashflows=pv_forecast_total,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        tv_pct_of_total_value=tv_pct,
        enterprise_value=None,
        cash=None,
        total_debt=None,
        net_debt=None,
        non_operating_adjustments=0.0,
        equity_value=equity_value,
        shares_outstanding=shares_outstanding,
        intrinsic_value_per_share=intrinsic_per_share,
        current_market_price=current_market_price,
        upside_absolute=upside_abs,
        upside_percent=upside_pct,
        margin_of_safety_pct=margin_of_safety_pct,
        suggested_buy_price=suggested_buy,
        valuation_label=label,
        forecast_pv_table=df,
    )
