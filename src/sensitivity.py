"""2D Sensitivity Analysis across Discount Rate (WACC/Ke) and Terminal Growth Rate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.dcf_engine import run_fcfe_dcf, run_fcff_dcf


@dataclass
class SensitivityResult:
    """Matrix of intrinsic values per share and optional upside percentages."""

    intrinsic_value_matrix: pd.DataFrame  # Index = Discount Rates, Columns = Terminal Growth Rates
    upside_pct_matrix: Optional[pd.DataFrame]
    discount_rates: list[float]
    terminal_growth_rates: list[float]


def build_range(center: float, step: float, count_each_side: int = 2, min_val: float = -0.05) -> list[float]:
    """Generate a symmetric sorted list of rates around a center value."""
    vals = [round(center + i * step, 6) for i in range(-count_each_side, count_each_side + 1)]
    return [v for v in vals if v >= min_val]


def compute_sensitivity_matrix(
    forecast_df: pd.DataFrame,
    discount_rates: list[float],
    terminal_growth_rates: list[float],
    shares_outstanding: float,
    methodology: str = "FCFF",
    cash: float = 0.0,
    total_debt: float = 0.0,
    non_operating_adjustments: float = 0.0,
    current_market_price: Optional[float] = None,
) -> SensitivityResult:
    """Compute Intrinsic Value Per Share for every (discount_rate, terminal_growth) combination.

    Invalid combinations where discount_rate <= terminal_growth_rate produce NaN cleanly.
    """
    if shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be strictly positive for sensitivity analysis.")

    row_labels = [f"{r * 100:.2f}%" for r in discount_rates]
    col_labels = [f"{g * 100:.2f}%" for g in terminal_growth_rates]

    val_mat = pd.DataFrame(np.nan, index=row_labels, columns=col_labels, dtype=float)
    upside_mat = (
        pd.DataFrame(np.nan, index=row_labels, columns=col_labels, dtype=float)
        if current_market_price is not None and current_market_price > 0
        else None
    )

    for r, r_lbl in zip(discount_rates, row_labels):
        for g, g_lbl in zip(terminal_growth_rates, col_labels):
            if r <= 0 or r <= g:
                continue
            if methodology == "FCFE":
                res = run_fcfe_dcf(
                    forecast_df=forecast_df,
                    cost_of_equity=r,
                    terminal_growth_rate=g,
                    shares_outstanding=shares_outstanding,
                    current_market_price=current_market_price,
                )
            else:
                res = run_fcff_dcf(
                    forecast_df=forecast_df,
                    wacc=r,
                    terminal_growth_rate=g,
                    cash=cash,
                    total_debt=total_debt,
                    shares_outstanding=shares_outstanding,
                    current_market_price=current_market_price,
                    non_operating_adjustments=non_operating_adjustments,
                )
            val_mat.loc[r_lbl, g_lbl] = res.intrinsic_value_per_share
            if upside_mat is not None and res.upside_percent is not None:
                upside_mat.loc[r_lbl, g_lbl] = res.upside_percent

    return SensitivityResult(
        intrinsic_value_matrix=val_mat,
        upside_pct_matrix=upside_mat,
        discount_rates=discount_rates,
        terminal_growth_rates=terminal_growth_rates,
    )
