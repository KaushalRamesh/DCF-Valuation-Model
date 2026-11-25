"""Unit tests for 2D WACC vs Terminal Growth Sensitivity Analysis."""

import numpy as np
import pandas as pd
import pytest

from src.dcf_engine import run_fcfe_dcf, run_fcff_dcf
from src.sensitivity import build_range, compute_sensitivity_matrix


def test_sensitivity_matrix_dimensions_and_values():
    """Verify sensitivity grid matches direct DCF calculation and marks r <= g as NaN."""
    forecast_df = pd.DataFrame(
        {
            "Year": [f"Year {i}" for i in range(1, 6)],
            "Period": [1, 2, 3, 4, 5],
            "UFCF": [100.0, 110.0, 121.0, 133.1, 146.41],
        }
    )
    wacc_list = [0.05, 0.09, 0.10, 0.11]
    tg_list = [0.02, 0.03, 0.05]

    sens = compute_sensitivity_matrix(
        forecast_df=forecast_df,
        discount_rates=wacc_list,
        terminal_growth_rates=tg_list,
        shares_outstanding=100.0,
        methodology="FCFF",
        cash=50.0,
        total_debt=100.0,
        current_market_price=20.0,
    )

    mat = sens.intrinsic_value_matrix
    assert mat.shape == (4, 3)

    # Cell (WACC=5%, g=5%) must be NaN since r <= g
    assert np.isnan(mat.loc["5.00%", "5.00%"])

    # Cell (WACC=10%, g=3%) must match direct run_fcff_dcf
    direct = run_fcff_dcf(
        forecast_df=forecast_df,
        wacc=0.10,
        terminal_growth_rate=0.03,
        cash=50.0,
        total_debt=100.0,
        shares_outstanding=100.0,
        current_market_price=20.0,
    )
    assert mat.loc["10.00%", "3.00%"] == pytest.approx(direct.intrinsic_value_per_share)

    # Higher terminal growth at fixed WACC increases intrinsic value
    assert mat.loc["10.00%", "5.00%"] > mat.loc["10.00%", "2.00%"]
    # Higher WACC at fixed terminal growth decreases intrinsic value
    assert mat.loc["9.00%", "3.00%"] > mat.loc["11.00%", "3.00%"]


def test_sensitivity_invalid_shares_raises():
    forecast_df = pd.DataFrame({"Period": [1, 2, 3, 4, 5], "UFCF": [100.0] * 5})
    with pytest.raises(ValueError):
        compute_sensitivity_matrix(forecast_df, [0.10], [0.03], shares_outstanding=0.0)


def test_build_range():
    r = build_range(center=0.10, step=0.01, count_each_side=2, min_val=0.01)
    assert r == [0.08, 0.09, 0.10, 0.11, 0.12]


def test_sensitivity_fcfe_mode_recalculates_without_debt_subtraction():
    """Verify FCFE sensitivity grid matches direct run_fcfe_dcf and ignores cash/debt."""
    forecast_df = pd.DataFrame(
        {
            "Year": [f"Year {i}" for i in range(1, 6)],
            "Period": [1, 2, 3, 4, 5],
            "FCFE": [80.0, 90.0, 100.0, 110.0, 120.0],
        }
    )
    ke_list = [0.09, 0.10, 0.11]
    tg_list = [0.03, 0.04]

    sens = compute_sensitivity_matrix(
        forecast_df=forecast_df,
        discount_rates=ke_list,
        terminal_growth_rates=tg_list,
        shares_outstanding=100.0,
        methodology="FCFE",
        cash=500.0,
        total_debt=2000.0,
        current_market_price=20.0,
    )
    direct = run_fcfe_dcf(
        forecast_df=forecast_df,
        cost_of_equity=0.10,
        terminal_growth_rate=0.04,
        shares_outstanding=100.0,
        current_market_price=20.0,
    )
    assert sens.intrinsic_value_matrix.loc["10.00%", "4.00%"] == pytest.approx(direct.intrinsic_value_per_share)


def test_sensitivity_invalid_methodology_raises():
    forecast_df = pd.DataFrame({"Period": [1, 2, 3, 4, 5], "UFCF": [100.0] * 5})
    with pytest.raises(ValueError):
        compute_sensitivity_matrix(forecast_df, [0.10], [0.03], shares_outstanding=10.0, methodology="INVALID")
