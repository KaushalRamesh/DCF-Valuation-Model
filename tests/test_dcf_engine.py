"""Unit tests for Free Cash Flow forecasting, Terminal Value, EV, Equity Value, and Intrinsic Value per Share."""

import numpy as np
import pandas as pd
import pytest

from src.dcf_engine import (
    calculate_terminal_value_gordon,
    classify_valuation_label,
    run_fcfe_dcf,
    run_fcff_dcf,
)
from src.forecasting import (
    FCFEForecastAssumptions,
    ForecastAssumptions,
    build_fcfe_forecast,
    build_fcff_forecast,
)


def test_fcff_forecast_manual_verification():
    """Verify 5-year UFCF formula: EBIT*(1-t) + D&A - CapEx - Change in NWC."""
    assumptions = ForecastAssumptions(
        scenario="Base",
        base_revenue=1000.0,
        revenue_growth_rates=[0.10, 0.10, 0.10, 0.10, 0.10],
        ebit_margins=[0.20, 0.20, 0.20, 0.20, 0.20],
        tax_rate=0.25,
        da_pct_of_revenue=0.05,
        capex_pct_of_revenue=0.08,
        nwc_pct_of_revenue=0.10,
        derivation_notes={},
    )
    df = build_fcff_forecast(assumptions)
    assert len(df) == 5

    # Year 1 manual calculation:
    # Rev1 = 1000 * 1.10 = 1100; dRev = 100
    # EBIT1 = 1100 * 0.20 = 220
    # Taxes1 = 220 * 0.25 = 55; NOPAT1 = 165
    # D&A1 = 1100 * 0.05 = 55
    # CapEx1 = 1100 * 0.08 = 88
    # dNWC1 = 100 * 0.10 = 10
    # UFCF1 = 165 + 55 - 88 - 10 = 122
    yr1 = df.iloc[0]
    assert yr1["Revenue"] == pytest.approx(1100.0)
    assert yr1["EBIT"] == pytest.approx(220.0)
    assert yr1["Taxes"] == pytest.approx(55.0)
    assert yr1["NOPAT"] == pytest.approx(165.0)
    assert yr1["D&A"] == pytest.approx(55.0)
    assert yr1["CapEx"] == pytest.approx(88.0)
    assert yr1["Change in NWC"] == pytest.approx(10.0)
    assert yr1["UFCF"] == pytest.approx(122.0)


def test_fcff_forecast_negative_growth_and_negative_fcf():
    """Verify negative revenue growth reduces NWC (cash inflow) and negative EBIT yields negative UFCF."""
    assumptions = ForecastAssumptions(
        scenario="Conservative",
        base_revenue=1000.0,
        revenue_growth_rates=[-0.10, -0.05, 0.0, 0.02, 0.03],
        ebit_margins=[-0.10, -0.05, 0.05, 0.08, 0.10],
        tax_rate=0.20,
        da_pct_of_revenue=0.02,
        capex_pct_of_revenue=0.05,
        nwc_pct_of_revenue=0.10,
        derivation_notes={},
    )
    df = build_fcff_forecast(assumptions)
    # Yr 1: Rev = 900, dRev = -100, EBIT = -90, NOPAT = -72, DA = 18, CapEx = 45, dNWC = -10
    # UFCF1 = -72 + 18 - 45 - (-10) = -89
    assert df.iloc[0]["Revenue"] == pytest.approx(900.0)
    assert df.iloc[0]["Change in NWC"] == pytest.approx(-10.0)
    assert df.iloc[0]["UFCF"] == pytest.approx(-89.0)


def test_terminal_value_gordon_growth():
    """Verify TV = Final Year FCF * (1 + g) / (WACC - g)."""
    tv = calculate_terminal_value_gordon(final_year_fcf=200.0, discount_rate=0.10, terminal_growth_rate=0.03)
    # 200 * 1.03 / 0.07 = 2942.857142857143
    assert tv == pytest.approx(200.0 * 1.03 / 0.07)


def test_terminal_value_invalid_wacc_le_growth():
    """Verify ValueError when discount rate <= terminal growth."""
    with pytest.raises(ValueError):
        calculate_terminal_value_gordon(100.0, discount_rate=0.04, terminal_growth_rate=0.04)
    with pytest.raises(ValueError):
        calculate_terminal_value_gordon(100.0, discount_rate=0.03, terminal_growth_rate=0.04)


def test_run_fcff_dcf_manual_verification():
    """Manually verify PV of FCF, PV of TV, EV, Equity Value, Intrinsic Price, and Margin of Safety."""
    forecast_df = pd.DataFrame(
        {
            "Year": ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
            "Period": [1, 2, 3, 4, 5],
            "UFCF": [100.0, 110.0, 120.0, 130.0, 140.0],
        }
    )
    wacc = 0.10
    g = 0.03
    cash = 150.0
    debt = 250.0
    shares = 100.0
    market_price = 18.0
    mos = 0.20

    res = run_fcff_dcf(
        forecast_df=forecast_df,
        wacc=wacc,
        terminal_growth_rate=g,
        cash=cash,
        total_debt=debt,
        shares_outstanding=shares,
        current_market_price=market_price,
        margin_of_safety_pct=mos,
    )

    expected_pv_fcf = sum(fcf / ((1.0 + wacc) ** t) for t, fcf in zip([1, 2, 3, 4, 5], [100.0, 110.0, 120.0, 130.0, 140.0]))
    expected_tv = 140.0 * 1.03 / (0.10 - 0.03)
    expected_pv_tv = expected_tv / ((1.0 + wacc) ** 5)
    expected_ev = expected_pv_fcf + expected_pv_tv
    expected_eq = expected_ev + cash - debt
    expected_intrinsic = expected_eq / shares
    expected_buy = expected_intrinsic * 0.80
    expected_upside_pct = (expected_intrinsic - market_price) / market_price * 100.0

    assert res.pv_forecast_cashflows == pytest.approx(expected_pv_fcf)
    assert res.terminal_value == pytest.approx(expected_tv)
    assert res.pv_terminal_value == pytest.approx(expected_pv_tv)
    assert res.enterprise_value == pytest.approx(expected_ev)
    assert res.net_debt == pytest.approx(100.0)
    assert res.equity_value == pytest.approx(expected_eq)
    assert res.intrinsic_value_per_share == pytest.approx(expected_intrinsic)
    assert res.suggested_buy_price == pytest.approx(expected_buy)
    assert res.upside_percent == pytest.approx(expected_upside_pct)


def test_run_fcff_dcf_invalid_shares():
    """Zero or negative shares outstanding must raise ValueError."""
    forecast_df = pd.DataFrame({"Period": [1, 2, 3, 4, 5], "UFCF": [100.0] * 5})
    with pytest.raises(ValueError):
        run_fcff_dcf(forecast_df, wacc=0.10, terminal_growth_rate=0.03, cash=10.0, total_debt=5.0, shares_outstanding=0.0)
    with pytest.raises(ValueError):
        run_fcff_dcf(forecast_df, wacc=0.10, terminal_growth_rate=0.03, cash=10.0, total_debt=5.0, shares_outstanding=-50.0)


def test_run_fcfe_dcf_for_financials():
    """Verify FCFE mode skips Enterprise Value and discounts directly at Cost of Equity."""
    fe_assump = FCFEForecastAssumptions(
        scenario="Base",
        base_revenue=1000.0,
        base_net_income=200.0,
        revenue_growth_rates=[0.10] * 5,
        net_margin_rates=[0.20] * 5,
        fcfe_conversion_ratio=0.75,
        derivation_notes={},
    )
    df = build_fcfe_forecast(fe_assump)
    # Year 1: Rev=1100, NI=220, FCFE=165
    assert df.iloc[0]["FCFE"] == pytest.approx(165.0)

    res = run_fcfe_dcf(
        forecast_df=df,
        cost_of_equity=0.12,
        terminal_growth_rate=0.04,
        shares_outstanding=100.0,
        current_market_price=20.0,
    )
    assert res.methodology == "FCFE"
    assert res.enterprise_value is None
    assert res.equity_value > 0
    assert res.intrinsic_value_per_share == pytest.approx(res.equity_value / 100.0)


def test_valuation_label_wording():
    """Ensure labels never claim 'undervalued' or 'overvalued' definitively."""
    assert classify_valuation_label(120.0, 100.0) == "Below estimated value"
    assert classify_valuation_label(105.0, 100.0) == "Near estimated value"
    assert classify_valuation_label(80.0, 100.0) == "Above estimated value"
    assert classify_valuation_label(100.0, None) == "Market price unavailable"
