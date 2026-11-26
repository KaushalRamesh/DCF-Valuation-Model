"""Unit tests for valuation bridge, intrinsic value per share, scenario ordering, and formula audit."""

import numpy as np
import pandas as pd
import pytest

from src.audit import build_formula_audit
from src.dcf_engine import run_fcfe_dcf, run_fcff_dcf
from src.financial_analysis import analyze_historical_financials
from src.forecasting import (
    FCFEForecastAssumptions,
    ForecastAssumptions,
    build_fcfe_forecast,
    build_fcff_forecast,
    derive_initial_fcfe_assumptions,
    derive_initial_fcff_assumptions,
)
from src.industry import detect_industry_profile, get_methodology_explanation
from src.validation import validate_company_data
from src.wacc import WACCInputs, calculate_cost_of_equity, calculate_wacc


def _make_adani_like_data():
    """Construct a realistic INR Crore-scale dataset matching Adani Power's FY23-FY26 financials."""
    cols = ["FY2023", "FY2024", "FY2025", "FY2026"]
    cr = 1e7
    income_df = pd.DataFrame(
        {
            "FY2023": [38023.03 * cr, 27188.52 * cr, 10834.51 * cr, 6608.29 * cr, 10726.64 * cr, -3051.94 * cr, 3040.16 * cr],
            "FY2024": [50051.07 * cr, 30292.73 * cr, 19758.34 * cr, 14747.46 * cr, 20828.79 * cr, -37.28 * cr, 3079.43 * cr],
            "FY2025": [56131.82 * cr, 32490.36 * cr, 23641.46 * cr, 17584.53 * cr, 12938.77 * cr, 3609.90 * cr, 3072.61 * cr],
            "FY2026": [54152.16 * cr, 31721.09 * cr, 22431.07 * cr, 15739.57 * cr, 12834.30 * cr, 2528.43 * cr, 2953.81 * cr],
        },
        index=[
            "Revenue",
            "Cost of Revenue",
            "Gross Profit",
            "Operating Income / EBIT",
            "Net Income",
            "Income Tax Expense",
            "Interest Expense",
        ],
    )
    cashflow_df = pd.DataFrame(
        {
            "FY2023": [3303.68 * cr, 3243.68 * cr, 8430.53 * cr, 5186.85 * cr, -1350.60 * cr],
            "FY2024": [3931.33 * cr, 2602.45 * cr, 14170.15 * cr, 11567.70 * cr, -4450.08 * cr],
            "FY2025": [4308.88 * cr, 11559.04 * cr, 21501.11 * cr, 9942.07 * cr, -440.10 * cr],
            "FY2026": [4564.53 * cr, 23350.09 * cr, 20513.65 * cr, -2836.44 * cr, -447.11 * cr],
        },
        index=[
            "Depreciation & Amortization",
            "Capital Expenditure",
            "Operating Cash Flow",
            "Free Cash Flow",
            "Change in Working Capital",
        ],
    )
    balance_df = pd.DataFrame(
        {
            "FY2023": [349.23 * cr, 42180.58 * cr, 19553.00 * cr, 17744.59 * cr, 29875.66 * cr],
            "FY2024": [1136.25 * cr, 34431.07 * cr, 25586.58 * cr, 15788.67 * cr, 43145.03 * cr],
            "FY2025": [305.91 * cr, 39356.29 * cr, 26329.23 * cr, 16440.99 * cr, 56347.09 * cr],
            "FY2026": [883.25 * cr, 54518.77 * cr, 28688.08 * cr, 20328.21 * cr, 64936.88 * cr],
        },
        index=[
            "Total Cash & Equivalents",
            "Total Debt",
            "Current Assets",
            "Current Liabilities",
            "Total Equity",
        ],
    )
    market_data = {
        "ticker": "ADANIPOWER.NS",
        "company_name": "Adani Power Limited",
        "sector": "Utilities",
        "industry": "Utilities - Independent Power Producers",
        "current_price": 206.0,
        "shares_outstanding": 19284694705.0,
        "market_cap": 3972647092224.0,
        "beta": 0.48,
        "currency": "INR",
        "financial_currency": "INR",
    }
    return income_df, cashflow_df, balance_df, market_data


def test_fcff_valuation_bridge_net_debt_net_cash_and_non_op():
    """Verify EV -> Equity Value bridge (+ Cash - Debt + Non-Op Adj) and per-share division."""
    forecast_df = pd.DataFrame(
        {"Year": [f"Year {i}" for i in range(1, 6)], "Period": [1, 2, 3, 4, 5], "UFCF": [100.0, 110.0, 120.0, 130.0, 140.0]}
    )
    # Net debt case with non-operating adjustment
    res_debt = run_fcff_dcf(
        forecast_df=forecast_df,
        wacc=0.10,
        terminal_growth_rate=0.03,
        cash=50.0,
        total_debt=250.0,
        shares_outstanding=100.0,
        non_operating_adjustments=25.0,
    )
    assert res_debt.net_debt == pytest.approx(200.0)
    assert res_debt.equity_value == pytest.approx(res_debt.enterprise_value + 50.0 - 250.0 + 25.0)
    assert res_debt.intrinsic_value_per_share == pytest.approx(res_debt.equity_value / 100.0)

    # Net cash case (Cash > Debt raises Equity Value above EV)
    res_cash = run_fcff_dcf(
        forecast_df=forecast_df,
        wacc=0.10,
        terminal_growth_rate=0.03,
        cash=400.0,
        total_debt=100.0,
        shares_outstanding=100.0,
    )
    assert res_cash.net_debt == pytest.approx(-300.0)
    assert res_cash.equity_value == pytest.approx(res_cash.enterprise_value + 300.0)
    assert res_cash.intrinsic_value_per_share > res_debt.intrinsic_value_per_share


def test_fcfe_direct_equity_does_not_subtract_debt():
    """Verify FCFE calculates Equity Value directly without an EV bridge or debt subtraction."""
    forecast_df = pd.DataFrame(
        {"Year": [f"Year {i}" for i in range(1, 6)], "Period": [1, 2, 3, 4, 5], "FCFE": [100.0, 110.0, 120.0, 130.0, 140.0]}
    )
    ke = 0.12
    g = 0.04
    shares = 50.0
    res = run_fcfe_dcf(
        forecast_df=forecast_df,
        cost_of_equity=ke,
        terminal_growth_rate=g,
        shares_outstanding=shares,
        current_market_price=30.0,
    )

    expected_pv_fcfe = sum(cf / ((1.0 + ke) ** t) for t, cf in zip([1, 2, 3, 4, 5], [100.0, 110.0, 120.0, 130.0, 140.0]))
    expected_tv = 140.0 * 1.04 / (ke - g)
    expected_pv_tv = expected_tv / ((1.0 + ke) ** 5)
    expected_eq = expected_pv_fcfe + expected_pv_tv

    assert res.enterprise_value is None
    assert res.cash is None
    assert res.total_debt is None
    assert res.net_debt is None
    assert res.pv_forecast_cashflows == pytest.approx(expected_pv_fcfe)
    assert res.terminal_value == pytest.approx(expected_tv)
    assert res.pv_terminal_value == pytest.approx(expected_pv_tv)
    assert res.equity_value == pytest.approx(expected_eq)
    assert res.intrinsic_value_per_share == pytest.approx(expected_eq / shares)


def test_adani_power_end_to_end_fcff_fcfe_optimistic_and_audit():
    """Verify Adani Power-scale valuation, scenario ordering, units, and full formula audit reconciliation."""
    inc, cf, bs, mkt = _make_adani_like_data()
    analysis = analyze_historical_financials(inc, cf, bs)
    profile = detect_industry_profile(mkt["sector"], mkt["industry"])
    assert profile.recommended_method == "FCFF"

    rf, beta, erp = 0.07, 0.48, 0.065
    kd, wacc_tax, tg = 0.0542, 0.25, 0.04
    debt = float(bs.loc["Total Debt", "FY2026"])
    cash = float(bs.loc["Total Cash & Equivalents", "FY2026"])
    shares = mkt["shares_outstanding"]
    price = mkt["current_price"]

    ke = calculate_cost_of_equity(rf, beta, erp)
    wacc_res = calculate_wacc(WACCInputs(rf, beta, erp, kd, wacc_tax, mkt["market_cap"], debt))

    # 1. Verify Optimistic > Base > Conservative in both FCFF and FCFE
    fcff_vals = {}
    fcfe_vals = {}
    for scen in ("Conservative", "Base", "Optimistic"):
        fa = derive_initial_fcff_assumptions(inc, analysis, scenario=scen, default_tax_rate=wacc_tax, terminal_growth_anchor=tg)
        fdf = build_fcff_forecast(fa)
        r_fcff = run_fcff_dcf(fdf, wacc_res.wacc, tg, cash, debt, shares, price)
        fcff_vals[scen] = (fa, r_fcff)

        fea = derive_initial_fcfe_assumptions(inc, analysis, scenario=scen, terminal_growth_anchor=tg)
        fedf = build_fcfe_forecast(fea)
        r_fcfe = run_fcfe_dcf(fedf, ke, tg, shares, price)
        fcfe_vals[scen] = (fea, r_fcfe)

    assert fcff_vals["Optimistic"][1].intrinsic_value_per_share > fcff_vals["Base"][1].intrinsic_value_per_share
    assert fcff_vals["Base"][1].intrinsic_value_per_share > fcff_vals["Conservative"][1].intrinsic_value_per_share

    assert fcfe_vals["Optimistic"][1].intrinsic_value_per_share > fcfe_vals["Base"][1].intrinsic_value_per_share
    assert fcfe_vals["Base"][1].intrinsic_value_per_share > fcfe_vals["Conservative"][1].intrinsic_value_per_share

    # 2. Check exact Adani Power Optimistic FCFE numbers & audit reconciliation
    fea_opt, res_fcfe_opt = fcfe_vals["Optimistic"]
    assert res_fcfe_opt.pv_forecast_cashflows / 1e7 == pytest.approx(67480.27, abs=0.05)
    assert res_fcfe_opt.terminal_value / 1e7 == pytest.approx(381975.10, abs=0.05)
    assert res_fcfe_opt.pv_terminal_value / 1e7 == pytest.approx(235887.02, abs=0.05)
    assert res_fcfe_opt.equity_value / 1e7 == pytest.approx(303367.29, abs=0.05)
    assert res_fcfe_opt.intrinsic_value_per_share == pytest.approx(157.31, abs=0.02)

    audit_fcfe = build_formula_audit(
        dcf_result=res_fcfe_opt,
        assumptions=fea_opt,
        market_data=mkt,
        industry_profile=profile,
        wacc_res=wacc_res,
        ke_only=ke,
        risk_free_rate=rf,
        beta=beta,
        equity_risk_premium=erp,
        pre_tax_cost_of_debt=kd,
        wacc_tax_rate=wacc_tax,
        unit_mode="auto",
    )
    assert audit_fcfe.reconciliation_passed is True
    assert audit_fcfe.methodology == "FCFE"
    assert audit_fcfe.methodology_aligned is False  # Utilities recommends FCFF

    # 3. Check Adani Power Optimistic FCFF bridge & audit reconciliation
    fa_opt, res_fcff_opt = fcff_vals["Optimistic"]
    assert res_fcff_opt.enterprise_value == pytest.approx(
        res_fcff_opt.pv_forecast_cashflows + res_fcff_opt.pv_terminal_value
    )
    assert res_fcff_opt.equity_value == pytest.approx(res_fcff_opt.enterprise_value + cash - debt)
    assert res_fcff_opt.intrinsic_value_per_share == pytest.approx(res_fcff_opt.equity_value / shares)

    audit_fcff = build_formula_audit(
        dcf_result=res_fcff_opt,
        assumptions=fa_opt,
        market_data=mkt,
        industry_profile=profile,
        wacc_res=wacc_res,
        ke_only=ke,
        risk_free_rate=rf,
        beta=beta,
        equity_risk_premium=erp,
        pre_tax_cost_of_debt=kd,
        wacc_tax_rate=wacc_tax,
        unit_mode="auto",
    )
    assert audit_fcff.reconciliation_passed is True
    assert audit_fcff.methodology == "FCFF"
    assert audit_fcff.methodology_aligned is True


def test_active_methodology_explanation_matches_selected_mode():
    """Active explanation must match FCFF/FCFE regardless of industry recommendation."""
    assert "Cost of Equity" in get_methodology_explanation("FCFE")
    assert "without forcing an Enterprise Value bridge" in get_methodology_explanation("FCFE")
    assert "WACC" in get_methodology_explanation("FCFF")
    assert "Enterprise Value" in get_methodology_explanation("FCFF")


def test_validation_detects_market_cap_share_unit_mismatch():
    """Market Cap vs Price * Shares mismatch >15% must trigger a unit-consistency warning."""
    inc = pd.DataFrame({"FY2026": [1000.0]}, index=["Revenue"])
    issues = validate_company_data(
        market_data={
            "current_price": 200.0,
            "shares_outstanding": 100.0,
            "market_cap": 500000.0,  # Implied is 20,000 -> huge unit mismatch
            "beta": 1.0,
            "currency": "INR",
            "financial_currency": "INR",
        },
        income_df=inc,
        cashflow_df=pd.DataFrame(),
        balance_df=pd.DataFrame(),
    )
    assert any(i.category == "Unit Consistency" and i.field == "market_cap" for i in issues)


def test_cash_short_term_investments_and_current_debt_nwc():
    """Cash + ST investments must be prioritized over pure cash, and Current Debt excluded from Operating NWC."""
    from unittest.mock import patch
    from src.data_fetcher import fetch_company_financials

    class _MockEternalTicker:
        info = {
            "currentPrice": 371.7,
            "sharesOutstanding": 9212774343.0,
            "marketCap": 3424388382720.0,
            "beta": 0.24,
            "currency": "INR",
            "financialCurrency": "INR",
            "sector": "Consumer Cyclical",
            "industry": "Internet Retail",
            "longName": "Eternal Limited",
        }
        financials = pd.DataFrame(
            {
                "2024-03-31": [12114e7, 9232e7, -383e7, 351e7],
                "2025-03-31": [20243e7, 14678e7, 10e7, 527e7],
                "2026-03-31": [54364e7, 24197e7, -63e7, 366e7],
            },
            index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
        )
        cashflow = pd.DataFrame(
            {
                "2024-03-31": [526e7, -215e7],
                "2025-03-31": [863e7, -936e7],
                "2026-03-31": [1597e7, -1751e7],
            },
            index=["Depreciation And Amortization", "Capital Expenditure"],
        )
        balance_sheet = pd.DataFrame(
            {
                "2024-03-31": [309e7, 3405e7, 749e7, 161e7, 5458e7, 2083e7],
                "2025-03-31": [666e7, 7638e7, 2045e7, 391e7, 11701e7, 3326e7],
                "2026-03-31": [993e7, 8586e7, 4592e7, 759e7, 15700e7, 5672e7],
            },
            index=[
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Total Debt",
                "Current Debt And Capital Lease Obligation",
                "Current Assets",
                "Current Liabilities",
            ],
        )

    with patch("src.data_fetcher.yf.Ticker", return_value=_MockEternalTicker()):
        data = fetch_company_financials("ETERNAL.NS")

    # Cash should include Short-Term Investments (8586 Cr), not just 993 Cr
    assert data.balance_sheet.loc["Total Cash & Equivalents", "FY2026"] == pytest.approx(8586e7)
    assert data.balance_sheet.loc["Current Debt", "FY2026"] == pytest.approx(759e7)

    analysis = analyze_historical_financials(data.income_statement, data.cashflow_statement, data.balance_sheet)
    # Operating NWC = (15700 - 8586) - (5672 - 759) = 2201 Cr -> 2201 / 54364 = 4.05% (not 16.62%)
    assert analysis.summary_stats["latest_nwc_pct_rev"] == pytest.approx(2201.0 / 54364.0, rel=1e-4)

    # Low beta (0.24) must trigger a warning
    issues = validate_company_data(data.market_data, data.income_statement, data.cashflow_statement, data.balance_sheet)
    assert any(i.field == "beta" and i.severity == "warning" for i in issues)

    # Turnaround margin recovery path must yield positive Year 5 EBIT, positive TV, and Optimistic > Base > Conservative
    results = {}
    for scen in ("Conservative", "Base", "Optimistic"):
        fa = derive_initial_fcff_assumptions(data.income_statement, analysis, scenario=scen, default_tax_rate=0.25, terminal_growth_anchor=0.04)
        fdf = build_fcff_forecast(fa)
        assert fa.ebit_margins[-1] > fa.ebit_margins[0]
        assert fa.ebit_margins[-1] > 0.0
        res = run_fcff_dcf(
            fdf,
            wacc=0.0853,
            terminal_growth_rate=0.04,
            cash=8586e7,
            total_debt=4592e7,
            shares_outstanding=9212774343.0,
            current_market_price=371.7,
        )
        assert res.terminal_value > 0
        assert res.intrinsic_value_per_share > 0
        results[scen] = res.intrinsic_value_per_share

    assert results["Optimistic"] > results["Base"] > results["Conservative"]


def test_reliance_style_capex_normalization_and_pretax_income_tax_rate():
    """Peak CapEx cycle must fade toward steady-state D&A reinvestment and tax rate must use Pretax Income."""
    cr = 1e7
    inc = pd.DataFrame(
        {
            "FY2023": [877835.0 * cr, 102015.0 * cr, 20376.0 * cr, 94046.0 * cr, 18958.0 * cr],
            "FY2024": [901064.0 * cr, 111666.0 * cr, 25707.0 * cr, 104340.0 * cr, 21772.0 * cr],
            "FY2025": [964693.0 * cr, 112462.0 * cr, 25230.0 * cr, 106017.0 * cr, 22121.0 * cr],
            "FY2026": [1057219.0 * cr, 121377.0 * cr, 27552.0 * cr, 123162.0 * cr, 24056.0 * cr],
        },
        index=["Revenue", "Operating Income / EBIT", "Income Tax Expense", "Pretax Income", "Interest Expense"],
    )
    cf = pd.DataFrame(
        {
            "FY2023": [40303.0 * cr, 140988.0 * cr],
            "FY2024": [50832.0 * cr, 152883.0 * cr],
            "FY2025": [53136.0 * cr, 139967.0 * cr],
            "FY2026": [57688.0 * cr, 122916.0 * cr],
        },
        index=["Depreciation & Amortization", "Capital Expenditure"],
    )
    bs = pd.DataFrame(
        {
            "FY2023": [161233.0 * cr, 334392.0 * cr, 134986.0 * cr, 425296.0 * cr, 395743.0 * cr],
            "FY2024": [211310.0 * cr, 346142.0 * cr, 106015.0 * cr, 470100.0 * cr, 397367.0 * cr],
            "FY2025": [238792.0 * cr, 369575.0 * cr, 115534.0 * cr, 499270.0 * cr, 453737.0 * cr],
            "FY2026": [277212.0 * cr, 398000.0 * cr, 111051.0 * cr, 594249.0 * cr, 541254.0 * cr],
        },
        index=["Total Cash & Equivalents", "Total Debt", "Current Debt", "Current Assets", "Current Liabilities"],
    )

    analysis = analyze_historical_financials(inc, cf, bs)
    # Using Pretax Income (123,162 Cr) rather than EBIT-Interest (97,321 Cr) keeps tax rate ~23.8%, not >28%
    assert 0.21 < analysis.summary_stats["median_tax_rate"] < 0.25

    fa = derive_initial_fcff_assumptions(inc, analysis, scenario="Base", default_tax_rate=0.25, terminal_growth_anchor=0.04)
    assert isinstance(fa.capex_pct_of_revenue, list)
    assert len(fa.capex_pct_of_revenue) == 5
    # CapEx fades downward from Yr 1 toward steady-state reinvestment relative to D&A
    assert fa.capex_pct_of_revenue[0] > fa.capex_pct_of_revenue[-1] > fa.da_pct_of_revenue

    fdf = build_fcff_forecast(fa)
    res = run_fcff_dcf(
        fdf,
        wacc=0.0777,
        terminal_growth_rate=0.04,
        cash=277212.0 * cr,
        total_debt=398000.0 * cr,
        shares_outstanding=13532472634.0,
        current_market_price=1373.90,
    )
    assert res.terminal_value > 0
    assert res.intrinsic_value_per_share > 0


