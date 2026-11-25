"""Unit tests for CAPM Cost of Equity and WACC calculations."""

import pytest

from src.wacc import WACCInputs, calculate_cost_of_equity, calculate_wacc, estimate_pre_tax_cost_of_debt


def test_capm_cost_of_equity():
    """Verify Ke = Rf + Beta * ERP."""
    ke = calculate_cost_of_equity(risk_free_rate=0.042, beta=1.2, equity_risk_premium=0.05)
    assert ke == pytest.approx(0.042 + 1.2 * 0.05)  # 10.2%


def test_wacc_manual_verification_market_cap_weights():
    """Verify WACC = (E/(D+E))*Ke + (D/(D+E))*Kd*(1-t) using Market Cap equity."""
    inputs = WACCInputs(
        risk_free_rate=0.07,
        beta=1.1,
        equity_risk_premium=0.06,
        pre_tax_cost_of_debt=0.09,
        tax_rate=0.25,
        market_cap_equity=8000.0,
        total_debt=2000.0,
    )
    res = calculate_wacc(inputs)

    expected_ke = 0.07 + 1.1 * 0.06  # 0.136
    expected_kd_after = 0.09 * (1.0 - 0.25)  # 0.0675
    expected_wacc = 0.80 * expected_ke + 0.20 * expected_kd_after  # 0.1088 + 0.0135 = 0.1223

    assert res.cost_of_equity == pytest.approx(expected_ke)
    assert res.after_tax_cost_of_debt == pytest.approx(expected_kd_after)
    assert res.equity_weight == pytest.approx(0.80)
    assert res.debt_weight == pytest.approx(0.20)
    assert res.wacc == pytest.approx(expected_wacc)


def test_wacc_zero_debt_equals_cost_of_equity():
    """With zero debt, WACC must equal Cost of Equity."""
    inputs = WACCInputs(
        risk_free_rate=0.04,
        beta=1.0,
        equity_risk_premium=0.05,
        pre_tax_cost_of_debt=0.06,
        tax_rate=0.21,
        market_cap_equity=5000.0,
        total_debt=0.0,
    )
    res = calculate_wacc(inputs)
    assert res.equity_weight == pytest.approx(1.0)
    assert res.debt_weight == pytest.approx(0.0)
    assert res.wacc == pytest.approx(0.09)


def test_wacc_invalid_capital_and_tax():
    """Verify invalid capital or tax bounds raise ValueError."""
    with pytest.raises(ValueError):
        calculate_wacc(WACCInputs(0.04, 1.0, 0.05, 0.06, 0.21, 0.0, 0.0))
    with pytest.raises(ValueError):
        calculate_wacc(WACCInputs(0.04, 1.0, 0.05, 0.06, 0.21, -100.0, 50.0))
    with pytest.raises(ValueError):
        calculate_wacc(WACCInputs(0.04, 1.0, 0.05, 0.06, 1.20, 1000.0, 200.0))


def test_estimate_cost_of_debt_fallback():
    """Cost of debt estimator should fall back gracefully when debt or interest is missing."""
    rate, _ = estimate_pre_tax_cost_of_debt(None, 1000.0, fallback_rate=0.085)
    assert rate == pytest.approx(0.085)
    rate2, _ = estimate_pre_tax_cost_of_debt(80.0, 1000.0, fallback_rate=0.085)
    assert rate2 == pytest.approx(0.08)
