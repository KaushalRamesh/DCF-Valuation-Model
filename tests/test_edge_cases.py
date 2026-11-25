"""Unit tests for edge cases: missing data, invalid tickers, currency mismatch, and industry detection."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config import format_large_currency, is_indian_market
from src.data_fetcher import TickerDataError, fetch_company_financials
from src.financial_analysis import analyze_historical_financials
from src.industry import detect_industry_profile
from src.validation import validate_company_data


def test_industry_detection_bank_vs_tech():
    bank_profile = detect_industry_profile("Financial Services", "Banks - Regional")
    assert bank_profile.is_financial_institution is True
    assert bank_profile.recommended_method == "FCFE"
    assert bank_profile.warning_message is not None

    tech_profile = detect_industry_profile("Technology", "Consumer Electronics")
    assert tech_profile.is_financial_institution is False
    assert tech_profile.recommended_method == "FCFF"
    assert tech_profile.warning_message is None


def test_missing_financial_data_not_replaced_with_zero():
    """Missing values must remain NaN and generate validation alerts, never silently become 0."""
    cols = ["FY2022", "FY2023", "FY2024"]
    inc = pd.DataFrame(
        {
            "FY2022": [100.0, np.nan],
            "FY2023": [110.0, np.nan],
            "FY2024": [120.0, np.nan],
        },
        index=["Revenue", "Operating Income / EBIT"],
    )
    cf = pd.DataFrame(columns=cols, index=["Depreciation & Amortization", "Capital Expenditure", "Free Cash Flow"], dtype=float)
    bs = pd.DataFrame(columns=cols, index=["Total Cash & Equivalents", "Total Debt"], dtype=float)

    analysis = analyze_historical_financials(inc, cf, bs)
    assert np.isnan(analysis.metrics_table.loc["EBIT Margin (%)", "FY2024"])
    assert np.isnan(analysis.metrics_table.loc["Free Cash Flow", "FY2024"])

    issues = validate_company_data(
        market_data={
            "current_price": 50.0,
            "shares_outstanding": 100.0,
            "market_cap": 5000.0,
            "beta": None,
            "currency": "USD",
            "financial_currency": "USD",
        },
        income_df=inc,
        cashflow_df=cf,
        balance_df=bs,
    )
    fields = {i.field for i in issues}
    assert "Operating Income / EBIT" in fields
    assert "beta" in fields
    assert "Total Cash & Equivalents" in fields


def test_currency_mismatch_validation_and_formatting():
    issues = validate_company_data(
        market_data={
            "current_price": 100.0,
            "shares_outstanding": 50.0,
            "market_cap": 5000.0,
            "beta": 1.1,
            "currency": "USD",
            "financial_currency": "TWD",
        },
        income_df=pd.DataFrame({"FY2024": [1000.0]}, index=["Revenue"]),
        cashflow_df=pd.DataFrame(),
        balance_df=pd.DataFrame(),
    )
    assert any(i.category == "Currency" and i.severity == "error" for i in issues)

    assert is_indian_market("RELIANCE.NS") is True
    assert is_indian_market("AAPL") is False
    assert format_large_currency(1e7, "INR", "auto") == "₹1.00 Cr"
    assert format_large_currency(1e6, "USD", "auto") == "$1.00 M"


def test_invalid_ticker_raises_clear_error():
    """Empty or non-existent ticker must raise TickerDataError cleanly without crashing."""
    with pytest.raises(TickerDataError):
        fetch_company_financials("   ")

    class _EmptyTicker:
        info = {}
        financials = pd.DataFrame()
        cashflow = pd.DataFrame()
        balance_sheet = pd.DataFrame()

        def history(self, period="5d"):
            return pd.DataFrame()

    with patch("src.data_fetcher.yf.Ticker", return_value=_EmptyTicker()):
        with pytest.raises(TickerDataError):
            fetch_company_financials("INVALIDTICKER999")
