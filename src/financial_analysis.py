"""Historical financial analysis: growth rates, profitability margins, reinvestment ratios, and FCF trend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class HistoricalAnalysisResult:
    """Computed historical metrics, ratios, and summary statistics."""

    metrics_table: pd.DataFrame  # Rows = calculated metrics, Columns = fiscal years
    summary_stats: dict[str, Optional[float]]
    fcf_trend_direction: str
    fcf_cagr: Optional[float]


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division returning NaN where denominator is zero or missing."""
    den_clean = den.replace(0.0, np.nan)
    return num / den_clean


def _safe_growth(series: pd.Series) -> pd.Series:
    """Compute YoY growth safely, returning NaN when prior period is <= 0 or missing."""
    s = pd.to_numeric(series, errors="coerce")
    prev = s.shift(1)
    growth = pd.Series(np.nan, index=s.index, dtype=float)
    valid = s.notna() & prev.notna() & (prev > 0)
    growth.loc[valid] = (s.loc[valid] - prev.loc[valid]) / prev.loc[valid]
    return growth


def compute_cagr(series: pd.Series) -> Optional[float]:
    """Compute Compound Annual Growth Rate between first and last positive values."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    start_val = float(s.iloc[0])
    end_val = float(s.iloc[-1])
    periods = len(s) - 1
    if start_val <= 0 or end_val <= 0 or periods <= 0:
        return None
    return (end_val / start_val) ** (1.0 / periods) - 1.0


def analyze_historical_financials(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame,
    balance_df: pd.DataFrame,
) -> HistoricalAnalysisResult:
    """Calculate historical growth rates, margins, CapEx/D&A ratios, NWC, and FCF trends."""
    if income_df.empty:
        return HistoricalAnalysisResult(
            metrics_table=pd.DataFrame(),
            summary_stats={},
            fcf_trend_direction="Unavailable",
            fcf_cagr=None,
        )

    cols = list(income_df.columns)

    def _row(df: pd.DataFrame, name: str) -> pd.Series:
        if df.empty or name not in df.index:
            return pd.Series(np.nan, index=cols, dtype=float)
        return pd.to_numeric(df.loc[name].reindex(cols), errors="coerce")

    rev = _row(income_df, "Revenue")
    gp = _row(income_df, "Gross Profit")
    ebit = _row(income_df, "Operating Income / EBIT")
    ni = _row(income_df, "Net Income")
    tax = _row(income_df, "Income Tax Expense")
    interest = _row(income_df, "Interest Expense")

    da = _row(cashflow_df, "Depreciation & Amortization")
    capex = _row(cashflow_df, "Capital Expenditure")
    ocf = _row(cashflow_df, "Operating Cash Flow")
    fcf = _row(cashflow_df, "Free Cash Flow")
    # If FCF row was completely unavailable, compute only where OCF and CapEx both exist
    fcf_effective = fcf.copy()
    missing_fcf = fcf_effective.isna() & ocf.notna() & capex.notna()
    fcf_effective.loc[missing_fcf] = ocf.loc[missing_fcf] - capex.loc[missing_fcf]

    ca = _row(balance_df, "Current Assets")
    cl = _row(balance_df, "Current Liabilities")
    cash = _row(balance_df, "Total Cash & Equivalents")

    # Operating Net Working Capital approximation: (Current Assets - Cash) - Current Liabilities
    # (Falls back to Current Assets - Current Liabilities only if Cash is missing)
    non_cash_ca = np.where(cash.notna(), ca - cash, ca)
    nwc = pd.Series(non_cash_ca, index=cols, dtype=float) - cl
    nwc = nwc.where(ca.notna() & cl.notna(), np.nan)
    change_in_nwc = nwc.diff()

    # Effective tax rate estimate: Tax / (EBIT - Interest) when pretax proxy > 0
    pretax_proxy = ebit - interest.fillna(0.0)
    eff_tax = _safe_divide(tax, pretax_proxy.where(pretax_proxy > 0, np.nan))
    eff_tax = eff_tax.clip(lower=0.0, upper=0.50)

    metrics = pd.DataFrame(index=[
        "Revenue Growth (%)",
        "EBIT Growth (%)",
        "Net Income Growth (%)",
        "Gross Margin (%)",
        "EBIT Margin (%)",
        "Net Profit Margin (%)",
        "FCF Margin (%)",
        "CapEx % of Revenue",
        "D&A % of Revenue",
        "Effective Tax Rate (%)",
        "Net Working Capital",
        "Change in NWC",
        "Free Cash Flow",
    ], columns=cols, dtype=float)

    metrics.loc["Revenue Growth (%)"] = _safe_growth(rev) * 100.0
    metrics.loc["EBIT Growth (%)"] = _safe_growth(ebit) * 100.0
    metrics.loc["Net Income Growth (%)"] = _safe_growth(ni) * 100.0
    metrics.loc["Gross Margin (%)"] = _safe_divide(gp, rev) * 100.0
    metrics.loc["EBIT Margin (%)"] = _safe_divide(ebit, rev) * 100.0
    metrics.loc["Net Profit Margin (%)"] = _safe_divide(ni, rev) * 100.0
    metrics.loc["FCF Margin (%)"] = _safe_divide(fcf_effective, rev) * 100.0
    metrics.loc["CapEx % of Revenue"] = _safe_divide(capex, rev) * 100.0
    metrics.loc["D&A % of Revenue"] = _safe_divide(da, rev) * 100.0
    metrics.loc["Effective Tax Rate (%)"] = eff_tax * 100.0
    metrics.loc["Net Working Capital"] = nwc
    metrics.loc["Change in NWC"] = change_in_nwc
    metrics.loc["Free Cash Flow"] = fcf_effective

    def _median_or_none(row_name: str, scale: float = 1.0) -> Optional[float]:
        s = metrics.loc[row_name].dropna()
        if s.empty:
            return None
        return float(s.median()) / scale

    fcf_valid = fcf_effective.dropna()
    fcf_cagr = compute_cagr(fcf_effective)
    if len(fcf_valid) < 2:
        trend = "Insufficient Data"
    else:
        x = np.arange(len(fcf_valid), dtype=float)
        slope = float(np.polyfit(x, fcf_valid.values, 1)[0])
        mean_abs = float(np.mean(np.abs(fcf_valid.values))) or 1.0
        rel_slope = slope / mean_abs
        if rel_slope > 0.05:
            trend = "Upward"
        elif rel_slope < -0.05:
            trend = "Downward"
        else:
            trend = "Stable / Volatile"

    summary = {
        "revenue_cagr": compute_cagr(rev),
        "median_revenue_growth": _median_or_none("Revenue Growth (%)", 100.0),
        "median_ebit_margin": _median_or_none("EBIT Margin (%)", 100.0),
        "median_net_margin": _median_or_none("Net Profit Margin (%)", 100.0),
        "median_fcf_margin": _median_or_none("FCF Margin (%)", 100.0),
        "median_capex_pct": _median_or_none("CapEx % of Revenue", 100.0),
        "median_da_pct": _median_or_none("D&A % of Revenue", 100.0),
        "median_tax_rate": _median_or_none("Effective Tax Rate (%)", 100.0),
        "latest_nwc_pct_rev": (
            float((nwc.iloc[-1] / rev.iloc[-1]))
            if pd.notna(nwc.iloc[-1]) and pd.notna(rev.iloc[-1]) and rev.iloc[-1] > 0
            else None
        ),
    }

    return HistoricalAnalysisResult(
        metrics_table=metrics,
        summary_stats=summary,
        fcf_trend_direction=trend,
        fcf_cagr=fcf_cagr,
    )
