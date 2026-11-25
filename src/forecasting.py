"""5-year financial forecast generation for FCFF (standard) and FCFE (financial institutions)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.financial_analysis import HistoricalAnalysisResult

SCENARIO_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "Conservative": {
        "growth_delta": -0.025,
        "margin_delta": -0.015,
        "capex_delta": 0.005,
        "fcfe_payout_delta": -0.08,
    },
    "Base": {
        "growth_delta": 0.0,
        "margin_delta": 0.0,
        "capex_delta": 0.0,
        "fcfe_payout_delta": 0.0,
    },
    "Optimistic": {
        "growth_delta": 0.025,
        "margin_delta": 0.015,
        "capex_delta": -0.005,
        "fcfe_payout_delta": 0.08,
    },
}


@dataclass
class ForecastAssumptions:
    """Editable 5-year forecast assumptions for FCFF DCF."""

    scenario: str
    base_revenue: float
    revenue_growth_rates: list[float]  # Length 5 (decimals, e.g. 0.10 = 10%)
    ebit_margins: list[float]  # Length 5 (decimals, e.g. 0.22 = 22%)
    tax_rate: float
    da_pct_of_revenue: float
    capex_pct_of_revenue: float
    nwc_pct_of_revenue: float  # Change in NWC = (Rev_t - Rev_{t-1}) * nwc_pct_of_revenue
    derivation_notes: dict[str, str]


@dataclass
class FCFEForecastAssumptions:
    """Editable 5-year forecast assumptions for Financial Institutions (FCFE mode)."""

    scenario: str
    base_revenue: float
    base_net_income: float
    revenue_growth_rates: list[float]
    net_margin_rates: list[float]
    fcfe_conversion_ratio: float  # Fraction of Net Income distributable after regulatory equity retention
    derivation_notes: dict[str, str]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _latest_valid(df: pd.DataFrame, row_name: str) -> Optional[float]:
    if df.empty or row_name not in df.index:
        return None
    s = pd.to_numeric(df.loc[row_name], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def derive_initial_fcff_assumptions(
    income_df: pd.DataFrame,
    analysis: HistoricalAnalysisResult,
    scenario: str = "Base",
    default_tax_rate: float = 0.25,
    terminal_growth_anchor: float = 0.035,
) -> ForecastAssumptions:
    """Derive sensible, non-naive initial 5-year assumptions from historical financials.

    Instead of blindly repeating raw arithmetic averages (which distort forecasts when a single
    historical year has an outlier spike or drop), this function:
    - Blends historical Revenue CAGR and median YoY growth, bounded to sustainable limits.
    - Applies gradual mean-reversion fade across Years 1–5 toward the long-term anchor.
    - Uses median historical margins and reinvestment ratios (robust to single-year anomalies).
    """
    adj = SCENARIO_ADJUSTMENTS.get(scenario, SCENARIO_ADJUSTMENTS["Base"])
    stats = analysis.summary_stats
    notes: dict[str, str] = {}

    base_rev = _latest_valid(income_df, "Revenue") or 0.0

    cagr = stats.get("revenue_cagr")
    med_g = stats.get("median_revenue_growth")
    if cagr is not None and med_g is not None:
        raw_g = 0.5 * cagr + 0.5 * med_g
        notes["revenue_growth"] = f"Blended historical CAGR ({cagr:.1%}) and median YoY growth ({med_g:.1%}) with fade"
    elif med_g is not None:
        raw_g = med_g
        notes["revenue_growth"] = f"Based on median historical YoY growth ({med_g:.1%}) with fade"
    elif cagr is not None:
        raw_g = cagr
        notes["revenue_growth"] = f"Based on historical CAGR ({cagr:.1%}) with fade"
    else:
        raw_g = 0.06
        notes["revenue_growth"] = "Default 6.0% (historical growth unavailable)"

    y1_growth = _clamp(raw_g + adj["growth_delta"], -0.10, 0.30)
    # Fade smoothly from Year 1 growth toward an anchor halfway to terminal growth by Year 5
    y5_target = _clamp(0.5 * y1_growth + 0.5 * (terminal_growth_anchor + adj["growth_delta"]), -0.05, 0.20)
    growth_rates = [
        round(float(y1_growth + (y5_target - y1_growth) * (i / 4.0)), 4) for i in range(5)
    ]

    med_ebit_m = stats.get("median_ebit_margin")
    latest_ebit = _latest_valid(income_df, "Operating Income / EBIT")
    latest_ebit_m = (latest_ebit / base_rev) if (latest_ebit is not None and base_rev > 0) else None
    if med_ebit_m is not None and latest_ebit_m is not None:
        base_margin = 0.5 * med_ebit_m + 0.5 * latest_ebit_m
        notes["ebit_margin"] = f"Blended median ({med_ebit_m:.1%}) and latest ({latest_ebit_m:.1%}) EBIT margin"
    elif med_ebit_m is not None:
        base_margin = med_ebit_m
        notes["ebit_margin"] = f"Based on median historical EBIT margin ({med_ebit_m:.1%})"
    else:
        base_margin = 0.15
        notes["ebit_margin"] = "Default 15.0% (historical EBIT margin unavailable)"

    adj_margin = _clamp(base_margin + adj["margin_delta"], -0.20, 0.60)
    ebit_margins = [round(float(adj_margin), 4) for _ in range(5)]

    med_tax = stats.get("median_tax_rate")
    if med_tax is not None:
        tax_rate = _clamp(med_tax, 0.10, 0.38)
        notes["tax_rate"] = f"Median historical effective tax rate ({med_tax:.1%}), bounded [10%, 38%]"
    else:
        tax_rate = default_tax_rate
        notes["tax_rate"] = f"Market default ({default_tax_rate:.1%}); effective tax rate unavailable"

    med_da = stats.get("median_da_pct")
    if med_da is not None:
        da_pct = _clamp(med_da, 0.0, 0.25)
        notes["da_pct"] = f"Median historical D&A % of revenue ({med_da:.1%})"
    else:
        da_pct = 0.03
        notes["da_pct"] = "Default 3.0% (historical D&A unavailable)"

    med_capex = stats.get("median_capex_pct")
    if med_capex is not None:
        capex_pct = _clamp(med_capex + adj["capex_delta"], 0.0, 0.35)
        notes["capex_pct"] = f"Median historical CapEx % of revenue ({med_capex:.1%})"
    else:
        capex_pct = _clamp(0.04 + adj["capex_delta"], 0.0, 0.35)
        notes["capex_pct"] = "Default 4.0% (historical CapEx unavailable)"

    latest_nwc_pct = stats.get("latest_nwc_pct_rev")
    if latest_nwc_pct is not None:
        nwc_pct = _clamp(latest_nwc_pct, -0.10, 0.25)
        notes["nwc_pct"] = f"Latest operating NWC % of revenue ({latest_nwc_pct:.1%}), bounded [-10%, 25%]"
    else:
        nwc_pct = 0.05
        notes["nwc_pct"] = "Default 5.0% incremental NWC / ΔRevenue (balance sheet NWC unavailable)"

    return ForecastAssumptions(
        scenario=scenario,
        base_revenue=float(base_rev),
        revenue_growth_rates=growth_rates,
        ebit_margins=ebit_margins,
        tax_rate=round(float(tax_rate), 4),
        da_pct_of_revenue=round(float(da_pct), 4),
        capex_pct_of_revenue=round(float(capex_pct), 4),
        nwc_pct_of_revenue=round(float(nwc_pct), 4),
        derivation_notes=notes,
    )


def build_fcff_forecast(assumptions: ForecastAssumptions) -> pd.DataFrame:
    """Build a 5-year Unlevered Free Cash Flow (FCFF) forecast table.

    Uses the formula:
        UFCF = EBIT * (1 - Tax Rate) + D&A - CapEx - Change in NWC
    """
    if len(assumptions.revenue_growth_rates) != 5 or len(assumptions.ebit_margins) != 5:
        raise ValueError("Forecast requires exactly 5 years of growth rates and EBIT margins.")

    rows = []
    prev_rev = assumptions.base_revenue

    for idx in range(5):
        year_num = idx + 1
        g = assumptions.revenue_growth_rates[idx]
        margin = assumptions.ebit_margins[idx]

        revenue = prev_rev * (1.0 + g)
        rev_change = revenue - prev_rev

        ebit = revenue * margin
        taxes = ebit * assumptions.tax_rate
        nopat = ebit * (1.0 - assumptions.tax_rate)
        da = revenue * assumptions.da_pct_of_revenue
        capex = revenue * assumptions.capex_pct_of_revenue
        change_in_nwc = rev_change * assumptions.nwc_pct_of_revenue

        ufcf = nopat + da - capex - change_in_nwc

        rows.append(
            {
                "Year": f"Year {year_num}",
                "Period": year_num,
                "Revenue": revenue,
                "Revenue Growth (%)": g * 100.0,
                "EBIT": ebit,
                "EBIT Margin (%)": margin * 100.0,
                "Taxes": taxes,
                "NOPAT": nopat,
                "D&A": da,
                "CapEx": capex,
                "Change in NWC": change_in_nwc,
                "UFCF": ufcf,
            }
        )
        prev_rev = revenue

    return pd.DataFrame(rows)


def derive_initial_fcfe_assumptions(
    income_df: pd.DataFrame,
    analysis: HistoricalAnalysisResult,
    scenario: str = "Base",
    terminal_growth_anchor: float = 0.04,
) -> FCFEForecastAssumptions:
    """Derive 5-year FCFE assumptions for banks and financial institutions."""
    adj = SCENARIO_ADJUSTMENTS.get(scenario, SCENARIO_ADJUSTMENTS["Base"])
    stats = analysis.summary_stats
    notes: dict[str, str] = {}

    base_rev = _latest_valid(income_df, "Revenue") or 0.0
    base_ni = _latest_valid(income_df, "Net Income") or 0.0

    cagr = stats.get("revenue_cagr")
    med_g = stats.get("median_revenue_growth")
    raw_g = (
        0.5 * cagr + 0.5 * med_g
        if (cagr is not None and med_g is not None)
        else (med_g if med_g is not None else (cagr if cagr is not None else 0.08))
    )
    y1_g = _clamp(raw_g + adj["growth_delta"], -0.05, 0.25)
    y5_g = _clamp(0.5 * y1_g + 0.5 * (terminal_growth_anchor + adj["growth_delta"]), 0.0, 0.18)
    growth_rates = [round(float(y1_g + (y5_g - y1_g) * (i / 4.0)), 4) for i in range(5)]
    notes["growth"] = "Blended historical top-line growth with 5-year fade"

    med_nm = stats.get("median_net_margin")
    if med_nm is None and base_rev > 0 and base_ni != 0:
        med_nm = base_ni / base_rev
    base_nm = med_nm if med_nm is not None else 0.18
    adj_nm = _clamp(base_nm + adj["margin_delta"], -0.10, 0.50)
    net_margins = [round(float(adj_nm), 4) for _ in range(5)]
    notes["net_margin"] = f"Based on historical net profit margin ({base_nm:.1%})"

    fcfe_ratio = _clamp(0.75 + adj["fcfe_payout_delta"], 0.20, 1.00)
    notes["fcfe_ratio"] = "Distributable earnings ratio after regulatory capital retention (editable)"

    return FCFEForecastAssumptions(
        scenario=scenario,
        base_revenue=float(base_rev),
        base_net_income=float(base_ni),
        revenue_growth_rates=growth_rates,
        net_margin_rates=net_margins,
        fcfe_conversion_ratio=round(float(fcfe_ratio), 4),
        derivation_notes=notes,
    )


def build_fcfe_forecast(assumptions: FCFEForecastAssumptions) -> pd.DataFrame:
    """Build a 5-year Free Cash Flow to Equity (FCFE) forecast for financial institutions."""
    if len(assumptions.revenue_growth_rates) != 5 or len(assumptions.net_margin_rates) != 5:
        raise ValueError("FCFE forecast requires exactly 5 years of growth rates and net margins.")

    rows = []
    prev_rev = assumptions.base_revenue

    for idx in range(5):
        year_num = idx + 1
        g = assumptions.revenue_growth_rates[idx]
        nm = assumptions.net_margin_rates[idx]

        revenue = prev_rev * (1.0 + g)
        net_income = revenue * nm
        retained_capital = net_income * (1.0 - assumptions.fcfe_conversion_ratio)
        fcfe = net_income * assumptions.fcfe_conversion_ratio

        rows.append(
            {
                "Year": f"Year {year_num}",
                "Period": year_num,
                "Revenue": revenue,
                "Revenue Growth (%)": g * 100.0,
                "Net Income": net_income,
                "Net Margin (%)": nm * 100.0,
                "Regulatory Capital Retention": retained_capital,
                "FCFE": fcfe,
            }
        )
        prev_rev = revenue

    return pd.DataFrame(rows)
