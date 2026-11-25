"""Sidebar controls for unit scaling, valuation mode, WACC/CAPM, Terminal Value, and manual data overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import streamlit as st

from src.config import is_indian_market
from src.data_fetcher import CompanyFinancialData
from src.industry import IndustryProfile
from src.wacc import COUNTRY_PRESETS, estimate_pre_tax_cost_of_debt


@dataclass
class SidebarValuationConfig:
    """User-configured valuation, WACC, terminal value, and manual override inputs."""

    unit_mode: str
    valuation_method: str  # "FCFF" or "FCFE"
    country_preset_key: str
    risk_free_rate: float
    beta: float
    equity_risk_premium: float
    pre_tax_cost_of_debt: float
    wacc_tax_rate: float
    use_custom_discount_rate: bool
    custom_discount_rate: float
    terminal_growth_rate: float
    margin_of_safety_pct: float
    non_operating_adjustments: float
    market_overrides: dict[str, Optional[float]]
    statement_overrides: dict[str, Optional[float]]


def _latest_val(df, row_name: str) -> Optional[float]:
    if df is None or df.empty or row_name not in df.index:
        return None
    s = df.loc[row_name].dropna()
    if s.empty:
        return None
    val = float(s.iloc[-1])
    return None if np.isnan(val) else val


def render_sidebar(
    data: CompanyFinancialData,
    industry_profile: IndustryProfile,
) -> SidebarValuationConfig:
    """Render sidebar controls for valuation methodology, WACC, Terminal Value, and manual overrides."""
    ticker = data.ticker
    mkt = data.market_data
    indian = is_indian_market(ticker, mkt.get("currency"), mkt.get("exchange"))
    default_preset = "IN" if indian else "US"

    st.sidebar.markdown("## Model Controls")

    unit_options = {
        "auto": "Auto (₹ Cr for INR / $ M for USD)",
        "crore": "Crores (÷ 10⁷)",
        "millions": "Millions (÷ 10⁶)",
        "billions": "Billions (÷ 10⁹)",
        "raw": "Raw Units",
    }
    unit_mode = st.sidebar.selectbox(
        "Display Scale",
        options=list(unit_options.keys()),
        format_func=lambda k: unit_options[k],
        index=0,
        key=f"unit_mode_{ticker}",
    )

    # Industry methodology selector
    st.sidebar.markdown("### 1. Valuation Methodology")
    if industry_profile.is_financial_institution:
        method_choices = ["FCFE", "FCFF"]
        default_idx = 0
        st.sidebar.warning("Financial institution detected: FCFE recommended over FCFF.")
    else:
        method_choices = ["FCFF", "FCFE"]
        default_idx = 0

    valuation_method = st.sidebar.radio(
        "Cash Flow Basis",
        options=method_choices,
        index=default_idx,
        format_func=lambda m: (
            "FCFF (Enterprise DCF)"
            if m == "FCFF"
            else (
                "FCFE (Equity DCF — Recommended)"
                if industry_profile.is_financial_institution
                else "FCFE (Equity DCF — Direct)"
            )
        ),
        key=f"val_method_{ticker}",
    )
    if not industry_profile.is_financial_institution and valuation_method == "FCFE":
        st.sidebar.info(
            "Non-financial company detected: FCFF is recommended. FCFE mode projects distributable Net Income "
            "and skips the EV / Cash / Debt bridge."
        )

    # WACC / CAPM Controls
    st.sidebar.markdown("### 2. Discount Rate (CAPM & WACC)")
    preset_key = st.sidebar.selectbox(
        "Market Risk Preset",
        options=["US", "IN"],
        index=1 if default_preset == "IN" else 0,
        format_func=lambda k: COUNTRY_PRESETS[k].name,
        key=f"preset_{ticker}",
    )
    preset = COUNTRY_PRESETS[preset_key]

    rf_pct = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=round(preset.risk_free_rate * 100.0, 2),
        step=0.10,
        key=f"rf_{ticker}_{preset_key}",
    )

    raw_beta = mkt.get("beta")
    default_beta = float(raw_beta) if raw_beta is not None else 1.0
    beta_val = st.sidebar.number_input(
        "Beta" + (" (Default 1.00 - Unavailable)" if raw_beta is None else ""),
        min_value=-1.0,
        max_value=5.0,
        value=round(default_beta, 2),
        step=0.05,
        key=f"beta_{ticker}",
    )

    erp_pct = st.sidebar.number_input(
        "Equity Risk Premium (%)",
        min_value=0.0,
        max_value=15.0,
        value=round(preset.equity_risk_premium * 100.0, 2),
        step=0.10,
        key=f"erp_{ticker}_{preset_key}",
    )

    latest_interest = _latest_val(data.income_statement, "Interest Expense")
    latest_debt = _latest_val(data.balance_sheet, "Total Debt")
    est_kd, kd_source = estimate_pre_tax_cost_of_debt(
        latest_interest, latest_debt, preset.default_cost_of_debt
    )

    kd_pct = st.sidebar.number_input(
        "Pre-Tax Cost of Debt (%)",
        min_value=0.0,
        max_value=25.0,
        value=round(est_kd * 100.0, 2),
        step=0.10,
        help=f"Initial estimate: {kd_source}",
        key=f"kd_{ticker}_{preset_key}",
    )

    wacc_tax_pct = st.sidebar.number_input(
        "Marginal Tax Rate for Debt Shield (%)",
        min_value=0.0,
        max_value=50.0,
        value=round(preset.default_tax_rate * 100.0, 1),
        step=0.5,
        key=f"wacc_tax_{ticker}_{preset_key}",
    )

    use_custom_r = st.sidebar.checkbox(
        "Override Calculated Discount Rate",
        value=False,
        key=f"override_r_{ticker}",
    )
    custom_r_pct = 10.0
    if use_custom_r:
        custom_r_pct = st.sidebar.number_input(
            "Custom Discount Rate (%)",
            min_value=1.0,
            max_value=35.0,
            value=10.0,
            step=0.25,
            key=f"custom_r_{ticker}",
        )

    # Terminal Value & Margin of Safety
    st.sidebar.markdown("### 3. Terminal Value & Margin of Safety")
    tg_pct = st.sidebar.number_input(
        "Terminal Growth Rate (%)",
        min_value=-3.0,
        max_value=10.0,
        value=round(preset.default_terminal_growth * 100.0, 2),
        step=0.25,
        key=f"tg_{ticker}_{preset_key}",
    )

    mos_pct = st.sidebar.slider(
        "Desired Margin of Safety (%)",
        min_value=0,
        max_value=60,
        value=20,
        step=5,
        key=f"mos_{ticker}",
    )

    # Manual Overrides for Missing or Custom Financial Data
    st.sidebar.markdown("### 4. Manual Data Overrides")
    st.sidebar.caption(
        "Leave blank or unchanged to use reported values. Any value entered here is flagged as [Manual]."
    )

    market_overrides: dict[str, Optional[float]] = {}
    statement_overrides: dict[str, Optional[float]] = {}

    with st.sidebar.expander("Market & Balance Sheet Overrides", expanded=False):
        def _opt_num(label: str, key: str, current_val: Optional[float]) -> Optional[float]:
            placeholder = f"{current_val:,.4g}" if current_val is not None else "Unavailable"
            raw = st.text_input(label, value="", placeholder=placeholder, key=key)
            if raw.strip() == "":
                return None
            try:
                return float(raw.replace(",", ""))
            except ValueError:
                st.warning(f"Invalid number for {label}; ignored.")
                return None

        p_ov = _opt_num("Current Price (per share)", f"ov_price_{ticker}", mkt.get("current_price"))
        if p_ov is not None:
            market_overrides["current_price"] = p_ov

        s_ov = _opt_num("Shares Outstanding (exact count)", f"ov_shares_{ticker}", mkt.get("shares_outstanding"))
        if s_ov is not None:
            market_overrides["shares_outstanding"] = s_ov

        m_ov = _opt_num("Market Cap (raw currency)", f"ov_mcap_{ticker}", mkt.get("market_cap"))
        if m_ov is not None:
            market_overrides["market_cap"] = m_ov

        cash_ov = _opt_num(
            "Cash & Equivalents (raw currency)",
            f"ov_cash_{ticker}",
            _latest_val(data.balance_sheet, "Total Cash & Equivalents"),
        )
        if cash_ov is not None:
            statement_overrides["bs:Total Cash & Equivalents"] = cash_ov

        debt_ov = _opt_num(
            "Total Debt (raw currency)",
            f"ov_debt_{ticker}",
            _latest_val(data.balance_sheet, "Total Debt"),
        )
        if debt_ov is not None:
            statement_overrides["bs:Total Debt"] = debt_ov

        rev_ov = _opt_num(
            "Latest Revenue (raw currency)",
            f"ov_rev_{ticker}",
            _latest_val(data.income_statement, "Revenue"),
        )
        if rev_ov is not None:
            statement_overrides["inc:Revenue"] = rev_ov

        ebit_ov = _opt_num(
            "Latest EBIT (raw currency)",
            f"ov_ebit_{ticker}",
            _latest_val(data.income_statement, "Operating Income / EBIT"),
        )
        if ebit_ov is not None:
            statement_overrides["inc:Operating Income / EBIT"] = ebit_ov

        non_op_adj = _opt_num(
            "Non-Operating Assets / Claims (+/- raw)",
            f"ov_nonop_{ticker}",
            0.0,
        ) or 0.0

    return SidebarValuationConfig(
        unit_mode=unit_mode,
        valuation_method=valuation_method,
        country_preset_key=preset_key,
        risk_free_rate=rf_pct / 100.0,
        beta=beta_val,
        equity_risk_premium=erp_pct / 100.0,
        pre_tax_cost_of_debt=kd_pct / 100.0,
        wacc_tax_rate=wacc_tax_pct / 100.0,
        use_custom_discount_rate=use_custom_r,
        custom_discount_rate=custom_r_pct / 100.0,
        terminal_growth_rate=tg_pct / 100.0,
        margin_of_safety_pct=mos_pct / 100.0,
        non_operating_adjustments=non_op_adj,
        market_overrides=market_overrides,
        statement_overrides=statement_overrides,
    )
