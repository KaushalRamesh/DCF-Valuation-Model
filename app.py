"""DCF Analyzer — Universal Discounted Cash Flow Valuation Dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import (
    chart_fcf_forecast,
    chart_historical_fcf,
    chart_historical_revenue,
    chart_intrinsic_vs_market,
    chart_revenue_forecast,
    chart_sensitivity_heatmap,
    chart_value_breakdown,
)
from components.metrics import (
    render_company_header,
    render_formula_audit,
    render_validation_issues,
    render_valuation_summary_cards,
    render_wacc_breakdown,
)
from components.sidebar import render_sidebar
from components.tables import (
    format_analysis_table_for_display,
    format_forecast_table_for_display,
    format_statement_for_display,
    render_sensitivity_table,
)
from src.audit import build_formula_audit
from src.config import SETTINGS, format_large_currency, format_per_share, get_display_scale
from src.data_fetcher import CompanyFinancialData, TickerDataError, apply_manual_overrides, fetch_company_financials
from src.dcf_engine import DCFValuationResult, run_fcfe_dcf, run_fcff_dcf
from src.financial_analysis import analyze_historical_financials
from src.forecasting import (
    FCFEForecastAssumptions,
    ForecastAssumptions,
    build_fcfe_forecast,
    build_fcff_forecast,
    derive_initial_fcfe_assumptions,
    derive_initial_fcff_assumptions,
)
from src.industry import detect_industry_profile
from src.sensitivity import build_range, compute_sensitivity_matrix
from src.validation import validate_company_data, validate_dcf_inputs
from src.wacc import WACCInputs, calculate_cost_of_equity, calculate_wacc

st.set_page_config(
    page_title="DCF Analyzer | Universal Stock Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark financial analyst theme styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0B0E14;
            color: #E6EDF3;
        }
        .dcf-header-card {
            background: #131722;
            border: 1px solid #1F2937;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 14px;
        }
        .dcf-company-title {
            font-size: 1.55rem;
            font-weight: 700;
            color: #F9FAFB;
            line-height: 1.25;
        }
        .dcf-ticker-badge {
            display: inline-block;
            background: #1F2937;
            color: #10B981;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 2px 10px;
            border-radius: 6px;
            margin-left: 8px;
            vertical-align: middle;
        }
        .dcf-meta-row {
            margin-top: 6px;
            color: #9CA3AF;
            font-size: 0.9rem;
        }
        .dcf-price-label {
            font-size: 0.8rem;
            color: #9CA3AF;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .dcf-price-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #F9FAFB;
        }
        .dcf-timestamp {
            font-size: 0.78rem;
            color: #6B7280;
        }
        .dcf-status-banner {
            background: #131722;
            border: 1px solid #1F2937;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 10px 0 14px 0;
        }
        div[data-testid="stMetric"] {
            background-color: #131722;
            border: 1px solid #1F2937;
            padding: 12px 14px;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=SETTINGS.cache_ttl_seconds, show_spinner=False)
def _cached_fetch(ticker: str) -> CompanyFinancialData:
    return fetch_company_financials(ticker)


def _latest_number(df: pd.DataFrame, row_name: str) -> float | None:
    if df.empty or row_name not in df.index:
        return None
    s = pd.to_numeric(df.loc[row_name], errors="coerce").dropna()
    if s.empty:
        return None
    val = float(s.iloc[-1])
    return None if np.isnan(val) else val


def _select_example_ticker(symbol: str) -> None:
    st.session_state["active_ticker"] = symbol
    st.session_state["ticker_input_box"] = symbol


def _submit_ticker() -> None:
    cleaned = (st.session_state.get("ticker_input_box") or "").strip().upper()
    if cleaned:
        st.session_state["active_ticker"] = cleaned
        st.session_state["ticker_input_box"] = cleaned


def _render_search_bar() -> str:
    st.markdown("### DCF Analyzer")
    st.caption(
        "Fundamental Discounted Cash Flow valuation tool for NSE India (.NS) and US equities. "
        "Estimates depend strictly on user assumptions and reported financial statements."
    )

    if "active_ticker" not in st.session_state:
        st.session_state["active_ticker"] = "AAPL"
    if "ticker_input_box" not in st.session_state:
        st.session_state["ticker_input_box"] = st.session_state["active_ticker"]

    col_in, col_btn = st.columns([4, 1])
    with col_in:
        entered = st.text_input(
            "Stock Ticker",
            key="ticker_input_box",
            placeholder="Enter ticker (e.g., RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, AAPL, MSFT, GOOGL)",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_clicked = st.button("Analyze", type="primary", width="stretch", on_click=_submit_ticker)

    examples = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "MSFT", "GOOGL"]
    ex_cols = st.columns(len(examples))
    for idx, sym in enumerate(examples):
        ex_cols[idx].button(
            sym,
            width="stretch",
            key=f"ex_{sym}",
            on_click=_select_example_ticker,
            args=(sym,),
        )

    if analyze_clicked and not (entered or "").strip():
        st.warning("Please enter a valid stock ticker before clicking Analyze.")

    return st.session_state["active_ticker"]


def main() -> None:
    active_ticker = _render_search_bar()

    with st.spinner(f"Retrieving financial statements and market data for {active_ticker}..."):
        try:
            raw_data = _cached_fetch(active_ticker)
        except TickerDataError as exc:
            st.error(f"**Data Retrieval Error:** {exc}")
            st.info(
                "Tip: For Indian NSE stocks append `.NS` (e.g., `RELIANCE.NS`, `TCS.NS`). "
                "For US stocks enter the standard symbol (e.g., `AAPL`, `MSFT`)."
            )
            return
        except Exception as exc:
            st.error(f"**Unexpected Error:** Unable to process '{active_ticker}': {exc}")
            return

    industry_profile = detect_industry_profile(
        raw_data.market_data.get("sector"),
        raw_data.market_data.get("industry"),
    )

    cfg = render_sidebar(raw_data, industry_profile)
    data = apply_manual_overrides(raw_data, cfg.market_overrides, cfg.statement_overrides)

    mkt = data.market_data
    currency = mkt.get("currency", "USD")

    # Header
    render_company_header(mkt, data.provenance, cfg.unit_mode)

    if industry_profile.warning_message:
        st.warning(f"**Industry Guidance:** {industry_profile.warning_message}")

    # Validation Layer
    data_issues = validate_company_data(
        mkt,
        data.income_statement,
        data.cashflow_statement,
        data.balance_sheet,
        is_financial_institution=industry_profile.is_financial_institution,
    )
    render_validation_issues(data_issues)

    # Historical Analysis
    analysis = analyze_historical_financials(
        data.income_statement,
        data.cashflow_statement,
        data.balance_sheet,
    )

    # Scenario & Forecast Assumptions State
    st.markdown("---")
    scen_col, info_col = st.columns([1.4, 2.6])
    with scen_col:
        scenario = st.radio(
            "Forecast Scenario",
            options=["Conservative", "Base", "Optimistic"],
            index=1,
            horizontal=True,
            key=f"scenario_{active_ticker}_{cfg.valuation_method}",
        )
    with info_col:
        method_note = (
            f" *(Note: {industry_profile.recommended_method} recommended for {industry_profile.sector})*"
            if cfg.valuation_method != industry_profile.recommended_method
            else ""
        )
        st.info(
            f"**Active Methodology:** {cfg.valuation_method}{method_note} | **Scenario:** {scenario} — "
            "Edit any annual growth rate, margin, or reinvestment assumption below."
        )

    # Derive initial defaults based on historical financials (non-naive)
    if cfg.valuation_method == "FCFF":
        default_assump = derive_initial_fcff_assumptions(
            data.income_statement,
            analysis,
            scenario=scenario,
            default_tax_rate=cfg.wacc_tax_rate,
            terminal_growth_anchor=cfg.terminal_growth_rate,
        )
    else:
        default_fe_assump = derive_initial_fcfe_assumptions(
            data.income_statement,
            analysis,
            scenario=scenario,
            terminal_growth_anchor=cfg.terminal_growth_rate,
        )

    # Tabs for clean organization
    tab_overview, tab_assumptions, tab_history, tab_sensitivity, tab_audit = st.tabs(
        [
            "Valuation Overview",
            "Forecast Assumptions",
            "Historical Financials",
            "Sensitivity Analysis",
            "Formula Audit & Disclaimer",
        ]
    )

    # Editable Forecast Assumptions inside Tab 2 (values persist via session_state keys)
    with tab_assumptions:
        st.subheader("Editable 5-Year Forecast Assumptions")
        if cfg.valuation_method == "FCFF":
            st.caption(
                f"Base Revenue: {format_large_currency(default_assump.base_revenue, currency, cfg.unit_mode)} | "
                "Initial assumptions blend historical trends with mean-reversion fade."
            )
            with st.expander("How Initial Assumptions Were Derived", expanded=False):
                for k, note in default_assump.derivation_notes.items():
                    st.write(f"- **{k}:** {note}")

            key_prefix = f"fcff_{active_ticker}_{scenario}"
            g_cols = st.columns(5)
            growth_rates: list[float] = []
            for i in range(5):
                val = g_cols[i].number_input(
                    f"Yr {i + 1} Rev Growth (%)",
                    min_value=-50.0,
                    max_value=100.0,
                    value=round(default_assump.revenue_growth_rates[i] * 100.0, 2),
                    step=0.5,
                    key=f"{key_prefix}_g_{i}",
                )
                growth_rates.append(val / 100.0)

            m_cols = st.columns(5)
            ebit_margins: list[float] = []
            for i in range(5):
                val = m_cols[i].number_input(
                    f"Yr {i + 1} EBIT Margin (%)",
                    min_value=-50.0,
                    max_value=80.0,
                    value=round(default_assump.ebit_margins[i] * 100.0, 2),
                    step=0.5,
                    key=f"{key_prefix}_m_{i}",
                )
                ebit_margins.append(val / 100.0)

            p1, p2, p3, p4 = st.columns(4)
            tax_rate_pct = p1.number_input(
                "Forecast Tax Rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=round(default_assump.tax_rate * 100.0, 2),
                step=0.5,
                key=f"{key_prefix}_tax",
            )
            da_pct = p2.number_input(
                "D&A (% of Revenue)",
                min_value=0.0,
                max_value=40.0,
                value=round(default_assump.da_pct_of_revenue * 100.0, 2),
                step=0.25,
                key=f"{key_prefix}_da",
            )
            capex_pct = p3.number_input(
                "CapEx (% of Revenue)",
                min_value=0.0,
                max_value=60.0,
                value=round(default_assump.capex_pct_of_revenue * 100.0, 2),
                step=0.25,
                key=f"{key_prefix}_capex",
            )
            nwc_pct = p4.number_input(
                "Incremental NWC (% of ΔRev)",
                min_value=-30.0,
                max_value=50.0,
                value=round(default_assump.nwc_pct_of_revenue * 100.0, 2),
                step=0.5,
                key=f"{key_prefix}_nwc",
            )

            active_fcff_assump = ForecastAssumptions(
                scenario=scenario,
                base_revenue=default_assump.base_revenue,
                revenue_growth_rates=growth_rates,
                ebit_margins=ebit_margins,
                tax_rate=tax_rate_pct / 100.0,
                da_pct_of_revenue=da_pct / 100.0,
                capex_pct_of_revenue=capex_pct / 100.0,
                nwc_pct_of_revenue=nwc_pct / 100.0,
                derivation_notes=default_assump.derivation_notes,
            )
            forecast_df = build_fcff_forecast(active_fcff_assump)
        else:
            st.caption(
                f"Base Revenue: {format_large_currency(default_fe_assump.base_revenue, currency, cfg.unit_mode)} | "
                "FCFE mode projects Net Income and distributable equity cash flows."
            )
            key_prefix = f"fcfe_{active_ticker}_{scenario}"
            g_cols = st.columns(5)
            growth_rates = []
            for i in range(5):
                val = g_cols[i].number_input(
                    f"Yr {i + 1} Rev Growth (%)",
                    min_value=-50.0,
                    max_value=100.0,
                    value=round(default_fe_assump.revenue_growth_rates[i] * 100.0, 2),
                    step=0.5,
                    key=f"{key_prefix}_g_{i}",
                )
                growth_rates.append(val / 100.0)

            m_cols = st.columns(5)
            net_margins = []
            for i in range(5):
                val = m_cols[i].number_input(
                    f"Yr {i + 1} Net Margin (%)",
                    min_value=-50.0,
                    max_value=70.0,
                    value=round(default_fe_assump.net_margin_rates[i] * 100.0, 2),
                    step=0.5,
                    key=f"{key_prefix}_nm_{i}",
                )
                net_margins.append(val / 100.0)

            fcfe_ratio_pct = st.number_input(
                "Distributable FCFE Conversion (% of Net Income)",
                min_value=10.0,
                max_value=100.0,
                value=round(default_fe_assump.fcfe_conversion_ratio * 100.0, 1),
                step=2.5,
                key=f"{key_prefix}_ratio",
            )
            active_fcfe_assump = FCFEForecastAssumptions(
                scenario=scenario,
                base_revenue=default_fe_assump.base_revenue,
                base_net_income=default_fe_assump.base_net_income,
                revenue_growth_rates=growth_rates,
                net_margin_rates=net_margins,
                fcfe_conversion_ratio=fcfe_ratio_pct / 100.0,
                derivation_notes=default_fe_assump.derivation_notes,
            )
            forecast_df = build_fcfe_forecast(active_fcfe_assump)

    # Calculate WACC / Cost of Equity
    shares = mkt.get("shares_outstanding")
    price = mkt.get("current_price")
    market_cap_eq = mkt.get("market_cap")
    if market_cap_eq is None and price is not None and shares is not None:
        market_cap_eq = price * shares

    cash_val = _latest_number(data.balance_sheet, "Total Cash & Equivalents")
    debt_val = _latest_number(data.balance_sheet, "Total Debt")

    wacc_res = None
    ke_only = calculate_cost_of_equity(cfg.risk_free_rate, cfg.beta, cfg.equity_risk_premium)

    if market_cap_eq is not None and market_cap_eq > 0 and debt_val is not None and debt_val >= 0:
        wacc_res = calculate_wacc(
            WACCInputs(
                risk_free_rate=cfg.risk_free_rate,
                beta=cfg.beta,
                equity_risk_premium=cfg.equity_risk_premium,
                pre_tax_cost_of_debt=cfg.pre_tax_cost_of_debt,
                tax_rate=cfg.wacc_tax_rate,
                market_cap_equity=market_cap_eq,
                total_debt=debt_val,
            )
        )

    if cfg.use_custom_discount_rate:
        active_discount_rate = cfg.custom_discount_rate
        discount_rate_source = "Custom Discount Rate Override"
    elif cfg.valuation_method == "FCFE":
        active_discount_rate = ke_only
        discount_rate_source = "Cost of Equity (CAPM)"
    elif wacc_res is not None:
        active_discount_rate = wacc_res.wacc
        discount_rate_source = "WACC (Market-Value Weighted)"
    else:
        active_discount_rate = ke_only
        discount_rate_source = "Cost of Equity (WACC fallback — Market Cap or Debt unavailable)"

    # Execute DCF Valuation
    dcf_result: DCFValuationResult | None = None
    dcf_error_msg: str | None = None

    pre_issues = validate_dcf_inputs(
        discount_rate=active_discount_rate,
        terminal_growth=cfg.terminal_growth_rate,
        shares_outstanding=shares,
    )

    can_run_dcf = not any(i.severity == "error" for i in pre_issues)
    if data.raw_currency_mismatch:
        can_run_dcf = False
        dcf_error_msg = (
            f"Currency mismatch between trading price ({mkt.get('currency')}) and financial statements "
            f"({mkt.get('financial_currency')}). Convert statements or override currency-aligned values before valuing."
        )
    elif cfg.valuation_method == "FCFF" and (cash_val is None or debt_val is None):
        can_run_dcf = False
        dcf_error_msg = (
            "Cash & Equivalents or Total Debt is unavailable for the Equity Value bridge. "
            "Enter missing values under 'Manual Data Overrides' in the sidebar (do not assume zero silently)."
        )
    elif (
        (cfg.valuation_method == "FCFF" and default_assump.base_revenue <= 0)
        or (cfg.valuation_method == "FCFE" and default_fe_assump.base_revenue <= 0)
    ):
        can_run_dcf = False
        dcf_error_msg = "Latest Revenue is unavailable or non-positive; enter Latest Revenue in Manual Overrides to build a forecast."

    if can_run_dcf:
        try:
            if cfg.valuation_method == "FCFF":
                dcf_result = run_fcff_dcf(
                    forecast_df=forecast_df,
                    wacc=active_discount_rate,
                    terminal_growth_rate=cfg.terminal_growth_rate,
                    cash=float(cash_val),
                    total_debt=float(debt_val),
                    shares_outstanding=float(shares),
                    current_market_price=price,
                    margin_of_safety_pct=cfg.margin_of_safety_pct,
                    non_operating_adjustments=cfg.non_operating_adjustments,
                    discount_rate_source=discount_rate_source,
                )
            else:
                dcf_result = run_fcfe_dcf(
                    forecast_df=forecast_df,
                    cost_of_equity=active_discount_rate,
                    terminal_growth_rate=cfg.terminal_growth_rate,
                    shares_outstanding=float(shares),
                    current_market_price=price,
                    margin_of_safety_pct=cfg.margin_of_safety_pct,
                    discount_rate_source=discount_rate_source,
                )
        except ValueError as exc:
            dcf_error_msg = str(exc)

    # Tab 1: Valuation Overview
    with tab_overview:
        if pre_issues:
            render_validation_issues(pre_issues)
        if dcf_error_msg:
            st.error(f"**Valuation Blocked:** {dcf_error_msg}")

        if dcf_result is not None:
            post_issues = validate_dcf_inputs(
                discount_rate=dcf_result.discount_rate,
                terminal_growth=dcf_result.terminal_growth_rate,
                shares_outstanding=dcf_result.shares_outstanding,
                terminal_value=dcf_result.terminal_value,
                pv_terminal_value=dcf_result.pv_terminal_value,
                total_value=(
                    dcf_result.enterprise_value
                    if dcf_result.enterprise_value is not None
                    else dcf_result.equity_value
                ),
            )
            # Show post-valuation warnings (e.g. high TV% or negative TV)
            post_warnings = [i for i in post_issues if i.field in {"tv_share", "terminal_value"}]
            if post_warnings:
                render_validation_issues(post_warnings)

            render_valuation_summary_cards(dcf_result, currency, cfg.unit_mode, industry_profile)

            c_left, c_right = st.columns(2)
            with c_left:
                st.plotly_chart(
                    chart_intrinsic_vs_market(
                        dcf_result.intrinsic_value_per_share,
                        dcf_result.current_market_price,
                        dcf_result.suggested_buy_price,
                        currency,
                    ),
                    width="stretch",
                )
            with c_right:
                st.plotly_chart(
                    chart_value_breakdown(dcf_result, currency, cfg.unit_mode),
                    width="stretch",
                )

            st.markdown("#### 5-Year Cash Flow Projections")
            fcf_col = "UFCF" if cfg.valuation_method == "FCFF" else "FCFE"
            st.dataframe(
                format_forecast_table_for_display(dcf_result.forecast_pv_table, currency, cfg.unit_mode),
                width="stretch",
                hide_index=True,
            )

            fc1, fc2 = st.columns(2)
            with fc1:
                st.plotly_chart(
                    chart_revenue_forecast(data.income_statement, dcf_result.forecast_pv_table, currency, cfg.unit_mode),
                    width="stretch",
                )
            with fc2:
                st.plotly_chart(
                    chart_fcf_forecast(dcf_result.forecast_pv_table, currency, cfg.unit_mode, fcf_col=fcf_col),
                    width="stretch",
                )

        st.markdown("#### Discount Rate & WACC Calculation")
        st.info(f"**Active Discount Rate ({cfg.valuation_method}):** {active_discount_rate:.2%} — Source: {discount_rate_source}")
        if wacc_res is not None:
            render_wacc_breakdown(
                wacc_res,
                currency,
                cfg.unit_mode,
                active_methodology=cfg.valuation_method,
                use_custom_discount_rate=cfg.use_custom_discount_rate,
            )
        else:
            st.warning(
                f"Full WACC requires Market Cap and Total Debt. Using Cost of Equity (CAPM) = **{ke_only:.2%}**."
            )

    # Tab 3: Historical Financials & Analysis
    with tab_history:
        st.subheader("Historical Financial Analysis")
        if not analysis.metrics_table.empty:
            s_cols = st.columns(4)
            rev_cagr = analysis.summary_stats.get("revenue_cagr")
            med_ebit = analysis.summary_stats.get("median_ebit_margin")
            med_fcf = analysis.summary_stats.get("median_fcf_margin")
            s_cols[0].metric("Revenue CAGR", f"{rev_cagr:.2%}" if rev_cagr is not None else "Unavailable")
            s_cols[1].metric("Median EBIT Margin", f"{med_ebit:.2%}" if med_ebit is not None else "Unavailable")
            s_cols[2].metric("Median FCF Margin", f"{med_fcf:.2%}" if med_fcf is not None else "Unavailable")
            s_cols[3].metric(
                "Historical FCF Trend",
                analysis.fcf_trend_direction,
                delta=f"CAGR: {analysis.fcf_cagr:.2%}" if analysis.fcf_cagr is not None else None,
            )

            hc1, hc2 = st.columns(2)
            with hc1:
                st.plotly_chart(
                    chart_historical_revenue(data.income_statement, currency, cfg.unit_mode),
                    width="stretch",
                )
            with hc2:
                st.plotly_chart(
                    chart_historical_fcf(analysis.metrics_table, currency, cfg.unit_mode),
                    width="stretch",
                )

            st.markdown("#### Calculated Historical Ratios & Margins")
            st.dataframe(
                format_analysis_table_for_display(analysis.metrics_table, currency, cfg.unit_mode),
                width="stretch",
            )
        else:
            st.warning("Insufficient historical data to compute ratios.")

        _, unit_suffix = get_display_scale(currency, cfg.unit_mode)
        st.markdown(f"#### Reported Financial Statements ({currency} {unit_suffix})")
        st.markdown("**Income Statement**")
        st.dataframe(
            format_statement_for_display(data.income_statement, data.provenance, "inc", currency, cfg.unit_mode),
            width="stretch",
        )
        st.markdown("**Cash Flow Statement** *(CapEx shown as positive cash outflow magnitude)*")
        st.dataframe(
            format_statement_for_display(data.cashflow_statement, data.provenance, "cf", currency, cfg.unit_mode),
            width="stretch",
        )
        st.markdown("**Balance Sheet**")
        st.dataframe(
            format_statement_for_display(data.balance_sheet, data.provenance, "bs", currency, cfg.unit_mode),
            width="stretch",
        )

    # Tab 4: Sensitivity Analysis
    with tab_sensitivity:
        st.subheader("Interactive Sensitivity Analysis (Discount Rate vs Terminal Growth)")
        if dcf_result is None or shares is None or shares <= 0:
            st.warning("Complete valid valuation inputs first to generate the sensitivity matrix.")
        else:
            default_center_r = min(30.0, max(2.0, round(active_discount_rate * 100.0, 2)))
            default_center_g = min(10.0, max(-2.0, round(cfg.terminal_growth_rate * 100.0, 2)))
            rc_key = f"sens_rc_{active_ticker}"
            gc_key = f"sens_gc_{active_ticker}"
            prev_r_key = f"sens_prev_r_{active_ticker}"
            prev_g_key = f"sens_prev_g_{active_ticker}"

            if st.session_state.get(prev_r_key) != default_center_r:
                st.session_state[rc_key] = default_center_r
                st.session_state[prev_r_key] = default_center_r
            if st.session_state.get(prev_g_key) != default_center_g:
                st.session_state[gc_key] = default_center_g
                st.session_state[prev_g_key] = default_center_g

            sc1, sc2, sc3, sc4 = st.columns(4)
            center_r_pct = sc1.number_input(
                "Center Discount Rate (%)",
                min_value=2.0,
                max_value=30.0,
                value=default_center_r,
                step=0.5,
                key=rc_key,
            )
            step_r_pct = sc2.number_input(
                "Discount Rate Step (%)",
                min_value=0.25,
                max_value=3.0,
                value=1.0,
                step=0.25,
                key=f"sens_rs_{active_ticker}",
            )
            center_g_pct = sc3.number_input(
                "Center Terminal Growth (%)",
                min_value=-2.0,
                max_value=10.0,
                value=default_center_g,
                step=0.5,
                key=gc_key,
            )
            step_g_pct = sc4.number_input(
                "Terminal Growth Step (%)",
                min_value=0.25,
                max_value=2.5,
                value=1.0,
                step=0.25,
                key=f"sens_gs_{active_ticker}",
            )

            r_list = build_range(center_r_pct / 100.0, step_r_pct / 100.0, count_each_side=2, min_val=0.01)
            g_list = build_range(center_g_pct / 100.0, step_g_pct / 100.0, count_each_side=2, min_val=-0.03)

            sens_res = compute_sensitivity_matrix(
                forecast_df=forecast_df,
                discount_rates=r_list,
                terminal_growth_rates=g_list,
                shares_outstanding=float(shares),
                methodology=cfg.valuation_method,
                cash=float(cash_val or 0.0),
                total_debt=float(debt_val or 0.0),
                non_operating_adjustments=cfg.non_operating_adjustments,
                current_market_price=price,
            )

            st.plotly_chart(
                chart_sensitivity_heatmap(sens_res, currency, current_market_price=price),
                width="stretch",
            )
            st.markdown("#### Intrinsic Value Per Share Sensitivity Table")
            st.caption(
                f"Active Methodology: **{cfg.valuation_method}** | Scenario: **{scenario}** — "
                "Green cells: >10% above current market price | Amber: within ±10% | Red: >10% below market price."
            )
            render_sensitivity_table(sens_res.intrinsic_value_matrix, currency, price)

    # Tab 5: Formula Audit & Disclaimer
    with tab_audit:
        st.subheader("Live Valuation Formula Audit & Transparency")
        if dcf_result is not None:
            active_assump = active_fcff_assump if cfg.valuation_method == "FCFF" else active_fcfe_assump
            audit_report = build_formula_audit(
                dcf_result=dcf_result,
                assumptions=active_assump,
                market_data=mkt,
                industry_profile=industry_profile,
                wacc_res=wacc_res,
                ke_only=ke_only,
                risk_free_rate=cfg.risk_free_rate,
                beta=cfg.beta,
                equity_risk_premium=cfg.equity_risk_premium,
                pre_tax_cost_of_debt=cfg.pre_tax_cost_of_debt,
                wacc_tax_rate=cfg.wacc_tax_rate,
                unit_mode=cfg.unit_mode,
            )
            render_formula_audit(audit_report)
            st.markdown("---")
        else:
            st.warning("Complete valid valuation inputs to generate the live numerical formula audit.")

        st.subheader("Reference Formulas & Disclaimer")
        st.markdown(
            """
            **1. Unlevered Free Cash Flow (UFCF)**
            ```
            UFCF = EBIT × (1 - Tax Rate) + D&A - CapEx - Change in NWC
            ```
            **2. CAPM & WACC**
            ```
            Cost of Equity (Ke) = Risk-Free Rate + Beta × Equity Risk Premium
            WACC = (E / (D + E)) × Ke + (D / (D + E)) × Cost of Debt × (1 - Tax Rate)
            ```
            *(Equity weight E uses Market Capitalization, never book equity.)*

            **3. Terminal Value (Gordon Growth Model)**
            ```
            Terminal Value = Final Year FCF × (1 + g) / (Discount Rate - g)
            ```
            **4. Enterprise Value & Equity Value (FCFF Mode)**
            ```
            Enterprise Value = Σ PV(UFCF_t) + PV(Terminal Value)
            Equity Value     = Enterprise Value + Cash - Total Debt + Non-Operating Adjustments
            Intrinsic Value  = Equity Value / Shares Outstanding
            Suggested Buy    = Intrinsic Value × (1 - Margin of Safety)
            ```
            **5. Financial Institutions (FCFE Mode)**
            - Banks, insurers, and financial institutions are flagged automatically.
            - Discounts Free Cash Flow to Equity (FCFE) directly at Cost of Equity (Ke) to estimate Equity Value without forcing an Enterprise Value bridge.
            """
        )
        st.warning(
            "**Disclaimer:** DCF Analyzer is an educational and analytical financial modeling tool. "
            "Intrinsic value estimates depend heavily on user assumptions (growth, margins, WACC, terminal growth) "
            "and historical data retrieved from Yahoo Finance (`yfinance`). This application does not provide "
            "investment advice or guaranteed stock-price predictions."
        )


if __name__ == "__main__":
    main()
