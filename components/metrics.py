"""Top header, KPI metric cards, WACC breakdown, and validation banners."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.audit import FormulaAuditReport
from src.config import format_large_currency, format_percent, format_per_share
from src.dcf_engine import DCFValuationResult
from src.industry import IndustryProfile, get_methodology_explanation
from src.validation import ValidationIssue
from src.wacc import WACCResult


def render_company_header(market_data: dict[str, Any], provenance: dict[str, str], unit_mode: str) -> None:
    """Render the company summary header with price, exchange, sector, industry, and update timestamp."""
    currency = market_data.get("currency", "USD")
    price = market_data.get("current_price")
    mcap = market_data.get("market_cap")
    shares = market_data.get("shares_outstanding")
    beta = market_data.get("beta")

    price_tag = " [Manual]" if provenance.get("market:current_price") == "Manual" else ""
    shares_tag = " [Manual]" if provenance.get("market:shares_outstanding") == "Manual" else ""

    st.markdown(
        f"""
        <div class="dcf-header-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                <div>
                    <div class="dcf-company-title">{market_data.get("company_name", market_data.get("ticker"))}
                        <span class="dcf-ticker-badge">{market_data.get("ticker")}</span>
                    </div>
                    <div class="dcf-meta-row">
                        <span><strong>Exchange:</strong> {market_data.get("exchange", "Unavailable")}</span> &nbsp;•&nbsp;
                        <span><strong>Sector:</strong> {market_data.get("sector", "Unavailable")}</span> &nbsp;•&nbsp;
                        <span><strong>Industry:</strong> {market_data.get("industry", "Unavailable")}</span> &nbsp;•&nbsp;
                        <span><strong>Currency:</strong> {currency}</span>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="dcf-price-label">Current Market Price{price_tag}</div>
                    <div class="dcf-price-value">{format_per_share(price, currency)}</div>
                    <div class="dcf-timestamp">Last updated: {market_data.get("last_updated", "Unavailable")} ({market_data.get("source", "yfinance")})</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Cap", format_large_currency(mcap, currency, unit_mode))
    c2.metric(f"Shares Outstanding{shares_tag}", f"{shares:,.0f}" if shares else "Unavailable")
    c3.metric("Reported Beta", f"{beta:.2f}" if beta is not None else "Unavailable")
    c4.metric("Financial Statement Currency", market_data.get("financial_currency", currency))


def render_validation_issues(issues: list[ValidationIssue]) -> None:
    """Display structured validation errors, warnings, and informational notices."""
    if not issues:
        return
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    infos = [i for i in issues if i.severity == "info"]

    for issue in errors:
        st.error(f"**{issue.category} ({issue.field}):** {issue.message}")
    if warnings:
        with st.expander(f"Data & Model Warnings ({len(warnings)})", expanded=True):
            for w in warnings:
                st.warning(f"**{w.category} ({w.field}):** {w.message}")
    if infos:
        with st.expander(f"Analytical Notes ({len(infos)})", expanded=False):
            for info in infos:
                st.info(f"**{info.category} ({info.field}):** {info.message}")


def render_valuation_summary_cards(
    result: DCFValuationResult,
    currency: str,
    unit_mode: str,
    industry_profile: IndustryProfile,
) -> None:
    """Display headline Intrinsic Value, Upside/Downside, Margin of Safety, and EV/Equity bridge."""
    upside_pct = result.upside_percent
    upside_abs = result.upside_absolute

    if result.valuation_label == "Below estimated value":
        badge_color = "#10B981"
    elif result.valuation_label == "Above estimated value":
        badge_color = "#EF4444"
    else:
        badge_color = "#F59E0B"

    active_explanation = get_methodology_explanation(result.methodology)
    mismatch_note = ""
    if result.methodology != industry_profile.recommended_method:
        mismatch_note = (
            f" Note: Industry profile ({industry_profile.sector} / {industry_profile.industry}) "
            f"recommends {industry_profile.recommended_method}, while active valuation uses {result.methodology}."
        )

    st.markdown(
        f"""
        <div class="dcf-status-banner" style="border-left: 4px solid {badge_color};">
            <div style="line-height: 1.4;">
                <span style="color:{badge_color}; font-weight:700; font-size:0.92rem;">
                    Valuation Assessment: {result.valuation_label}
                </span>
                <span style="color:#9CA3AF; margin-left:8px; font-size:0.82rem;">
                    ({active_explanation}{mismatch_note})
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Estimated Intrinsic Value / Share", format_per_share(result.intrinsic_value_per_share, currency))
    k2.metric(
        "Current Market Price",
        format_per_share(result.current_market_price, currency),
    )

    if upside_pct is not None and upside_abs is not None:
        delta_color = "normal"
        k3.metric(
            "Implied Upside / Downside",
            f"{upside_pct:+.2f}%",
            delta=f"{format_per_share(upside_abs, currency)} / share",
            delta_color=delta_color,
        )
    else:
        k3.metric("Implied Upside / Downside", "Unavailable")

    k4.metric(
        f"Suggested Buy Price ({result.margin_of_safety_pct:.0%} MoS)",
        format_per_share(result.suggested_buy_price, currency),
    )

    st.markdown("#### Valuation Bridge & Terminal Value Breakdown")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("PV of 5Y Cash Flows", format_large_currency(result.pv_forecast_cashflows, currency, unit_mode))
    b2.metric("Undiscounted Terminal Value", format_large_currency(result.terminal_value, currency, unit_mode))
    b3.metric("PV of Terminal Value", format_large_currency(result.pv_terminal_value, currency, unit_mode))
    b4.metric(
        "TV % of Total Value",
        f"{result.tv_pct_of_total_value:.1f}%" if result.tv_pct_of_total_value is not None else "N/A",
    )

    if result.methodology == "FCFF":
        b5.metric("Enterprise Value", format_large_currency(result.enterprise_value, currency, unit_mode))
        if abs(result.non_operating_adjustments) > 0:
            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("Cash & Equivalents", format_large_currency(result.cash, currency, unit_mode))
            e2.metric("Total Debt", format_large_currency(result.total_debt, currency, unit_mode))
            e3.metric("Net Debt (Debt - Cash)", format_large_currency(result.net_debt, currency, unit_mode))
            e4.metric("Non-Op Adjustments", format_large_currency(result.non_operating_adjustments, currency, unit_mode))
            e5.metric("Equity Value", format_large_currency(result.equity_value, currency, unit_mode))
        else:
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Cash & Equivalents", format_large_currency(result.cash, currency, unit_mode))
            e2.metric("Total Debt", format_large_currency(result.total_debt, currency, unit_mode))
            e3.metric("Net Debt (Debt - Cash)", format_large_currency(result.net_debt, currency, unit_mode))
            e4.metric("Equity Value", format_large_currency(result.equity_value, currency, unit_mode))
        st.caption(
            f"FCFF Bridge (discounted at {result.discount_rate:.2%}): "
            "Enterprise Value = PV(5Y UFCF) + PV(Terminal Value); "
            "Equity Value = Enterprise Value + Cash - Total Debt"
            + (" + Non-Operating Adjustments." if abs(result.non_operating_adjustments) > 0 else ".")
        )
    else:
        b5.metric("Equity Value (FCFE)", format_large_currency(result.equity_value, currency, unit_mode))
        st.caption(
            f"Direct Equity Valuation (FCFE, discounted at {result.discount_rate:.2%}): "
            "Equity Value = PV(5Y FCFE) + PV(Terminal Value). "
            "Enterprise Value is not computed and debt is not subtracted again."
        )


def render_wacc_breakdown(
    wacc_res: WACCResult,
    currency: str,
    unit_mode: str,
    active_methodology: str = "FCFF",
    use_custom_discount_rate: bool = False,
) -> None:
    """Display transparent CAPM & WACC formula breakdown."""
    c1, c2, c3, c4 = st.columns(4)
    ke_label = "Cost of Equity (CAPM — Active)" if (active_methodology == "FCFE" and not use_custom_discount_rate) else "Cost of Equity (CAPM)"
    wacc_label = "Calculated WACC (Active)" if (active_methodology == "FCFF" and not use_custom_discount_rate) else "Calculated WACC (Reference)"
    c1.metric(ke_label, format_percent(wacc_res.cost_of_equity))
    c2.metric("After-Tax Cost of Debt", format_percent(wacc_res.after_tax_cost_of_debt))
    c3.metric("Capital Weights (E / D)", f"{wacc_res.equity_weight:.1%} / {wacc_res.debt_weight:.1%}")
    c4.metric(wacc_label, format_percent(wacc_res.wacc))
    st.caption(
        f"Equity Weight uses Market Capitalization ({format_large_currency(wacc_res.equity_value, currency, unit_mode)}), "
        f"not book equity."
    )
    st.code(wacc_res.formula_summary, language="text")


def render_formula_audit(audit: FormulaAuditReport) -> None:
    """Render the live, step-by-step DCF Formula Audit tables and reconciliation status."""
    if audit.reconciliation_passed:
        st.success(
            f"**Formula Audit Reconciliation Passed** — Active Methodology: **{audit.methodology}** | "
            f"Scenario: **{audit.scenario}** | Max reconciliation delta: `{audit.max_reconciliation_error:.2e}`"
        )
    else:
        st.error(
            f"**Formula Audit Reconciliation Mismatch** — Max delta: `{audit.max_reconciliation_error:.4f}`"
        )

    if not audit.methodology_aligned:
        st.warning(
            f"Selected methodology (**{audit.methodology}**) differs from industry-recommended "
            f"methodology (**{audit.recommended_methodology}**)."
        )

    for note in audit.optimistic_check_notes:
        st.caption(note)

    st.markdown("##### 1. Major Inputs & Unit Consistency Check")
    st.dataframe(audit.inputs_table, width="stretch", hide_index=True)

    st.markdown("##### 2. Discount Rate Calculation Audit")
    st.dataframe(audit.discount_rate_audit, width="stretch", hide_index=True)

    st.markdown("##### 3. 5-Year Cash Flow Discounting Audit")
    st.dataframe(audit.cashflow_audit_table, width="stretch", hide_index=True)

    st.markdown("##### 4. Terminal Value & Present Value Audit")
    st.dataframe(audit.terminal_value_audit, width="stretch", hide_index=True)

    st.markdown("##### 5. Valuation Bridge & Intrinsic Value Per Share Audit")
    st.dataframe(audit.bridge_audit, width="stretch", hide_index=True)
