"""Detailed formula audit builder for verifying DCF inputs, discounting, bridge, and per-share math."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.config import format_large_currency, format_per_share, get_display_scale
from src.dcf_engine import DCFValuationResult, calculate_terminal_value_gordon
from src.forecasting import FCFEForecastAssumptions, ForecastAssumptions
from src.industry import IndustryProfile
from src.wacc import WACCResult


@dataclass(frozen=True)
class FormulaAuditReport:
    """Complete, independently recomputed audit of a DCF valuation run."""

    methodology: str
    recommended_methodology: str
    methodology_aligned: bool
    scenario: str
    currency: str
    unit_mode: str
    unit_label: str
    inputs_table: pd.DataFrame
    discount_rate_audit: pd.DataFrame
    cashflow_audit_table: pd.DataFrame
    terminal_value_audit: pd.DataFrame
    bridge_audit: pd.DataFrame
    optimistic_check_notes: list[str]
    reconciliation_passed: bool
    max_reconciliation_error: float


def build_formula_audit(
    dcf_result: DCFValuationResult,
    assumptions: Union[ForecastAssumptions, FCFEForecastAssumptions],
    market_data: dict[str, Any],
    industry_profile: IndustryProfile,
    wacc_res: Optional[WACCResult],
    ke_only: float,
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
    pre_tax_cost_of_debt: float,
    wacc_tax_rate: float,
    unit_mode: str = "auto",
) -> FormulaAuditReport:
    """Build an auditable, independently verified breakdown of every major DCF input and step."""
    currency = str(market_data.get("currency") or "USD")
    fin_currency = str(market_data.get("financial_currency") or currency)
    divisor, suffix = get_display_scale(currency, unit_mode)
    unit_label = f"{currency} {suffix}".strip()

    price = market_data.get("current_price")
    shares = float(dcf_result.shares_outstanding)
    reported_mcap = market_data.get("market_cap")
    implied_mcap = (float(price) * shares) if (price is not None and price > 0) else None

    # 1. Major Inputs & Unit Consistency Table
    input_rows: list[dict[str, str]] = [
        {"Category": "Methodology", "Item": "Active Valuation Mode", "Value": dcf_result.methodology, "Notes": f"Industry recommendation: {industry_profile.recommended_method}"},
        {"Category": "Methodology", "Item": "Forecast Scenario", "Value": assumptions.scenario, "Notes": "Conservative / Base / Optimistic"},
        {"Category": "Units", "Item": "Trading / Statement Currency", "Value": f"{currency} / {fin_currency}", "Notes": f"Display scale: ÷ {divisor:,.0f} ({unit_label})"},
        {"Category": "Market", "Item": "Current Market Price", "Value": format_per_share(price, currency), "Notes": "Raw per-share currency"},
        {"Category": "Market", "Item": "Shares Outstanding", "Value": f"{shares:,.0f} ({shares / divisor:,.2f} {suffix})".strip(), "Notes": "Exact share count used in Equity Value / Shares"},
        {"Category": "Market", "Item": "Reported Market Cap", "Value": format_large_currency(reported_mcap, currency, unit_mode), "Notes": "Used as Equity Weight E in WACC"},
        {"Category": "Market", "Item": "Implied Market Cap (Price × Shares)", "Value": format_large_currency(implied_mcap, currency, unit_mode), "Notes": "Cross-check for unit consistency"},
        {"Category": "Forecast", "Item": "Base Year Revenue", "Value": format_large_currency(assumptions.base_revenue, currency, unit_mode), "Notes": "Latest reported annual Revenue"},
        {"Category": "Forecast", "Item": "Yr 1–5 Revenue Growth", "Value": ", ".join(f"{g * 100:.2f}%" for g in assumptions.revenue_growth_rates), "Notes": "Annual top-line growth rates"},
    ]

    if isinstance(assumptions, ForecastAssumptions):
        capex_display = (
            ", ".join(f"{c * 100:.2f}%" for c in assumptions.capex_pct_of_revenue)
            if isinstance(assumptions.capex_pct_of_revenue, (list, tuple))
            else f"{assumptions.capex_pct_of_revenue * 100:.2f}%"
        )
        input_rows.extend([
            {"Category": "Forecast (FCFF)", "Item": "Yr 1–5 EBIT Margins", "Value": ", ".join(f"{m * 100:.2f}%" for m in assumptions.ebit_margins), "Notes": "EBIT / Revenue"},
            {"Category": "Forecast (FCFF)", "Item": "Forecast Tax Rate", "Value": f"{assumptions.tax_rate * 100:.2f}%", "Notes": "Applied to EBIT for NOPAT"},
            {"Category": "Forecast (FCFF)", "Item": "D&A (% of Revenue)", "Value": f"{assumptions.da_pct_of_revenue * 100:.2f}%", "Notes": "Added back to NOPAT"},
            {"Category": "Forecast (FCFF)", "Item": "Yr 1–5 CapEx (% of Revenue)", "Value": capex_display, "Notes": "Subtracted from operating cash flow"},
            {"Category": "Forecast (FCFF)", "Item": "Incremental NWC (% of ΔRev)", "Value": f"{assumptions.nwc_pct_of_revenue * 100:.2f}%", "Notes": "ΔNWC = (Rev_t - Rev_{t-1}) × NWC%"},
            {"Category": "Bridge (FCFF)", "Item": "Cash & Equivalents", "Value": format_large_currency(dcf_result.cash, currency, unit_mode), "Notes": "Added to Enterprise Value"},
            {"Category": "Bridge (FCFF)", "Item": "Total Debt", "Value": format_large_currency(dcf_result.total_debt, currency, unit_mode), "Notes": "Subtracted from Enterprise Value"},
            {"Category": "Bridge (FCFF)", "Item": "Non-Operating Adjustments", "Value": format_large_currency(dcf_result.non_operating_adjustments, currency, unit_mode), "Notes": "Added to Equity Value"},
        ])
    else:
        input_rows.extend([
            {"Category": "Forecast (FCFE)", "Item": "Yr 1–5 Net Margins", "Value": ", ".join(f"{m * 100:.2f}%" for m in assumptions.net_margin_rates), "Notes": "Net Income / Revenue"},
            {"Category": "Forecast (FCFE)", "Item": "Distributable FCFE Ratio", "Value": f"{assumptions.fcfe_conversion_ratio * 100:.1f}%", "Notes": "FCFE = Net Income × Ratio"},
            {"Category": "Bridge (FCFE)", "Item": "Cash / Debt Bridge", "Value": "Not Applied (Direct Equity)", "Notes": "FCFE already reflects equity cash flow; debt is not subtracted again"},
        ])

    inputs_table = pd.DataFrame(input_rows)

    # 2. Discount Rate Audit Table
    dr_rows: list[dict[str, str]] = [
        {
            "Component": "Cost of Equity (CAPM)",
            "Formula": "Ke = Rf + β × ERP",
            "Substituted Calculation": f"{risk_free_rate:.2%} + {beta:.2f} × {equity_risk_premium:.2%}",
            "Result": f"{ke_only:.4%}",
        }
    ]
    if wacc_res is not None:
        dr_rows.extend([
            {
                "Component": "After-Tax Cost of Debt",
                "Formula": "Kd_after = Kd × (1 - Tax)",
                "Substituted Calculation": f"{pre_tax_cost_of_debt:.2%} × (1 - {wacc_tax_rate:.2%})",
                "Result": f"{wacc_res.after_tax_cost_of_debt:.4%}",
            },
            {
                "Component": "Capital Structure Weights",
                "Formula": "E / (D+E) and D / (D+E)",
                "Substituted Calculation": (
                    f"E={format_large_currency(wacc_res.equity_value, currency, unit_mode)}, "
                    f"D={format_large_currency(wacc_res.debt_value, currency, unit_mode)}"
                ),
                "Result": f"{wacc_res.equity_weight:.2%} E / {wacc_res.debt_weight:.2%} D",
            },
            {
                "Component": "Calculated WACC",
                "Formula": "WACC = wE × Ke + wD × Kd_after",
                "Substituted Calculation": (
                    f"({wacc_res.equity_weight:.2%} × {wacc_res.cost_of_equity:.2%}) + "
                    f"({wacc_res.debt_weight:.2%} × {wacc_res.after_tax_cost_of_debt:.2%})"
                ),
                "Result": f"{wacc_res.wacc:.4%}",
            },
        ])
    dr_rows.append(
        {
            "Component": "Active Discount Rate Used",
            "Formula": dcf_result.discount_rate_source or ("WACC" if dcf_result.methodology == "FCFF" else "Ke"),
            "Substituted Calculation": f"Applied to {dcf_result.methodology} cash flows & Terminal Value",
            "Result": f"{dcf_result.discount_rate:.4%}",
        }
    )
    discount_rate_audit = pd.DataFrame(dr_rows)

    # 3. Cash Flow & PV Schedule Audit (independently recomputed)
    df = dcf_result.forecast_pv_table.copy()
    fcf_col = "UFCF" if dcf_result.methodology == "FCFF" else "FCFE"
    periods = df["Period"].astype(float).values
    fcfs = df[fcf_col].astype(float).values
    r = float(dcf_result.discount_rate)
    g = float(dcf_result.terminal_growth_rate)

    recomputed_dfs = (1.0 + r) ** periods
    recomputed_pvs = fcfs / recomputed_dfs
    recomputed_pv_fcf_sum = float(np.sum(recomputed_pvs))

    cf_audit_rows = []
    for idx in range(len(df)):
        cf_audit_rows.append(
            {
                "Year": df["Year"].iloc[idx],
                "Period (t)": int(periods[idx]),
                "Revenue": format_large_currency(float(df["Revenue"].iloc[idx]), currency, unit_mode),
                f"{fcf_col}_t": format_large_currency(float(fcfs[idx]), currency, unit_mode),
                "Discount Factor (1+r)^t": f"{recomputed_dfs[idx]:.6f}x",
                "PV Formula": f"{fcf_col}_{int(periods[idx])} / (1 + {r:.4%})^{int(periods[idx])}",
                f"PV({fcf_col}_t)": format_large_currency(float(recomputed_pvs[idx]), currency, unit_mode),
            }
        )
    cashflow_audit_table = pd.DataFrame(cf_audit_rows)

    # 4. Terminal Value & Discounting Audit (independently recomputed)
    final_fcf = float(fcfs[-1])
    n_years = float(periods[-1])
    recomputed_tv = calculate_terminal_value_gordon(final_fcf, r, g)
    terminal_df = (1.0 + r) ** n_years
    recomputed_pv_tv = recomputed_tv / terminal_df

    tv_rows = [
        {
            "Step": f"Final-Year Cash Flow ({fcf_col}_{int(n_years)})",
            "Formula": f"Year {int(n_years)} {fcf_col}",
            "Substituted Calculation": format_large_currency(final_fcf, currency, unit_mode),
            "Result": format_large_currency(final_fcf, currency, unit_mode),
        },
        {
            "Step": "Undiscounted Terminal Value",
            "Formula": f"TV = {fcf_col}_N × (1 + g) / (r - g)",
            "Substituted Calculation": (
                f"{format_large_currency(final_fcf, currency, unit_mode)} × (1 + {g:.2%}) / "
                f"({r:.4%} - {g:.2%})"
            ),
            "Result": format_large_currency(recomputed_tv, currency, unit_mode),
        },
        {
            "Step": f"Terminal Discount Factor (N = {int(n_years)})",
            "Formula": "(1 + r)^N",
            "Substituted Calculation": f"(1 + {r:.4%})^{int(n_years)}",
            "Result": f"{terminal_df:.6f}x",
        },
        {
            "Step": "Present Value of Terminal Value",
            "Formula": "PV(TV) = TV / (1 + r)^N",
            "Substituted Calculation": f"{format_large_currency(recomputed_tv, currency, unit_mode)} / {terminal_df:.6f}",
            "Result": format_large_currency(recomputed_pv_tv, currency, unit_mode),
        },
    ]
    terminal_value_audit = pd.DataFrame(tv_rows)

    # 5. Valuation Bridge & Per-Share Audit (independently recomputed)
    if dcf_result.methodology == "FCFF":
        recomputed_ev = recomputed_pv_fcf_sum + recomputed_pv_tv
        cash_v = float(dcf_result.cash or 0.0)
        debt_v = float(dcf_result.total_debt or 0.0)
        non_op_v = float(dcf_result.non_operating_adjustments or 0.0)
        recomputed_eq = recomputed_ev + cash_v - debt_v + non_op_v
        bridge_rows = [
            {
                "Step": "PV of 5-Year UFCF",
                "Formula": "Σ PV(UFCF_t)",
                "Substituted Calculation": "Sum of discounted Year 1–5 UFCF",
                "Result": format_large_currency(recomputed_pv_fcf_sum, currency, unit_mode),
            },
            {
                "Step": "PV of Terminal Value",
                "Formula": "TV / (1 + WACC)^N",
                "Substituted Calculation": "Discounted from Year 5 to present",
                "Result": format_large_currency(recomputed_pv_tv, currency, unit_mode),
            },
            {
                "Step": "Enterprise Value",
                "Formula": "EV = PV(5Y UFCF) + PV(TV)",
                "Substituted Calculation": (
                    f"{format_large_currency(recomputed_pv_fcf_sum, currency, unit_mode)} + "
                    f"{format_large_currency(recomputed_pv_tv, currency, unit_mode)}"
                ),
                "Result": format_large_currency(recomputed_ev, currency, unit_mode),
            },
            {
                "Step": "Equity Value Bridge",
                "Formula": "Equity = EV + Cash - Debt + Non-Op Adj",
                "Substituted Calculation": (
                    f"{format_large_currency(recomputed_ev, currency, unit_mode)} + "
                    f"{format_large_currency(cash_v, currency, unit_mode)} - "
                    f"{format_large_currency(debt_v, currency, unit_mode)}"
                    + (f" + {format_large_currency(non_op_v, currency, unit_mode)}" if non_op_v != 0 else "")
                ),
                "Result": format_large_currency(recomputed_eq, currency, unit_mode),
            },
        ]
    else:
        recomputed_ev = None
        recomputed_eq = recomputed_pv_fcf_sum + recomputed_pv_tv
        bridge_rows = [
            {
                "Step": "PV of 5-Year FCFE",
                "Formula": "Σ PV(FCFE_t)",
                "Substituted Calculation": "Sum of discounted Year 1–5 FCFE at Ke",
                "Result": format_large_currency(recomputed_pv_fcf_sum, currency, unit_mode),
            },
            {
                "Step": "PV of Terminal Value",
                "Formula": "TV / (1 + Ke)^N",
                "Substituted Calculation": "Discounted from Year 5 to present at Ke",
                "Result": format_large_currency(recomputed_pv_tv, currency, unit_mode),
            },
            {
                "Step": "Enterprise Value",
                "Formula": "N/A in FCFE Mode",
                "Substituted Calculation": "FCFE values Equity directly (no Cash/Debt bridge)",
                "Result": "N/A (Direct Equity)",
            },
            {
                "Step": "Equity Value (Direct FCFE)",
                "Formula": "Equity = PV(5Y FCFE) + PV(TV)",
                "Substituted Calculation": (
                    f"{format_large_currency(recomputed_pv_fcf_sum, currency, unit_mode)} + "
                    f"{format_large_currency(recomputed_pv_tv, currency, unit_mode)}"
                ),
                "Result": format_large_currency(recomputed_eq, currency, unit_mode),
            },
        ]

    recomputed_intrinsic = recomputed_eq / shares
    recomputed_buy = recomputed_intrinsic * (1.0 - dcf_result.margin_of_safety_pct)
    bridge_rows.extend([
        {
            "Step": "Intrinsic Value Per Share",
            "Formula": "Equity Value / Shares Outstanding",
            "Substituted Calculation": f"{format_large_currency(recomputed_eq, currency, unit_mode)} / {shares:,.0f} shares",
            "Result": format_per_share(recomputed_intrinsic, currency),
        },
        {
            "Step": f"Target Buy Price ({dcf_result.margin_of_safety_pct:.0%} MoS)",
            "Formula": "Intrinsic Value × (1 - MoS)",
            "Substituted Calculation": f"{format_per_share(recomputed_intrinsic, currency)} × (1 - {dcf_result.margin_of_safety_pct:.0%})",
            "Result": format_per_share(recomputed_buy, currency),
        },
    ])
    bridge_audit = pd.DataFrame(bridge_rows)

    # Check exact numerical reconciliation against DCFValuationResult
    diffs = [
        abs(recomputed_pv_fcf_sum - dcf_result.pv_forecast_cashflows),
        abs(recomputed_tv - dcf_result.terminal_value),
        abs(recomputed_pv_tv - dcf_result.pv_terminal_value),
        abs(recomputed_eq - dcf_result.equity_value),
        abs(recomputed_intrinsic - dcf_result.intrinsic_value_per_share),
    ]
    if recomputed_ev is not None and dcf_result.enterprise_value is not None:
        diffs.append(abs(recomputed_ev - dcf_result.enterprise_value))
    max_err = max(diffs) if diffs else 0.0
    rel_tol = max(1e-6, abs(dcf_result.equity_value) * 1e-9)
    reconciliation_passed = max_err <= rel_tol

    optimistic_notes = []
    if assumptions.scenario == "Optimistic":
        if isinstance(assumptions, ForecastAssumptions):
            optimistic_notes.append(
                "Optimistic scenario applies +2.50% to Revenue growth, +1.50% to EBIT margin, "
                "and -0.50% to CapEx % of Revenue relative to Base defaults."
            )
        else:
            optimistic_notes.append(
                "Optimistic scenario applies +2.50% to Revenue growth, +1.50% to Net margin, "
                "and +8.00% to Distributable FCFE conversion ratio relative to Base defaults."
            )
    else:
        optimistic_notes.append(f"Active scenario is '{assumptions.scenario}'.")

    return FormulaAuditReport(
        methodology=dcf_result.methodology,
        recommended_methodology=industry_profile.recommended_method,
        methodology_aligned=(dcf_result.methodology == industry_profile.recommended_method),
        scenario=assumptions.scenario,
        currency=currency,
        unit_mode=unit_mode,
        unit_label=unit_label,
        inputs_table=inputs_table,
        discount_rate_audit=discount_rate_audit,
        cashflow_audit_table=cashflow_audit_table,
        terminal_value_audit=terminal_value_audit,
        bridge_audit=bridge_audit,
        optimistic_check_notes=optimistic_notes,
        reconciliation_passed=reconciliation_passed,
        max_reconciliation_error=max_err,
    )
