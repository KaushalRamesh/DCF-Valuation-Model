"""Formatted tables for historical statements, provenance badges, forecast projections, and sensitivity."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src.config import format_large_currency, format_percent, format_per_share, get_display_scale


def format_statement_for_display(
    df: pd.DataFrame,
    provenance: dict[str, str],
    prefix: str,
    currency: str,
    unit_mode: str = "auto",
) -> pd.DataFrame:
    """Format a financial statement DataFrame with currency units and source provenance tags."""
    if df.empty:
        return pd.DataFrame({"Status": ["No statement periods retrieved"]})

    _, suffix = get_display_scale(currency, unit_mode)
    display_df = pd.DataFrame(index=df.index)
    display_df["Source"] = [provenance.get(f"{prefix}:{idx}", "Unavailable") for idx in df.index]

    for col in df.columns:
        display_df[col] = [
            format_large_currency(val, currency=currency, unit_mode=unit_mode, decimals=2)
            for val in df[col]
        ]

    unit_note = f"({currency} {suffix})".strip()
    display_df.index.name = f"Line Item {unit_note}"
    return display_df


def format_analysis_table_for_display(
    metrics_table: pd.DataFrame,
    currency: str,
    unit_mode: str = "auto",
) -> pd.DataFrame:
    """Format historical ratios (%) and monetary NWC/FCF rows cleanly."""
    if metrics_table.empty:
        return pd.DataFrame()

    monetary_rows = {"Net Working Capital", "Change in NWC", "Free Cash Flow"}
    formatted = pd.DataFrame(index=metrics_table.index, columns=metrics_table.columns, dtype=str)

    for row in metrics_table.index:
        for col in metrics_table.columns:
            val = metrics_table.loc[row, col]
            if row in monetary_rows:
                formatted.loc[row, col] = format_large_currency(val, currency, unit_mode, decimals=2)
            else:
                formatted.loc[row, col] = format_percent(val, decimals=2, is_decimal=False)
    return formatted


def format_forecast_table_for_display(
    forecast_df: pd.DataFrame,
    currency: str,
    unit_mode: str = "auto",
) -> pd.DataFrame:
    """Format 5-year forecast table for clean readability."""
    if forecast_df.empty:
        return pd.DataFrame()

    pct_cols = {"Revenue Growth (%)", "EBIT Margin (%)", "Net Margin (%)"}
    skip_cols = {"Year", "Period", "Discount Factor"}

    out = pd.DataFrame()
    out["Year"] = forecast_df["Year"]

    for col in forecast_df.columns:
        if col in ("Year", "Period"):
            continue
        if col == "Discount Factor":
            out[col] = [f"{v:.4f}x" for v in forecast_df[col]]
        elif col in pct_cols:
            out[col] = [format_percent(v, decimals=2, is_decimal=False) for v in forecast_df[col]]
        else:
            out[col] = [format_large_currency(v, currency, unit_mode, decimals=2) for v in forecast_df[col]]
    return out


def render_sensitivity_table(
    matrix: pd.DataFrame,
    currency: str,
    current_market_price: Optional[float] = None,
) -> None:
    """Render sensitivity table with green/red conditional styling relative to current market price."""
    def _style_cell(val: float) -> str:
        if pd.isna(val):
            return "color: #9CA3AF; background-color: #1F2937;"
        if current_market_price is not None and current_market_price > 0:
            if val > current_market_price * 1.10:
                return "color: #34D399; background-color: rgba(16, 185, 129, 0.14); font-weight: 600;"
            if val < current_market_price * 0.90:
                return "color: #F87171; background-color: rgba(239, 68, 68, 0.14); font-weight: 600;"
            return "color: #FBBF24; background-color: rgba(245, 158, 11, 0.10);"
        return "color: #E6EDF3;"

    styled = matrix.style.map(_style_cell).format(
        lambda v: "Invalid (r ≤ g)" if pd.isna(v) else format_per_share(v, currency)
    )
    st.dataframe(styled, width="stretch")
