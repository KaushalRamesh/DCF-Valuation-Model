"""Interactive Plotly charts with dark financial dashboard styling and currency-aware formatting."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import get_currency_symbol, get_display_scale
from src.dcf_engine import DCFValuationResult
from src.sensitivity import SensitivityResult

BG_PAPER = "#0B0E14"
BG_PLOT = "#131722"
GRID_COLOR = "#1F2937"
TEXT_COLOR = "#E6EDF3"
MUTED_TEXT = "#9CA3AF"
GREEN = "#10B981"
RED = "#EF4444"
BLUE = "#3B82F6"
AMBER = "#F59E0B"
PURPLE = "#8B5CF6"


def _apply_dark_layout(fig: go.Figure, title: str, yaxis_title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT_COLOR)),
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        font=dict(color=TEXT_COLOR, family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=55, r=25, t=48, b=42),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(title=yaxis_title, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def chart_historical_revenue(
    income_df: pd.DataFrame,
    currency: str,
    unit_mode: str = "auto",
) -> go.Figure:
    """Bar chart of historical reported Revenue."""
    divisor, unit = get_display_scale(currency, unit_mode)
    sym = get_currency_symbol(currency)
    unit_label = f"{sym} {unit}".strip()

    fig = go.Figure()
    if not income_df.empty and "Revenue" in income_df.index:
        s = pd.to_numeric(income_df.loc["Revenue"], errors="coerce").dropna()
        scaled = s / divisor
        fig.add_trace(
            go.Bar(
                x=list(scaled.index),
                y=scaled.values,
                name="Revenue",
                marker_color=BLUE,
                text=[f"{sym}{v:,.1f}" for v in scaled.values],
                textposition="outside",
                hovertemplate=f"%{{x}}<br>Revenue: {sym}%{{y:,.2f}} {unit}<extra></extra>",
            )
        )
    return _apply_dark_layout(fig, "Historical Revenue", f"Amount ({unit_label})")


def chart_historical_fcf(
    metrics_table: pd.DataFrame,
    currency: str,
    unit_mode: str = "auto",
) -> go.Figure:
    """Bar chart of historical Free Cash Flow colored green for positive, red for negative."""
    divisor, unit = get_display_scale(currency, unit_mode)
    sym = get_currency_symbol(currency)
    unit_label = f"{sym} {unit}".strip()

    fig = go.Figure()
    if not metrics_table.empty and "Free Cash Flow" in metrics_table.index:
        s = pd.to_numeric(metrics_table.loc["Free Cash Flow"], errors="coerce").dropna()
        scaled = s / divisor
        colors = [GREEN if v >= 0 else RED for v in scaled.values]
        fig.add_trace(
            go.Bar(
                x=list(scaled.index),
                y=scaled.values,
                name="Free Cash Flow",
                marker_color=colors,
                text=[f"{sym}{v:,.1f}" for v in scaled.values],
                textposition="outside",
                hovertemplate=f"%{{x}}<br>FCF: {sym}%{{y:,.2f}} {unit}<extra></extra>",
            )
        )
    return _apply_dark_layout(fig, "Historical Free Cash Flow", f"Amount ({unit_label})")


def chart_revenue_forecast(
    income_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    currency: str,
    unit_mode: str = "auto",
) -> go.Figure:
    """Combined Historical vs 5-Year Projected Revenue chart."""
    divisor, unit = get_display_scale(currency, unit_mode)
    sym = get_currency_symbol(currency)
    unit_label = f"{sym} {unit}".strip()

    fig = go.Figure()
    if not income_df.empty and "Revenue" in income_df.index:
        hist = pd.to_numeric(income_df.loc["Revenue"], errors="coerce").dropna() / divisor
        fig.add_trace(
            go.Bar(
                x=list(hist.index),
                y=hist.values,
                name="Historical Revenue",
                marker_color="#4B5563",
                hovertemplate=f"%{{x}}<br>Historical: {sym}%{{y:,.2f}} {unit}<extra></extra>",
            )
        )

    if not forecast_df.empty and "Revenue" in forecast_df.columns:
        proj = forecast_df["Revenue"] / divisor
        fig.add_trace(
            go.Bar(
                x=forecast_df["Year"],
                y=proj.values,
                name="Forecast Revenue",
                marker_color=BLUE,
                text=[f"{sym}{v:,.1f}" for v in proj.values],
                textposition="outside",
                hovertemplate=f"%{{x}}<br>Forecast: {sym}%{{y:,.2f}} {unit}<extra></extra>",
            )
        )
    return _apply_dark_layout(fig, "Revenue Forecast (Historical vs 5Y Projection)", f"Amount ({unit_label})")


def chart_fcf_forecast(
    forecast_df: pd.DataFrame,
    currency: str,
    unit_mode: str = "auto",
    fcf_col: str = "UFCF",
) -> go.Figure:
    """5-year projected Free Cash Flow vs Present Value of Free Cash Flow."""
    divisor, unit = get_display_scale(currency, unit_mode)
    sym = get_currency_symbol(currency)
    unit_label = f"{sym} {unit}".strip()

    fig = go.Figure()
    if not forecast_df.empty and fcf_col in forecast_df.columns:
        fcf_scaled = forecast_df[fcf_col] / divisor
        colors = [GREEN if v >= 0 else RED for v in fcf_scaled.values]
        fig.add_trace(
            go.Bar(
                x=forecast_df["Year"],
                y=fcf_scaled.values,
                name=f"Projected {fcf_col}",
                marker_color=colors,
                hovertemplate=f"%{{x}}<br>{fcf_col}: {sym}%{{y:,.2f}} {unit}<extra></extra>",
            )
        )
        if "PV of FCF" in forecast_df.columns:
            pv_scaled = forecast_df["PV of FCF"] / divisor
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["Year"],
                    y=pv_scaled.values,
                    name="Present Value of FCF",
                    mode="lines+markers",
                    line=dict(color=AMBER, width=2.5),
                    marker=dict(size=7),
                    hovertemplate=f"%{{x}}<br>PV: {sym}%{{y:,.2f}} {unit}<extra></extra>",
                )
            )
    return _apply_dark_layout(fig, f"5-Year {fcf_col} & Present Value Forecast", f"Amount ({unit_label})")


def chart_value_breakdown(
    dcf_result: DCFValuationResult,
    currency: str,
    unit_mode: str = "auto",
) -> go.Figure:
    """Enterprise Value (FCFF) or Equity Value (FCFE) waterfall / component breakdown."""
    divisor, unit = get_display_scale(currency, unit_mode)
    sym = get_currency_symbol(currency)
    unit_label = f"{sym} {unit}".strip()

    if dcf_result.methodology == "FCFF" and dcf_result.enterprise_value is not None:
        labels = ["PV of 5Y FCF", "PV of Terminal Value", "Enterprise Value", "+ Cash", "- Total Debt", "Equity Value"]
        measures = ["relative", "relative", "total", "relative", "relative", "total"]
        values = [
            dcf_result.pv_forecast_cashflows / divisor,
            dcf_result.pv_terminal_value / divisor,
            0.0,
            (dcf_result.cash or 0.0) / divisor,
            -(dcf_result.total_debt or 0.0) / divisor,
            0.0,
        ]
        title = "Enterprise Value to Equity Value Bridge"
    else:
        labels = ["PV of 5Y FCFE", "PV of Terminal Value", "Estimated Equity Value"]
        measures = ["relative", "relative", "total"]
        values = [
            dcf_result.pv_forecast_cashflows / divisor,
            dcf_result.pv_terminal_value / divisor,
            0.0,
        ]
        title = "FCFE Equity Value Breakdown"

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector=dict(line=dict(color="#4B5563", width=1)),
            increasing=dict(marker=dict(color=GREEN)),
            decreasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=BLUE)),
            texttemplate=f"{sym}%{{y:,.1f}}",
            textposition="outside",
        )
    )
    return _apply_dark_layout(fig, title, f"Amount ({unit_label})")


def chart_intrinsic_vs_market(
    intrinsic_price: float,
    market_price: Optional[float],
    suggested_buy_price: float,
    currency: str,
) -> go.Figure:
    """Comparison bar chart of Estimated Intrinsic Value, Margin-of-Safety Price, and Current Market Price."""
    sym = get_currency_symbol(currency)
    labels = ["Estimated Intrinsic Value", "Target Buy Price (w/ MoS)"]
    values = [intrinsic_price, suggested_buy_price]
    colors = [BLUE, AMBER]

    if market_price is not None and market_price > 0 and not np.isnan(market_price):
        labels.append("Current Market Price")
        values.append(market_price)
        colors.append(GREEN if intrinsic_price >= market_price else RED)

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{sym}{v:,.2f}" for v in values],
            textposition="outside",
            hovertemplate=f"%{{x}}<br>{sym}%{{y:,.2f}}<extra></extra>",
        )
    )
    return _apply_dark_layout(fig, "Estimated Intrinsic Value vs Market Price", f"Price per Share ({sym})")


def chart_sensitivity_heatmap(
    sensitivity: SensitivityResult,
    currency: str,
    current_market_price: Optional[float] = None,
) -> go.Figure:
    """Interactive heatmap of Intrinsic Value per Share across Discount Rate vs Terminal Growth Rate."""
    sym = get_currency_symbol(currency)
    mat = sensitivity.intrinsic_value_matrix

    z = mat.values
    text_mat = [
        [f"{sym}{val:,.2f}" if pd.notna(val) else "Invalid (r ≤ g)" for val in row]
        for row in z
    ]

    # Center colorscale on current market price if available so green > market, red < market
    if current_market_price is not None and current_market_price > 0:
        colorscale = [[0.0, "#7F1D1D"], [0.5, "#1F2937"], [1.0, "#065F46"]]
        zmid = current_market_price
    else:
        colorscale = "Viridis"
        zmid = None

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(mat.columns),
            y=list(mat.index),
            text=text_mat,
            texttemplate="%{text}",
            textfont=dict(size=11, color=TEXT_COLOR),
            colorscale=colorscale,
            zmid=zmid,
            hovertemplate=(
                "Discount Rate: %{y}<br>"
                "Terminal Growth: %{x}<br>"
                f"Intrinsic Value: {sym}%{{z:,.2f}}<extra></extra>"
            ),
            colorbar=dict(title=f"Value ({sym})"),
        )
    )
    fig.update_layout(
        title=dict(
            text="Sensitivity Heatmap: Intrinsic Value / Share (Discount Rate vs Terminal Growth)",
            font=dict(size=15, color=TEXT_COLOR),
        ),
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        font=dict(color=TEXT_COLOR, family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=70, r=30, t=55, b=55),
        xaxis=dict(title="Terminal Growth Rate"),
        yaxis=dict(title="Discount Rate (WACC / Ke)", autorange="reversed"),
    )
    return fig
