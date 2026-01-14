"""Chart components using Plotly."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Optional

from src.api.hyperliquid import PortfolioBreakdown
from src.utils.formatters import format_currency, format_percentage


# Color palette matching crypto dashboard theme
COLORS = {
    "perp": "#3bb5d3",      # Primary cyan
    "spot": "#7dd3fc",      # Light blue
    "background": "#1a2845", # Dark blue background
    "text": "#e2e8f0",       # Light text
    "grid": "#3a4556",       # Muted grid lines
    "positive": "#22c55e",   # Green for positive PnL
    "negative": "#e74c3c",   # Red for negative PnL
}


def create_portfolio_stacked_bar(
    breakdown: PortfolioBreakdown,
    height: int = 300,
    show_percentages: bool = True
) -> go.Figure:
    """
    Create a stacked horizontal bar chart for portfolio breakdown.

    Args:
        breakdown: PortfolioBreakdown with total, perp, and spot metrics
        height: Chart height in pixels
        show_percentages: Whether to show percentage labels

    Returns:
        Plotly Figure object
    """
    # Prepare data
    metrics = ["Account Value", "PnL", "Volume"]
    perp_values = [
        breakdown.perp.account_value,
        breakdown.perp.pnl,
        breakdown.perp.volume
    ]
    spot_values = [
        breakdown.spot.account_value,
        breakdown.spot.pnl,
        breakdown.spot.volume
    ]
    total_values = [
        breakdown.total.account_value,
        breakdown.total.pnl,
        breakdown.total.volume
    ]

    # Create hover text with formatted values
    perp_hover = [
        f"<b>Perp</b><br>{format_currency(v)}<br>{format_percentage(v, t)}"
        for v, t in zip(perp_values, total_values)
    ]
    spot_hover = [
        f"<b>Spot</b><br>{format_currency(v)}<br>{format_percentage(v, t)}"
        for v, t in zip(spot_values, total_values)
    ]

    # Create figure
    fig = go.Figure()

    # Add Perp bars
    fig.add_trace(go.Bar(
        name="Perp",
        y=metrics,
        x=perp_values,
        orientation="h",
        marker=dict(
            color=COLORS["perp"],
            line=dict(color=COLORS["perp"], width=1)
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=perp_hover,
        text=[format_percentage(v, t) for v, t in zip(perp_values, total_values)] if show_percentages else None,
        textposition="inside",
        textfont=dict(color="white", size=12, family="Inter"),
    ))

    # Add Spot bars
    fig.add_trace(go.Bar(
        name="Spot",
        y=metrics,
        x=spot_values,
        orientation="h",
        marker=dict(
            color=COLORS["spot"],
            line=dict(color=COLORS["spot"], width=1)
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=spot_hover,
        text=[format_percentage(v, t) for v, t in zip(spot_values, total_values)] if show_percentages else None,
        textposition="inside",
        textfont=dict(color="#1a2845", size=12, family="Inter"),
    ))

    # Update layout
    fig.update_layout(
        barmode="stack",
        height=height,
        margin=dict(l=120, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            size=14,
            color=COLORS["text"]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            gridwidth=1,
            zeroline=False,
            showline=False,
            tickformat="$,.0f",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            tickfont=dict(size=13, color=COLORS["text"]),
            autorange="reversed",  # Account Value at top
        ),
        hoverlabel=dict(
            bgcolor=COLORS["background"],
            font_size=13,
            font_family="Inter",
            bordercolor=COLORS["perp"]
        ),
    )

    return fig


def create_portfolio_metrics_cards(breakdown: PortfolioBreakdown) -> dict:
    """
    Create metric card data from breakdown.

    Returns:
        Dictionary with formatted metrics for display
    """
    return {
        "total_value": {
            "label": "Total Account Value",
            "value": format_currency(breakdown.total.account_value),
            "raw": breakdown.total.account_value
        },
        "total_pnl": {
            "label": "Total PnL",
            "value": format_currency(breakdown.total.pnl),
            "raw": breakdown.total.pnl,
            "is_positive": breakdown.total.pnl >= 0
        },
        "total_volume": {
            "label": "Total Volume",
            "value": format_currency(breakdown.total.volume),
            "raw": breakdown.total.volume
        },
        "perp_percentage": {
            "label": "Perp Allocation",
            "value": format_percentage(breakdown.perp.account_value, breakdown.total.account_value),
        },
        "spot_percentage": {
            "label": "Spot Allocation",
            "value": format_percentage(breakdown.spot.account_value, breakdown.total.account_value),
        }
    }
