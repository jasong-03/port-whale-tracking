"""
Hyperliquid Portfolio Dashboard

A Streamlit app to visualize Hyperliquid portfolio data with
stacked horizontal bar charts and distribution heatmaps.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
from datetime import datetime, timedelta
from src.api.hyperliquid import HyperliquidClient, get_mock_portfolio_breakdown, TradeFill
from src.utils.formatters import format_currency

# Color palette
COLORS = {
    "perp": "#3bb5d3",
    "spot": "#7dd3fc",
    "background": "#1a2845",
    "text": "#e2e8f0",
    "grid": "#3a4556",
}

# Heatmap color scale
HEATMAP_COLORSCALE = [
    [0, "#1a2845"],
    [0.25, "#1e3a5f"],
    [0.5, "#3bb5d3"],
    [0.75, "#7dd3fc"],
    [1, "#e2e8f0"]
]


@st.cache_data
def load_wallet_addresses():
    """Load wallet addresses from CSV file."""
    csv_path = Path(__file__).parent / "wallet_address.txt"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df
    return None


def create_screening_chart(df: pd.DataFrame, metric: str = "value", height: int = 800, mode: str = "value"):
    """Create a stacked horizontal bar chart for all wallets.

    Args:
        df: DataFrame with wallet data
        metric: "value", "pnl", or "volume"
        height: Chart height in pixels
        mode: "value" for absolute values, "percentage" for 100% stacked
    """

    if metric == "value":
        perp_col, spot_col, label = "perp_value", "spot_value", "Account Value"
        total_col = "total_value"
    elif metric == "pnl":
        perp_col, spot_col, label = "perp_pnl", "spot_pnl", "PnL"
        total_col = "total_pnl"
    else:
        perp_col, spot_col, label = "perp_volume", "spot_volume", "Volume"
        total_col = "total_volume"

    fig = go.Figure()

    if mode == "percentage":
        # Calculate percentages
        df = df.copy()
        total = df[perp_col].abs() + df[spot_col].abs()
        total = total.replace(0, 1)  # Avoid division by zero
        perp_pct = (df[perp_col].abs() / total * 100).round(1)
        spot_pct = (df[spot_col].abs() / total * 100).round(1)

        # Store original values for hover
        perp_values = df[perp_col]
        spot_values = df[spot_col]

        # Create custom hover text
        perp_hover = [f"<b>{name}</b><br>Perp: {pct:.1f}% ({format_currency(val)})"
                      for name, pct, val in zip(df["display_name"], perp_pct, perp_values)]
        spot_hover = [f"<b>{name}</b><br>Spot: {pct:.1f}% ({format_currency(val)})"
                      for name, pct, val in zip(df["display_name"], spot_pct, spot_values)]

        # Add Perp bars (percentage)
        fig.add_trace(go.Bar(
            name="Perp",
            y=df["display_name"],
            x=perp_pct,
            orientation="h",
            marker=dict(color=COLORS["perp"]),
            hoverinfo="text",
            hovertext=perp_hover,
        ))

        # Add Spot bars (percentage)
        fig.add_trace(go.Bar(
            name="Spot",
            y=df["display_name"],
            x=spot_pct,
            orientation="h",
            marker=dict(color=COLORS["spot"]),
            hoverinfo="text",
            hovertext=spot_hover,
        ))

        title_text = f"Portfolio Allocation (Perp vs Spot %)"
        x_tickformat = ".0f"
        x_ticksuffix = "%"
    else:
        # Original absolute value mode
        # Add Perp bars
        fig.add_trace(go.Bar(
            name="Perp",
            y=df["display_name"],
            x=df[perp_col],
            orientation="h",
            marker=dict(color=COLORS["perp"]),
            hovertemplate="<b>%{y}</b><br>Perp: %{x:$,.0f}<extra></extra>",
        ))

        # Add Spot bars
        fig.add_trace(go.Bar(
            name="Spot",
            y=df["display_name"],
            x=df[spot_col],
            orientation="h",
            marker=dict(color=COLORS["spot"]),
            hovertemplate="<b>%{y}</b><br>Spot: %{x:$,.0f}<extra></extra>",
        ))

        title_text = f"Portfolio Breakdown by {label}"
        x_tickformat = "$,.0f"
        x_ticksuffix = ""

    fig.update_layout(
        barmode="stack",
        height=height,
        margin=dict(l=250, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=11, color=COLORS["text"]),
        title=dict(
            text=title_text,
            font=dict(size=18, color=COLORS["text"]),
            x=0.5
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
            tickformat=x_tickformat,
            ticksuffix=x_ticksuffix,
            range=[0, 100] if mode == "percentage" else None,
        ),
        yaxis=dict(
            showgrid=False,
            autorange="reversed",
            tickfont=dict(size=10),
        ),
        hoverlabel=dict(
            bgcolor=COLORS["background"],
            font_size=12,
            bordercolor=COLORS["perp"]
        ),
    )

    return fig


def create_value_perp_heatmap(df: pd.DataFrame):
    """Create heatmap: Account Value (X) vs Perp % (Y)"""

    # Define bins
    value_bins = [0, 1e6, 5e6, 10e6, 50e6, 100e6, float('inf')]
    value_labels = ['<$1M', '$1-5M', '$5-10M', '$10-50M', '$50-100M', '>$100M']

    perp_bins = [0, 20, 40, 60, 80, 100.1]
    perp_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

    # Bin the data
    df['value_bin'] = pd.cut(df['total_value'], bins=value_bins, labels=value_labels)
    df['perp_bin'] = pd.cut(df['perp_pct'], bins=perp_bins, labels=perp_labels)

    # Create pivot table (count of wallets)
    heatmap_data = df.groupby(['perp_bin', 'value_bin'], observed=True).size().unstack(fill_value=0)

    # Reindex to ensure all bins are present
    heatmap_data = heatmap_data.reindex(index=perp_labels, columns=value_labels, fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=value_labels,
        y=perp_labels,
        colorscale=HEATMAP_COLORSCALE,
        hovertemplate="Value: %{x}<br>Perp %: %{y}<br>Wallets: %{z}<extra></extra>",
        text=heatmap_data.values,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
    ))

    fig.update_layout(
        title=dict(
            text="Distribution: Account Value vs Perp Allocation",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5
        ),
        xaxis_title="Account Value",
        yaxis_title="Perp Allocation %",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
    )

    return fig


def create_entity_perp_heatmap(df: pd.DataFrame):
    """Create heatmap: Entity Type (X) vs Perp % (Y), color = Total AUM"""

    perp_bins = [0, 20, 40, 60, 80, 100.1]
    perp_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

    df['perp_bin'] = pd.cut(df['perp_pct'], bins=perp_bins, labels=perp_labels)

    # Group by entity and perp_bin, sum total_value
    heatmap_data = df.groupby(['perp_bin', 'entity'], observed=True)['total_value'].sum().unstack(fill_value=0)

    # Reindex
    entities = df['entity'].unique().tolist()
    heatmap_data = heatmap_data.reindex(index=perp_labels, columns=sorted(entities), fill_value=0)

    # Format values for display (in millions)
    display_text = (heatmap_data / 1e6).round(1).astype(str) + 'M'
    display_text = display_text.replace('0.0M', '-')

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=sorted(entities),
        y=perp_labels,
        colorscale=HEATMAP_COLORSCALE,
        hovertemplate="Entity: %{x}<br>Perp %: %{y}<br>AUM: $%{z:,.0f}<extra></extra>",
        text=display_text.values,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
    ))

    fig.update_layout(
        title=dict(
            text="Distribution: Entity Type vs Perp Allocation (AUM)",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5
        ),
        xaxis_title="Entity Type",
        yaxis_title="Perp Allocation %",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
    )

    return fig


def create_value_pnl_heatmap(df: pd.DataFrame):
    """Create heatmap: Account Value (X) vs PnL (Y)"""

    value_bins = [0, 1e6, 5e6, 10e6, 50e6, 100e6, float('inf')]
    value_labels = ['<$1M', '$1-5M', '$5-10M', '$10-50M', '$50-100M', '>$100M']

    pnl_bins = [-float('inf'), -1e6, -100e3, 0, 100e3, 1e6, 10e6, float('inf')]
    pnl_labels = ['<-$1M', '-$1M to -$100K', '-$100K to $0', '$0 to $100K', '$100K to $1M', '$1M to $10M', '>$10M']

    df['value_bin'] = pd.cut(df['total_value'], bins=value_bins, labels=value_labels)
    df['pnl_bin'] = pd.cut(df['total_pnl'], bins=pnl_bins, labels=pnl_labels)

    # Create pivot table
    heatmap_data = df.groupby(['pnl_bin', 'value_bin'], observed=True).size().unstack(fill_value=0)

    # Reindex
    heatmap_data = heatmap_data.reindex(index=pnl_labels, columns=value_labels, fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=value_labels,
        y=pnl_labels,
        colorscale=HEATMAP_COLORSCALE,
        hovertemplate="Value: %{x}<br>PnL: %{y}<br>Wallets: %{z}<extra></extra>",
        text=heatmap_data.values,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
    ))

    fig.update_layout(
        title=dict(
            text="Distribution: Account Value vs PnL",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5
        ),
        xaxis_title="Account Value",
        yaxis_title="PnL Range",
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )

    return fig


def create_histogram(df: pd.DataFrame, column: str, title: str, bins: int = 20):
    """Create a histogram for distribution analysis."""

    fig = go.Figure(data=go.Histogram(
        x=df[column],
        nbinsx=bins,
        marker=dict(color=COLORS["perp"], line=dict(color=COLORS["spot"], width=1)),
        hovertemplate=f"{title}: %{{x}}<br>Count: %{{y}}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Distribution of {title}", font=dict(size=16, color=COLORS["text"]), x=0.5),
        xaxis_title=title,
        yaxis_title="Number of Wallets",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        xaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        yaxis=dict(showgrid=True, gridcolor=COLORS["grid"]),
        bargap=0.1,
    )

    return fig


def create_activity_calendar(fills_df: pd.DataFrame, year: int = None):
    """
    Create GitHub-style activity calendar heatmap.

    Colors:
    - Green = Long positions (Open Long, Close Short)
    - Red = Short positions (Open Short, Close Long)
    - Intensity based on trade count
    """
    if year is None:
        year = datetime.now().year

    # Create date range for the year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Calculate number of weeks (53 to cover full year)
    num_weeks = 53
    num_days = 7

    # Initialize matrices directly with numpy
    long_matrix = np.zeros((num_days, num_weeks))
    short_matrix = np.zeros((num_days, num_weeks))

    # Create date to (day_of_week, week_of_year) mapping
    date_to_coords = {}
    for d in all_dates:
        day_of_week = d.dayofweek  # 0=Monday, 6=Sunday
        week_of_year = (d - start_date).days // 7
        if week_of_year < num_weeks:
            date_to_coords[d.date()] = (day_of_week, week_of_year)

    # Fill matrices from fills data
    if len(fills_df) > 0:
        fills_df = fills_df.copy()
        fills_df['date'] = pd.to_datetime(fills_df['timestamp']).dt.date

        # Categorize trades
        long_directions = ['Open Long', 'Close Short', 'Buy']
        short_directions = ['Open Short', 'Close Long', 'Sell']

        for _, row in fills_df.iterrows():
            trade_date = row['date']
            if trade_date in date_to_coords:
                day_idx, week_idx = date_to_coords[trade_date]
                direction = row['direction']
                if direction in long_directions:
                    long_matrix[day_idx, week_idx] += 1
                elif direction in short_directions:
                    short_matrix[day_idx, week_idx] += 1

    # Create custom colorscale
    activity_colorscale = [
        [0, "#ef4444"],      # Bright red (short heavy)
        [0.20, "#f87171"],   # Red
        [0.40, "#fca5a5"],   # Light red
        [0.48, "#0f172a"],   # Dark (no activity)
        [0.52, "#0f172a"],   # Dark (no activity)
        [0.60, "#86efac"],   # Light green
        [0.80, "#4ade80"],   # Green
        [1, "#22c55e"]       # Bright green (long heavy)
    ]

    # Create display values based on rules
    display_values = np.full((num_days, num_weeks), 0.5)  # Default to dark (no activity)

    # Get max values for normalization
    max_long = max(long_matrix.max(), 1)
    max_short = max(short_matrix.max(), 1)

    for i in range(num_days):
        for j in range(num_weeks):
            long_val = long_matrix[i, j]
            short_val = short_matrix[i, j]
            total_val = long_val + short_val

            if total_val == 0:
                # Rule 1: No activity -> dark (0.5)
                display_values[i, j] = 0.5
            elif long_val > 0 and short_val == 0:
                # Rule 2: Long only -> green (0.52 to 1.0)
                intensity = min(long_val / max_long, 1.0)
                display_values[i, j] = 0.52 + (intensity * 0.48)
            elif short_val > 0 and long_val == 0:
                # Rule 3: Short only -> red (0 to 0.48)
                intensity = min(short_val / max_short, 1.0)
                display_values[i, j] = 0.48 - (intensity * 0.48)
            else:
                # Rule 4: Both -> based on ratio
                ratio = long_val / total_val
                if ratio > 0.5:
                    display_values[i, j] = 0.52 + ((ratio - 0.5) * 0.96)
                else:
                    display_values[i, j] = 0.48 - ((0.5 - ratio) * 0.96)

    # Create hover text
    hover_text = []
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day_idx in range(num_days):
        row_text = []
        for week_idx in range(num_weeks):
            try:
                date = start_date + timedelta(weeks=week_idx, days=day_idx)
                if date.year == year:
                    longs = int(long_matrix[day_idx, week_idx])
                    shorts = int(short_matrix[day_idx, week_idx])
                    total = longs + shorts
                    if total > 0:
                        row_text.append(f"{date.strftime('%Y-%m-%d')}<br>Longs: {longs}<br>Shorts: {shorts}")
                    else:
                        row_text.append(f"{date.strftime('%Y-%m-%d')}<br>No activity")
                else:
                    row_text.append("")
            except:
                row_text.append("")
        hover_text.append(row_text)

    # Create month labels for x-axis
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_positions = []
    for m in range(1, 13):
        first_day = datetime(year, m, 1)
        week_pos = (first_day - start_date).days // 7
        month_positions.append(week_pos)

    fig = go.Figure(data=go.Heatmap(
        z=display_values,
        x=list(range(num_weeks)),
        y=day_names,
        colorscale=activity_colorscale,
        showscale=False,
        hoverinfo='text',
        text=hover_text,
        xgap=2,
        ygap=2,
    ))

    fig.update_layout(
        title=dict(
            text=f"Trading Activity Calendar {year}",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5
        ),
        height=250,
        margin=dict(l=50, r=20, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        xaxis=dict(
            tickmode='array',
            tickvals=month_positions,
            ticktext=month_labels,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            autorange='reversed',
        ),
    )

    return fig


def create_activity_legend():
    """Create legend for activity calendar."""
    return """
    <div style="display: flex; align-items: center; gap: 20px; margin: 10px 0;">
        <span style="color: #94a3b8;">Less</span>
        <div style="display: flex; gap: 3px;">
            <div style="width: 15px; height: 15px; background: #dc2626; border-radius: 2px;" title="Short heavy"></div>
            <div style="width: 15px; height: 15px; background: #fca5a5; border-radius: 2px;" title="Some shorts"></div>
            <div style="width: 15px; height: 15px; background: #1a1a1a; border-radius: 2px; border: 1px solid #3a4556;" title="No activity"></div>
            <div style="width: 15px; height: 15px; background: #86efac; border-radius: 2px;" title="Some longs"></div>
            <div style="width: 15px; height: 15px; background: #22c55e; border-radius: 2px;" title="Long heavy"></div>
        </div>
        <span style="color: #94a3b8;">More</span>
        <span style="margin-left: 20px; color: #22c55e;">■ Long</span>
        <span style="color: #dc2626;">■ Short</span>
        <span style="margin-left: 10px; color: #1a1a1a; text-shadow: 0 0 1px #94a3b8;">■ No activity</span>
    </div>
    """


def create_activity_calendar_range(fills_df: pd.DataFrame, from_year: int, to_year: int):
    """
    Create a single combined activity calendar heatmap for a date range.
    Shows all years in one continuous heatmap with 4 trade types.
    """
    start_date = datetime(from_year, 1, 1)
    end_date = datetime(to_year, 12, 31)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Calculate total weeks across all years
    total_days = (end_date - start_date).days + 1
    num_weeks = (total_days // 7) + 2
    num_days = 7

    # Initialize 4 matrices for each trade type
    open_long_matrix = np.zeros((num_days, num_weeks))
    close_long_matrix = np.zeros((num_days, num_weeks))
    open_short_matrix = np.zeros((num_days, num_weeks))
    close_short_matrix = np.zeros((num_days, num_weeks))

    # Create date to coordinates mapping
    date_to_coords = {}
    for d in all_dates:
        day_of_week = d.dayofweek
        week_of_range = (d - start_date).days // 7
        if week_of_range < num_weeks:
            date_to_coords[d.date()] = (day_of_week, week_of_range)

    # Fill matrices from fills data
    if len(fills_df) > 0:
        fills_df = fills_df.copy()
        fills_df['date'] = pd.to_datetime(fills_df['timestamp']).dt.date

        for _, row in fills_df.iterrows():
            trade_date = row['date']
            if trade_date in date_to_coords:
                day_idx, week_idx = date_to_coords[trade_date]
                direction = row['direction']
                if direction == 'Open Long':
                    open_long_matrix[day_idx, week_idx] += 1
                elif direction == 'Close Long':
                    close_long_matrix[day_idx, week_idx] += 1
                elif direction == 'Open Short':
                    open_short_matrix[day_idx, week_idx] += 1
                elif direction == 'Close Short':
                    close_short_matrix[day_idx, week_idx] += 1

    # Aggregate for coloring: Long = Open Long + Close Short, Short = Open Short + Close Long
    long_matrix = open_long_matrix + close_short_matrix
    short_matrix = open_short_matrix + close_long_matrix

    # Color scale
    activity_colorscale = [
        [0, "#ef4444"], [0.20, "#f87171"], [0.40, "#fca5a5"],
        [0.48, "#0f172a"], [0.52, "#0f172a"],
        [0.60, "#86efac"], [0.80, "#4ade80"], [1, "#22c55e"]
    ]

    # Create display values
    display_values = np.full((num_days, num_weeks), 0.5)
    max_long = max(long_matrix.max(), 1)
    max_short = max(short_matrix.max(), 1)

    for i in range(num_days):
        for j in range(num_weeks):
            long_val = long_matrix[i, j]
            short_val = short_matrix[i, j]
            total_val = long_val + short_val

            if total_val == 0:
                display_values[i, j] = 0.5
            elif long_val > 0 and short_val == 0:
                intensity = min(long_val / max_long, 1.0)
                display_values[i, j] = 0.52 + (intensity * 0.48)
            elif short_val > 0 and long_val == 0:
                intensity = min(short_val / max_short, 1.0)
                display_values[i, j] = 0.48 - (intensity * 0.48)
            else:
                ratio = long_val / total_val
                if ratio > 0.5:
                    display_values[i, j] = 0.52 + ((ratio - 0.5) * 0.96)
                else:
                    display_values[i, j] = 0.48 - ((0.5 - ratio) * 0.96)

    # Create hover text - show only dominant activity type per day
    hover_text = []
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day_idx in range(num_days):
        row_text = []
        for week_idx in range(num_weeks):
            try:
                date = start_date + timedelta(weeks=week_idx, days=day_idx)
                if start_date <= date <= end_date:
                    ol = int(open_long_matrix[day_idx, week_idx])
                    cl = int(close_long_matrix[day_idx, week_idx])
                    os = int(open_short_matrix[day_idx, week_idx])
                    cs = int(close_short_matrix[day_idx, week_idx])
                    total = ol + cl + os + cs
                    if total > 0:
                        # Find dominant activity type
                        activities = {
                            'Open Long': (ol, '🟢'),
                            'Close Long': (cl, '🔵'),
                            'Open Short': (os, '🔴'),
                            'Close Short': (cs, '🟠')
                        }
                        dominant_type = max(activities.items(), key=lambda x: x[1][0])
                        dominant_name = dominant_type[0]
                        dominant_count = dominant_type[1][0]
                        dominant_emoji = dominant_type[1][1]

                        row_text.append(
                            f"<b>{date.strftime('%Y-%m-%d')}</b><br>"
                            f"{dominant_emoji} {dominant_name}: {dominant_count}<br>"
                            f"<b>Total: {total}</b>"
                        )
                    else:
                        row_text.append(f"{date.strftime('%Y-%m-%d')}<br>No activity")
                else:
                    row_text.append("")
            except:
                row_text.append("")
        hover_text.append(row_text)

    # Create month/year labels for x-axis
    month_labels = []
    month_positions = []
    for year in range(from_year, to_year + 1):
        for m in range(1, 13):
            try:
                first_day = datetime(year, m, 1)
                if start_date <= first_day <= end_date:
                    week_pos = (first_day - start_date).days // 7
                    label = f"{first_day.strftime('%b')}" if from_year == to_year else f"{first_day.strftime('%b %Y')}"
                    month_labels.append(label)
                    month_positions.append(week_pos)
            except:
                pass

    fig = go.Figure(data=go.Heatmap(
        z=display_values,
        x=list(range(num_weeks)),
        y=day_names,
        colorscale=activity_colorscale,
        showscale=False,
        hoverinfo='text',
        text=hover_text,
        xgap=2,
        ygap=2,
    ))

    title_text = f"Trading Activity Calendar {from_year}" if from_year == to_year else f"Trading Activity Calendar {from_year} - {to_year}"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=18, color=COLORS["text"]), x=0.5),
        height=250,
        margin=dict(l=50, r=20, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        xaxis=dict(
            tickmode='array',
            tickvals=month_positions,
            ticktext=month_labels,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(showgrid=False, zeroline=False, autorange='reversed'),
    )

    return fig


def create_all_wallets_heatmap(fills_df: pd.DataFrame, from_year: int = None, to_year: int = None, all_wallet_names: list = None):
    """
    Create a beautiful combined heatmap showing all wallets' activity for a date range.
    Y-axis: Wallet names
    X-axis: Days across the entire range
    Color: Green (long), Red (short), Dark (no activity)
    Hover: Shows dominant trade type per day
    """
    if from_year is None:
        from_year = datetime.now().year
    if to_year is None:
        to_year = from_year

    # Create date range
    start_date = datetime(from_year, 1, 1)
    end_date = datetime(to_year, 12, 31)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Get wallets list - use provided list or extract from fills_df
    if all_wallet_names is not None:
        # Use all wallets from the provided list, sorted by activity (active first)
        fills_df = fills_df.copy() if len(fills_df) > 0 else pd.DataFrame()
        if len(fills_df) > 0:
            fills_df['date'] = pd.to_datetime(fills_df['timestamp']).dt.date
            wallet_counts = fills_df.groupby('wallet').size()
            # Sort: wallets with activity first (by count desc), then wallets without activity
            wallets_with_activity = wallet_counts.sort_values(ascending=False).index.tolist()
            wallets_without_activity = [w for w in all_wallet_names if w not in wallets_with_activity]
            wallets = wallets_with_activity + wallets_without_activity
        else:
            wallets = all_wallet_names
    else:
        if len(fills_df) == 0 or 'wallet' not in fills_df.columns:
            return None
        fills_df = fills_df.copy()
        fills_df['date'] = pd.to_datetime(fills_df['timestamp']).dt.date
        wallet_counts = fills_df.groupby('wallet').size().sort_values(ascending=False)
        wallets = wallet_counts.index.tolist()

    # Create matrix: wallets x days
    num_days_total = len(all_dates)
    num_wallets = len(wallets)

    # Initialize 4 matrices for each trade type
    open_long_matrix = np.zeros((num_wallets, num_days_total))
    close_long_matrix = np.zeros((num_wallets, num_days_total))
    open_short_matrix = np.zeros((num_wallets, num_days_total))
    close_short_matrix = np.zeros((num_wallets, num_days_total))

    # Create date to index mapping
    date_to_idx = {d.date(): i for i, d in enumerate(all_dates)}

    # Fill matrices
    for wallet_idx, wallet in enumerate(wallets):
        wallet_fills = fills_df[fills_df['wallet'] == wallet]

        for _, row in wallet_fills.iterrows():
            trade_date = row['date']
            if trade_date in date_to_idx:
                day_idx = date_to_idx[trade_date]
                direction = row['direction']
                if direction == 'Open Long':
                    open_long_matrix[wallet_idx, day_idx] += 1
                elif direction == 'Close Long':
                    close_long_matrix[wallet_idx, day_idx] += 1
                elif direction == 'Open Short':
                    open_short_matrix[wallet_idx, day_idx] += 1
                elif direction == 'Close Short':
                    close_short_matrix[wallet_idx, day_idx] += 1

    # Aggregate for coloring
    long_matrix = open_long_matrix + close_short_matrix
    short_matrix = open_short_matrix + close_long_matrix

    # Enhanced color scale
    activity_colorscale = [
        [0, "#ef4444"], [0.15, "#f87171"], [0.30, "#fca5a5"],
        [0.45, "#0f172a"], [0.55, "#0f172a"],
        [0.70, "#86efac"], [0.85, "#4ade80"], [1, "#22c55e"]
    ]

    # Create display values based on rules
    display_values = np.full((num_wallets, num_days_total), 0.5)

    max_long = max(long_matrix.max(), 1)
    max_short = max(short_matrix.max(), 1)

    for i in range(num_wallets):
        for j in range(num_days_total):
            long_val = long_matrix[i, j]
            short_val = short_matrix[i, j]
            total_val = long_val + short_val

            if total_val == 0:
                display_values[i, j] = 0.5
            elif long_val > 0 and short_val == 0:
                intensity = min(long_val / max_long, 1.0)
                display_values[i, j] = 0.55 + (intensity * 0.45)
            elif short_val > 0 and long_val == 0:
                intensity = min(short_val / max_short, 1.0)
                display_values[i, j] = 0.45 - (intensity * 0.45)
            else:
                ratio = long_val / total_val
                if ratio > 0.5:
                    display_values[i, j] = 0.55 + ((ratio - 0.5) * 0.9)
                else:
                    display_values[i, j] = 0.45 - ((0.5 - ratio) * 0.9)

    # Create hover text - show only dominant activity type per day
    hover_text = []
    for wallet_idx, wallet in enumerate(wallets):
        row_text = []
        for day_idx, date in enumerate(all_dates):
            ol = int(open_long_matrix[wallet_idx, day_idx])
            cl = int(close_long_matrix[wallet_idx, day_idx])
            os = int(open_short_matrix[wallet_idx, day_idx])
            cs = int(close_short_matrix[wallet_idx, day_idx])
            total = ol + cl + os + cs
            if total > 0:
                # Find dominant activity type
                activities = {
                    'Open Long': (ol, '🟢'),
                    'Close Long': (cl, '🔵'),
                    'Open Short': (os, '🔴'),
                    'Close Short': (cs, '🟠')
                }
                dominant_type = max(activities.items(), key=lambda x: x[1][0])
                dominant_name = dominant_type[0]
                dominant_count = dominant_type[1][0]
                dominant_emoji = dominant_type[1][1]

                row_text.append(
                    f"<b>{wallet[:25]}</b><br>"
                    f"<b>{date.strftime('%b %d, %Y')}</b><br>"
                    f"{dominant_emoji} {dominant_name}: {dominant_count}<br>"
                    f"<b>Total: {total}</b>"
                )
            else:
                row_text.append(
                    f"<b>{wallet[:25]}</b><br>"
                    f"<b>{date.strftime('%b %d, %Y')}</b><br>"
                    f"<span style='color:#64748b'>No activity</span>"
                )
        hover_text.append(row_text)

    # Better wallet label formatting
    wallet_labels = [w[:32] + "..." if len(w) > 35 else w for w in wallets]

    # Create month tick positions for the entire range
    month_labels = []
    month_positions = []
    for year in range(from_year, to_year + 1):
        for m in range(1, 13):
            try:
                first_day = datetime(year, m, 1)
                if start_date <= first_day <= end_date:
                    day_idx = (first_day - start_date).days
                    label = f"{first_day.strftime('%b')}" if from_year == to_year else f"{first_day.strftime('%b %y')}"
                    month_labels.append(label)
                    month_positions.append(day_idx + 15)
            except:
                pass

    fig = go.Figure(data=go.Heatmap(
        z=display_values,
        x=list(range(num_days_total)),
        y=wallet_labels,
        colorscale=activity_colorscale,
        showscale=False,
        hoverinfo='text',
        text=hover_text,
        xgap=1,
        ygap=3,
        hoverongaps=False,
    ))

    # Calculate dynamic height
    row_height = 28
    chart_height = max(500, num_wallets * row_height + 120)

    title_text = f"<b>📊 Trading Activity Heatmap - {from_year}</b>" if from_year == to_year else f"<b>📊 Trading Activity Heatmap - {from_year} to {to_year}</b>"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20, color="#f1f5f9", family="Inter, sans-serif"), x=0.5, y=0.98),
        height=chart_height,
        margin=dict(l=280, r=30, t=80, b=60),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        xaxis=dict(
            tickmode='array',
            tickvals=month_positions,
            ticktext=month_labels,
            showgrid=True,
            gridcolor="rgba(51, 65, 85, 0.3)",
            gridwidth=1,
            zeroline=False,
            side='top',
            tickfont=dict(size=10, color="#94a3b8"),
            tickangle=45 if from_year != to_year else 0,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(51, 65, 85, 0.2)",
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=11, color="#cbd5e1"),
            ticklabelposition="outside left",
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            font=dict(size=13, color="#f1f5f9", family="Inter, sans-serif"),
        ),
    )

    # Add month separator lines
    for year in range(from_year, to_year + 1):
        for m in range(1, 13):
            try:
                first_day = datetime(year, m, 1)
                if start_date <= first_day <= end_date:
                    day_idx = (first_day - start_date).days
                    fig.add_vline(
                        x=day_idx - 0.5,
                        line=dict(color="rgba(71, 85, 105, 0.4)", width=1, dash="dot"),
                    )
            except:
                pass

    return fig


# Page config
st.set_page_config(
    page_title="Hyperliquid Whale Screener",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #1a2845; }
    [data-testid="stSidebar"] { background-color: #1e2c42; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    [data-testid="metric-container"] {
        background-color: #1e2c42;
        border: 1px solid #3a4556;
        border-radius: 8px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
    .stButton > button {
        background-color: #3bb5d3;
        color: #1a2845;
        border: none;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #7dd3fc; }
    hr { border-color: #3a4556; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2c42;
        border-radius: 8px;
        color: #e2e8f0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3bb5d3;
        color: #1a2845;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Load wallet addresses
    wallets_df = load_wallet_addresses()

    if wallets_df is None:
        st.error("❌ Could not load wallet addresses from wallet_address.txt")
        return

    # Sidebar
    with st.sidebar:
        st.title("🐋 Whale Screener")
        st.caption("Screen all Hyperliquid whale portfolios")

        st.divider()

        # Filters
        st.subheader("Filters")

        # Entity filter
        entities = ["All"] + sorted(wallets_df["Entity"].unique().tolist())
        selected_entity = st.selectbox("Entity Type", entities)

        # Number of wallets to show
        max_wallets = st.slider(
            "Number of Wallets",
            min_value=10,
            max_value=len(wallets_df),
            value=len(wallets_df),
            step=10
        )

        # Sort by
        sort_by = st.selectbox(
            "Sort By",
            ["Account Value (CSV)", "Perp %", "Total Value (Live)", "PnL"],
            index=0
        )

        st.divider()

        # Time period
        time_period = st.selectbox(
            "Time Period",
            options=["day", "week", "month", "allTime"],
            format_func=lambda x: {
                "day": "📅 Day",
                "week": "📆 Week",
                "month": "🗓️ Month",
                "allTime": "⏳ All Time"
            }.get(x, x)
        )

        st.divider()

        # Fetch button
        fetch_data = st.button("🔄 Fetch Live Data", type="primary", use_container_width=True)

        st.divider()
        st.info("💡 Click 'Fetch Live Data' to get real-time data from Hyperliquid API")

    # Main content
    st.title("🐋 Hyperliquid Whale Screener")
    st.caption("View Perp vs Spot allocation and distribution maps across all whale wallets")

    # Filter wallets
    filtered_df = wallets_df.copy()
    if selected_entity != "All":
        filtered_df = filtered_df[filtered_df["Entity"] == selected_entity]

    filtered_df = filtered_df.head(max_wallets)
    filtered_df["display_name"] = filtered_df["trader_address_label"].str[:40]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Wallets", len(filtered_df))
    with col2:
        total_value = filtered_df["account_value"].str.replace(",", "").astype(float).sum()
        st.metric("Total AUM", format_currency(total_value))
    with col3:
        st.metric("VCs", len(filtered_df[filtered_df["Entity"] == "VCs"]))
    with col4:
        st.metric("Retail", len(filtered_df[filtered_df["Entity"] == "retail"]))

    st.divider()

    # Fetch data if button clicked
    if fetch_data:
        with st.spinner(f"Fetching portfolio data for {len(filtered_df)} wallets..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            client = HyperliquidClient()
            results = []

            for i, (_, row) in enumerate(filtered_df.iterrows()):
                addr = row["trader_address"]
                status_text.text(f"Fetching {row['trader_address_label'][:30]}...")

                try:
                    breakdown = client.get_portfolio_breakdown(addr, time_period)
                    if breakdown and breakdown.total.account_value > 0:
                        results.append({
                            "address": addr,
                            "display_name": row["trader_address_label"][:40],
                            "entity": row["Entity"],
                            "total_value": breakdown.total.account_value,
                            "perp_value": breakdown.perp.account_value,
                            "spot_value": breakdown.spot.account_value,
                            "perp_pct": (breakdown.perp.account_value / breakdown.total.account_value * 100) if breakdown.total.account_value > 0 else 0,
                            "total_pnl": breakdown.total.pnl,
                            "perp_pnl": breakdown.perp.pnl,
                            "spot_pnl": breakdown.spot.pnl,
                            "total_volume": breakdown.total.volume,
                            "perp_volume": breakdown.perp.volume,
                            "spot_volume": breakdown.spot.volume,
                        })
                except Exception:
                    pass

                progress_bar.progress((i + 1) / len(filtered_df))
                time.sleep(0.05)

            progress_bar.empty()
            status_text.empty()

            if results:
                st.session_state.portfolio_data = pd.DataFrame(results)
                st.success(f"✅ Fetched data for {len(results)} wallets")
            else:
                st.error("❌ No data fetched. Try again.")

    # Display data with tabs
    if "portfolio_data" in st.session_state and len(st.session_state.portfolio_data) > 0:
        portfolio_df = st.session_state.portfolio_data.copy()

        # Sort
        if sort_by == "Perp %":
            portfolio_df = portfolio_df.sort_values("perp_pct", ascending=True)
        elif sort_by == "Total Value (Live)":
            portfolio_df = portfolio_df.sort_values("total_value", ascending=True)
        elif sort_by == "PnL":
            portfolio_df = portfolio_df.sort_values("total_pnl", ascending=True)

        # ==================== SECTION 1: Portfolio Breakdown ====================
        st.subheader("📊 Portfolio Breakdown")
        col_title, col_toggle = st.columns([3, 1])
        with col_toggle:
            chart_mode = st.toggle("Show as %", value=True, help="Toggle between absolute values and percentage allocation")

        chart_height = max(400, len(portfolio_df) * 25)
        display_mode = "percentage" if chart_mode else "value"
        fig = create_screening_chart(portfolio_df, metric="value", height=chart_height, mode=display_mode)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.divider()

        # ==================== SECTION 2: Distribution Maps ====================
        st.subheader("🗺️ Distribution Maps")

        # Value vs Perp % Heatmap
        st.markdown("#### Account Value vs Perp Allocation")
        st.caption("Heatmap showing how many wallets fall into each Value/Perp% bucket")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = create_value_perp_heatmap(portfolio_df.copy())
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with col2:
            st.markdown("##### 📈 Insights")
            high_perp = len(portfolio_df[portfolio_df['perp_pct'] > 80])
            low_perp = len(portfolio_df[portfolio_df['perp_pct'] < 20])
            whales_high_perp = len(portfolio_df[(portfolio_df['total_value'] > 10e6) & (portfolio_df['perp_pct'] > 80)])

            st.metric("High Perp (>80%)", high_perp)
            st.metric("Low Perp (<20%)", low_perp)
            st.metric("Whales >$10M + High Perp", whales_high_perp)

        # Histograms
        col1, col2 = st.columns(2)
        with col1:
            fig = create_histogram(portfolio_df, "perp_pct", "Perp %", bins=10)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with col2:
            fig = create_histogram(portfolio_df, "total_value", "Account Value ($)", bins=15)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.divider()

        # Entity Distribution
        st.markdown("#### 🏢 Entity Type vs Perp Allocation")
        st.caption("Heatmap showing total AUM by entity type and Perp% range")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = create_entity_perp_heatmap(portfolio_df.copy())
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with col2:
            st.markdown("##### 🏢 Entity Summary")
            entity_summary = portfolio_df.groupby('entity').agg({
                'total_value': 'sum',
                'perp_pct': 'mean',
                'address': 'count'
            }).round(1)
            entity_summary.columns = ['Total AUM', 'Avg Perp %', 'Count']
            entity_summary['Total AUM'] = entity_summary['Total AUM'].apply(lambda x: format_currency(x))
            entity_summary['Avg Perp %'] = entity_summary['Avg Perp %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(entity_summary, width="stretch")

        st.divider()

        # Value vs PnL
        st.markdown("#### 💰 Account Value vs PnL")
        st.caption("Heatmap showing wallet distribution by value and profitability")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = create_value_pnl_heatmap(portfolio_df.copy())
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with col2:
            st.markdown("##### 💰 PnL Summary")
            profitable = len(portfolio_df[portfolio_df['total_pnl'] > 0])
            losing = len(portfolio_df[portfolio_df['total_pnl'] < 0])
            total_pnl = portfolio_df['total_pnl'].sum()
            avg_pnl = portfolio_df['total_pnl'].mean()

            st.metric("Profitable Wallets", profitable)
            st.metric("Losing Wallets", losing)
            st.metric("Total PnL", format_currency(total_pnl))
            st.metric("Avg PnL", format_currency(avg_pnl))

        # PnL histogram
        fig = create_histogram(portfolio_df, "total_pnl", "PnL ($)", bins=20)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.divider()

        # ==================== SECTION 3: Key Activity ====================
        st.subheader("📅 Key Activity Calendar")
        st.caption("GitHub-style heatmap showing trading activity over time")

        # Wallet selector for activity
        col1, col2, col3, col4 = st.columns([2, 0.7, 0.7, 0.8])
        with col1:
            wallet_options = ["📊 All Wallets"] + portfolio_df["display_name"].tolist()
            selected_wallet = st.selectbox(
                "Select Wallet",
                wallet_options,
                key="activity_wallet"
            )
        with col2:
            current_year = datetime.now().year
            from_year = st.number_input(
                "From Year",
                min_value=2020,
                max_value=current_year,
                value=current_year,
                step=1,
                key="from_year_input"
            )
        with col3:
            to_year = st.number_input(
                "To Year",
                min_value=2020,
                max_value=current_year,
                value=current_year,
                step=1,
                key="to_year_input"
            )
        with col4:
            fetch_activity = st.button("🔄 Fetch Activity", type="primary", key="fetch_activity_btn")

        # Validate year range
        if from_year > to_year:
            st.error("⚠️ 'From Year' must be <= 'To Year'")

        # Legend
        st.markdown(create_activity_legend(), unsafe_allow_html=True)

        if fetch_activity and from_year <= to_year:
            is_all_wallets = selected_wallet == "📊 All Wallets"
            year_range = list(range(from_year, to_year + 1))
            year_label = f"{from_year}" if from_year == to_year else f"{from_year}-{to_year}"

            if is_all_wallets:
                # Use filtered_df (original CSV list) instead of portfolio_df to get ALL wallets
                all_wallets_df = filtered_df.copy()
                total_wallets = len(all_wallets_df)

                # Fetch from all wallets
                with st.spinner(f"Fetching trades for all {total_wallets} wallets ({year_label})..."):
                    client = HyperliquidClient()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    all_fills = []
                    start_time = datetime(from_year, 1, 1)
                    end_time = datetime(to_year, 12, 31, 23, 59, 59)

                    for i, (_, row) in enumerate(all_wallets_df.iterrows()):
                        wallet_address = row["trader_address"]  # Use trader_address from CSV
                        wallet_name = row["display_name"]
                        status_text.text(f"Fetching {wallet_name[:30]}...")

                        try:
                            fills = client.get_user_fills_by_time(wallet_address, start_time, end_time)
                            for f in fills:
                                all_fills.append({
                                    'wallet': wallet_name,
                                    'coin': f.coin,
                                    'side': f.side,
                                    'direction': f.direction,
                                    'size': f.size,
                                    'price': f.price,
                                    'pnl': f.pnl,
                                    'timestamp': f.timestamp,
                                    'fee': f.fee
                                })
                        except Exception:
                            pass

                        progress_bar.progress((i + 1) / total_wallets)
                        time.sleep(0.02)

                    progress_bar.empty()
                    status_text.empty()

                    # Always update session state
                    st.session_state.calendar_years = year_range
                    st.session_state.activity_mode = "all"
                    # Store all wallet names for heatmap (including those with no trades)
                    st.session_state.all_wallet_names = all_wallets_df["display_name"].tolist()

                    if all_fills:
                        st.session_state.activity_fills = pd.DataFrame(all_fills)
                        st.success(f"✅ Found {len(all_fills)} trades across all wallets in {year_label}")
                    else:
                        st.warning(f"No trades found for {year_label}")
                        st.session_state.activity_fills = pd.DataFrame()
            else:
                # Fetch single wallet
                wallet_row = portfolio_df[portfolio_df["display_name"] == selected_wallet].iloc[0]
                wallet_address = wallet_row["address"]

                with st.spinner(f"Fetching trades for {selected_wallet} ({year_label})..."):
                    client = HyperliquidClient()

                    start_time = datetime(from_year, 1, 1)
                    end_time = datetime(to_year, 12, 31, 23, 59, 59)

                    fills = client.get_user_fills_by_time(wallet_address, start_time, end_time)

                    # Always update session state
                    st.session_state.calendar_years = year_range
                    st.session_state.activity_mode = "single"

                    if fills:
                        fills_df = pd.DataFrame([{
                            'wallet': selected_wallet,
                            'coin': f.coin,
                            'side': f.side,
                            'direction': f.direction,
                            'size': f.size,
                            'price': f.price,
                            'pnl': f.pnl,
                            'timestamp': f.timestamp,
                            'fee': f.fee
                        } for f in fills])

                        st.session_state.activity_fills = fills_df
                        st.success(f"✅ Found {len(fills)} trades in {year_label}")
                    else:
                        st.warning(f"No trades found for {year_label}")
                        st.session_state.activity_fills = pd.DataFrame()

        # Display calendar if data exists
        if "activity_fills" in st.session_state:
            fills_df = st.session_state.activity_fills
            years = st.session_state.get("calendar_years", [datetime.now().year])

            # Create single combined calendar for the entire date range
            from_yr = min(years)
            to_yr = max(years)
            fig = create_activity_calendar_range(fills_df, from_yr, to_yr)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"main_calendar_{from_yr}_{to_yr}")

            if len(fills_df) > 0:
                st.divider()

                # Summary stats - 4 trade types
                open_long = len(fills_df[fills_df['direction'] == 'Open Long'])
                close_long = len(fills_df[fills_df['direction'] == 'Close Long'])
                open_short = len(fills_df[fills_df['direction'] == 'Open Short'])
                close_short = len(fills_df[fills_df['direction'] == 'Close Short'])
                total_pnl = fills_df['pnl'].sum()

                # Row 1: Total and PnL
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Trades", len(fills_df))
                with col2:
                    st.metric("Realized PnL", format_currency(total_pnl))

                # Row 2: 4 trade types
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🟢 Open Long", f"{open_long:,}")
                with col2:
                    st.metric("🔵 Close Long", f"{close_long:,}")
                with col3:
                    st.metric("🔴 Open Short", f"{open_short:,}")
                with col4:
                    st.metric("🟠 Close Short", f"{close_short:,}")

                # Show all wallets detail heatmap if in all mode
                is_all_mode = st.session_state.get("activity_mode", "single") == "all"
                if is_all_mode and 'wallet' in fills_df.columns:
                    st.divider()

                    # Enhanced header with legend
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                        border-radius: 16px;
                        padding: 24px;
                        margin: 16px 0;
                        border: 1px solid #334155;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                            <div>
                                <h2 style="margin: 0; color: #f1f5f9; font-size: 24px;">👛 All Wallets Activity</h2>
                                <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 14px;">
                                    Each row represents a wallet • Columns are days of the year • Hover for details
                                </p>
                            </div>
                            <div style="display: flex; gap: 24px; align-items: center;">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <div style="width: 16px; height: 16px; background: #22c55e; border-radius: 4px;"></div>
                                    <span style="color: #94a3b8; font-size: 13px;">Long</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <div style="width: 16px; height: 16px; background: #ef4444; border-radius: 4px;"></div>
                                    <span style="color: #94a3b8; font-size: 13px;">Short</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <div style="width: 16px; height: 16px; background: #0f172a; border-radius: 4px; border: 1px solid #334155;"></div>
                                    <span style="color: #94a3b8; font-size: 13px;">No Activity</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Single combined heatmap for all wallets
                    all_wallet_names = st.session_state.get("all_wallet_names", None)
                    all_wallets_fig = create_all_wallets_heatmap(fills_df.copy(), from_yr, to_yr, all_wallet_names)
                    if all_wallets_fig:
                        st.plotly_chart(all_wallets_fig, width="stretch", config={
                            "displayModeBar": True,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            "displaylogo": False
                        }, key=f"all_wallets_heatmap_{from_yr}_{to_yr}")

                st.divider()

                # Recent trades table
                st.markdown("### 📜 Recent Trades")

                recent_df = fills_df.sort_values('timestamp', ascending=False).head(100).copy()
                recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                recent_df['size'] = recent_df['size'].apply(lambda x: f"{x:,.4f}")
                recent_df['price'] = recent_df['price'].apply(lambda x: f"${x:,.2f}")
                recent_df['pnl'] = recent_df['pnl'].apply(lambda x: format_currency(x))
                recent_df['fee'] = recent_df['fee'].apply(lambda x: f"${x:,.4f}")

                # Show wallet column if viewing all wallets
                is_all_mode = st.session_state.get("activity_mode", "single") == "all"
                if is_all_mode and 'wallet' in recent_df.columns:
                    display_cols = ['timestamp', 'wallet', 'coin', 'direction', 'size', 'price', 'pnl', 'fee']
                    recent_df = recent_df[display_cols]
                    recent_df.columns = ['Time', 'Wallet', 'Coin', 'Direction', 'Size', 'Price', 'Realized PnL', 'Fee']
                else:
                    display_cols = ['timestamp', 'coin', 'direction', 'size', 'price', 'pnl', 'fee']
                    recent_df = recent_df[display_cols]
                    recent_df.columns = ['Time', 'Coin', 'Direction', 'Size', 'Price', 'Realized PnL', 'Fee']

                st.dataframe(recent_df, hide_index=True, width="stretch", height=400)

                # Trade breakdown by coin
                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🪙 Activity by Coin")
                    coin_activity = fills_df.groupby('coin').agg({
                        'size': 'count',
                        'pnl': 'sum'
                    }).reset_index()
                    coin_activity.columns = ['Coin', 'Trades', 'PnL']
                    coin_activity = coin_activity.sort_values('Trades', ascending=False).head(10)
                    coin_activity['PnL'] = coin_activity['PnL'].apply(lambda x: format_currency(x))
                    st.dataframe(coin_activity, hide_index=True, width="stretch")

                # Show wallet breakdown if viewing all wallets
                if is_all_mode and 'wallet' in fills_df.columns:
                    with col2:
                        st.markdown("### 👛 Activity by Wallet")
                        wallet_activity = fills_df.groupby('wallet').agg({
                            'size': 'count',
                            'pnl': 'sum'
                        }).reset_index()
                        wallet_activity.columns = ['Wallet', 'Trades', 'PnL']
                        wallet_activity = wallet_activity.sort_values('Trades', ascending=False)
                        wallet_activity_display = wallet_activity.head(10).copy()
                        wallet_activity_display['PnL'] = wallet_activity_display['PnL'].apply(lambda x: format_currency(x))
                        st.dataframe(wallet_activity_display, hide_index=True, width="stretch")

                    # Individual wallet details section
                    st.divider()
                    st.markdown("### 👛 Individual Wallet Activity")
                    st.caption("Expand each wallet to see their trading calendar")

                    # Get unique wallets sorted by trade count
                    wallets_sorted = wallet_activity['Wallet'].tolist()

                    # Show each wallet's calendar in an expander
                    for wallet_idx, wallet_name in enumerate(wallets_sorted):
                        wallet_fills = fills_df[fills_df['wallet'] == wallet_name].copy()
                        wallet_trades = len(wallet_fills)
                        wallet_pnl = wallet_fills['pnl'].sum()

                        # Count 4 trade types
                        w_open_long = len(wallet_fills[wallet_fills['direction'] == 'Open Long'])
                        w_close_long = len(wallet_fills[wallet_fills['direction'] == 'Close Long'])
                        w_open_short = len(wallet_fills[wallet_fills['direction'] == 'Open Short'])
                        w_close_short = len(wallet_fills[wallet_fills['direction'] == 'Close Short'])

                        # Create expander header with summary
                        pnl_color = "🟢" if wallet_pnl >= 0 else "🔴"
                        header = f"{wallet_name} | {wallet_trades} trades | {pnl_color} {format_currency(wallet_pnl)}"

                        with st.expander(header, expanded=False):
                            # Summary metrics - Row 1
                            mcol1, mcol2 = st.columns(2)
                            with mcol1:
                                st.metric("Total Trades", wallet_trades)
                            with mcol2:
                                st.metric("Realized PnL", format_currency(wallet_pnl))

                            # Row 2: 4 trade types
                            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                            with mcol1:
                                st.metric("🟢 Open Long", w_open_long)
                            with mcol2:
                                st.metric("🔵 Close Long", w_close_long)
                            with mcol3:
                                st.metric("🔴 Open Short", w_open_short)
                            with mcol4:
                                st.metric("🟠 Close Short", w_close_short)

                            # Single combined calendar for the date range
                            wallet_fig = create_activity_calendar_range(wallet_fills, from_yr, to_yr)
                            st.plotly_chart(wallet_fig, width="stretch", config={"displayModeBar": False}, key=f"wallet_detail_{wallet_idx}_{from_yr}_{to_yr}")

                            # Top coins for this wallet
                            wallet_coins = wallet_fills.groupby('coin').agg({
                                'size': 'count',
                                'pnl': 'sum'
                            }).reset_index()
                            wallet_coins.columns = ['Coin', 'Trades', 'PnL']
                            wallet_coins = wallet_coins.sort_values('Trades', ascending=False).head(5)
                            wallet_coins['PnL'] = wallet_coins['PnL'].apply(lambda x: format_currency(x))

                            st.markdown("**Top Coins:**")
                            st.dataframe(wallet_coins, hide_index=True, width="stretch")

        else:
            st.info("👆 Select a wallet and click 'Fetch Activity' to view trading calendar")

        # Detailed table at the bottom
        st.divider()
        st.subheader("📋 Detailed Data")

        table_df = portfolio_df[["display_name", "entity", "total_value", "perp_value", "spot_value", "perp_pct", "total_pnl"]].copy()
        table_df.columns = ["Wallet", "Entity", "Total Value", "Perp Value", "Spot Value", "Perp %", "PnL"]

        table_df["Total Value"] = table_df["Total Value"].apply(lambda x: format_currency(x))
        table_df["Perp Value"] = table_df["Perp Value"].apply(lambda x: format_currency(x))
        table_df["Spot Value"] = table_df["Spot Value"].apply(lambda x: format_currency(x))
        table_df["Perp %"] = table_df["Perp %"].apply(lambda x: f"{x:.1f}%")
        table_df["PnL"] = table_df["PnL"].apply(lambda x: format_currency(x))

        st.dataframe(table_df, hide_index=True, width="stretch", height=400)

    else:
        st.info("👆 Click 'Fetch Live Data' in the sidebar to load real-time Perp/Spot breakdown")

        st.subheader("📋 Loaded Wallets (from CSV)")
        display_df = filtered_df[["trader_address_label", "Entity", "account_value", "roi", "total_pnl(unrealize profit)"]].copy()
        display_df.columns = ["Wallet", "Entity", "Account Value", "ROI", "Unrealized PnL"]
        st.dataframe(display_df, hide_index=True, width="stretch", height=600)


if __name__ == "__main__":
    main()
