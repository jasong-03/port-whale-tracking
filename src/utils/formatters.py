"""Number formatting utilities."""


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format number as currency with K/M/B suffixes.

    Examples:
        1234 -> "$1.23K"
        1234567 -> "$1.23M"
        1234567890 -> "$1.23B"
    """
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.{decimals}f}B"
    elif abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.{decimals}f}K"
    else:
        return f"${value:,.{decimals}f}"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with K/M/B suffixes (no currency symbol).

    Examples:
        1234 -> "1.23K"
        1234567 -> "1.23M"
    """
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{decimals}f}B"
    elif abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    else:
        return f"{value:,.{decimals}f}"


def format_percentage(value: float, total: float) -> str:
    """Calculate and format percentage."""
    if total == 0:
        return "0%"
    pct = (value / total) * 100
    return f"{pct:.1f}%"
