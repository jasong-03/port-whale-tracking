"""Hyperliquid API client for fetching portfolio data."""

import requests
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PortfolioMetrics:
    """Portfolio metrics for a specific time period."""
    account_value: float
    pnl: float
    volume: float


@dataclass
class PortfolioBreakdown:
    """Breakdown of portfolio into Perp vs Spot."""
    total: PortfolioMetrics
    perp: PortfolioMetrics
    spot: PortfolioMetrics  # Calculated: total - perp


@dataclass
class TradeFill:
    """A single trade fill."""
    coin: str
    side: str  # "B" (buy) or "A" (ask/sell)
    direction: str  # "Open Long", "Open Short", "Close Long", "Close Short", etc.
    size: float
    price: float
    pnl: float
    timestamp: datetime
    fee: float


class HyperliquidClient:
    """Client for interacting with Hyperliquid Info API."""

    BASE_URL = "https://api.hyperliquid.xyz/info"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def get_portfolio(self, user_address: str) -> Optional[dict]:
        """
        Fetch user portfolio data.

        Args:
            user_address: Ethereum address (0x...)

        Returns:
            Raw portfolio response or None if error
        """
        try:
            response = self.session.post(
                self.BASE_URL,
                json={"type": "portfolio", "user": user_address}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching portfolio: {e}")
            return None

    def get_portfolio_breakdown(self, user_address: str, period: str = "day") -> Optional[PortfolioBreakdown]:
        """
        Get portfolio breakdown for Perp vs Spot.

        Args:
            user_address: Ethereum address
            period: Time period ("day", "week", "month", "allTime")

        Returns:
            PortfolioBreakdown with total, perp, and spot metrics
        """
        raw_data = self.get_portfolio(user_address)
        if not raw_data:
            return None

        # Convert list to dict for easier access
        portfolio_dict = {item[0]: item[1] for item in raw_data}

        # Get period data
        perp_period = f"perp{period.capitalize()}" if period != "allTime" else "perpAllTime"

        total_data = portfolio_dict.get(period, {})
        perp_data = portfolio_dict.get(perp_period, {})

        if not total_data:
            return None

        # Extract metrics
        total_metrics = self._extract_metrics(total_data)
        perp_metrics = self._extract_metrics(perp_data)

        # Calculate spot (total - perp)
        spot_metrics = PortfolioMetrics(
            account_value=max(0, total_metrics.account_value - perp_metrics.account_value),
            pnl=total_metrics.pnl - perp_metrics.pnl,
            volume=max(0, total_metrics.volume - perp_metrics.volume)
        )

        return PortfolioBreakdown(
            total=total_metrics,
            perp=perp_metrics,
            spot=spot_metrics
        )

    def _extract_metrics(self, period_data: dict) -> PortfolioMetrics:
        """Extract metrics from period data."""
        # Get latest account value from history
        account_value_history = period_data.get("accountValueHistory", [])
        account_value = float(account_value_history[-1][1]) if account_value_history else 0.0

        # Get latest PnL from history
        pnl_history = period_data.get("pnlHistory", [])
        pnl = float(pnl_history[-1][1]) if pnl_history else 0.0

        # Get volume
        volume = float(period_data.get("vlm", "0"))

        return PortfolioMetrics(
            account_value=account_value,
            pnl=pnl,
            volume=volume
        )

    def get_user_fills(self, user_address: str, limit: int = 2000) -> List[TradeFill]:
        """
        Fetch user's trade fills (trading history).

        Args:
            user_address: Ethereum address (0x...)
            limit: Max number of fills to return (max 2000)

        Returns:
            List of TradeFill objects
        """
        try:
            response = self.session.post(
                self.BASE_URL,
                json={"type": "userFills", "user": user_address}
            )
            response.raise_for_status()
            raw_fills = response.json()

            fills = []
            for fill in raw_fills[:limit]:
                try:
                    fills.append(TradeFill(
                        coin=fill.get("coin", ""),
                        side=fill.get("side", ""),
                        direction=fill.get("dir", ""),
                        size=float(fill.get("sz", 0)),
                        price=float(fill.get("px", 0)),
                        pnl=float(fill.get("closedPnl", 0)),
                        timestamp=datetime.fromtimestamp(fill.get("time", 0) / 1000),
                        fee=float(fill.get("fee", 0))
                    ))
                except (ValueError, TypeError):
                    continue

            return fills
        except requests.RequestException as e:
            print(f"Error fetching user fills: {e}")
            return []

    def get_user_fills_by_time(self, user_address: str, start_time: datetime, end_time: datetime = None) -> List[TradeFill]:
        """
        Fetch user's trade fills within a time range.

        Args:
            user_address: Ethereum address
            start_time: Start datetime
            end_time: End datetime (defaults to now)

        Returns:
            List of TradeFill objects
        """
        try:
            payload = {
                "type": "userFillsByTime",
                "user": user_address,
                "startTime": int(start_time.timestamp() * 1000)
            }
            if end_time:
                payload["endTime"] = int(end_time.timestamp() * 1000)

            response = self.session.post(self.BASE_URL, json=payload)
            response.raise_for_status()
            raw_fills = response.json()

            fills = []
            for fill in raw_fills:
                try:
                    fills.append(TradeFill(
                        coin=fill.get("coin", ""),
                        side=fill.get("side", ""),
                        direction=fill.get("dir", ""),
                        size=float(fill.get("sz", 0)),
                        price=float(fill.get("px", 0)),
                        pnl=float(fill.get("closedPnl", 0)),
                        timestamp=datetime.fromtimestamp(fill.get("time", 0) / 1000),
                        fee=float(fill.get("fee", 0))
                    ))
                except (ValueError, TypeError):
                    continue

            return fills
        except requests.RequestException as e:
            print(f"Error fetching user fills by time: {e}")
            return []


# Mock data for testing without real wallet
def get_mock_portfolio_breakdown() -> PortfolioBreakdown:
    """Return mock data for demonstration."""
    return PortfolioBreakdown(
        total=PortfolioMetrics(
            account_value=125000.50,
            pnl=8500.25,
            volume=1250000.00
        ),
        perp=PortfolioMetrics(
            account_value=95000.00,
            pnl=7200.00,
            volume=1100000.00
        ),
        spot=PortfolioMetrics(
            account_value=30000.50,
            pnl=1300.25,
            volume=150000.00
        )
    )
