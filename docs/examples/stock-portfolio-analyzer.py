"""Example: Stock Portfolio Analyzer — agentic object with tools, sandbox code, and structured output.

Usage: python stock-portfolio-analyzer.py <backend-url>
"""

import asyncio
import math
import statistics
import random
import sys
from dataclasses import dataclass
from enum import Enum

from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import agentic_object, tool


class Sector(Enum):
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCE = "Finance"
    ENERGY = "Energy"
    CONSUMER = "Consumer"


@dataclass
class Holding:
    ticker: str
    shares: int
    sector: Sector


@agentic_object(allow_code_execution=True, imports=[math, random, statistics])
class StockPortfolioAnalyzer(AgenticObjectBase):
    """You are a stock portfolio analyzer. You manage a portfolio of stock
    holdings, current prices, and historical returns. Use add_holding and
    set_price to manage data, list_holdings to inspect the portfolio, and
    the sandbox to calculate performance metrics like total value and
    sector allocation."""

    def __init__(self):
        super().__init__()
        self._holdings: list[Holding] = []
        self._prices: dict[str, float] = {}
        self._returns: list[float] = []

    @tool
    def add_holding(self, ticker: str, shares: int, sector: Sector) -> str:
        """Add a stock holding with ticker, number of shares, and sector."""
        self._holdings.append(Holding(ticker=ticker, shares=shares, sector=sector))
        return f"Added holding: {ticker} — {shares} shares ({sector.value})."

    @tool
    def set_price(self, ticker: str, price: float) -> str:
        """Set the current price for a stock ticker."""
        self._prices[ticker] = price
        return f"Price set for {ticker}: {price:.2f}."

    @tool
    def set_returns(self, returns: list[float]) -> str:
        """Set historical returns for a stock (list of daily percentage returns)."""
        self._returns = returns
        return f"Recorded {len(returns)} return values."

    @tool
    def list_holdings(self) -> list[dict]:
        """Return all holdings with current prices."""
        result = []
        for h in self._holdings:
            price = self._prices.get(h.ticker, 0.0)
            result.append({
                "ticker": h.ticker,
                "shares": h.shares,
                "sector": h.sector.value,
                "price": price,
                "value": round(h.shares * price, 2),
            })
        return result

    @tool
    def get_holdings_by_sector(self) -> dict[str, float]:
        """Return total portfolio value per sector."""
        sector_values: dict[str, float] = {}
        for h in self._holdings:
            price = self._prices.get(h.ticker, 0.0)
            value = h.shares * price
            sector_values[h.sector.value] = sector_values.get(h.sector.value, 0.0) + value
        return {k: round(v, 2) for k, v in sector_values.items()}


@dataclass
class PerformanceMetrics:
    total_value: float
    avg_return: float
    std_deviation: float
    sharpe_ratio: float


@dataclass
class SectorAllocation:
    sectors: list[dict[str, float]]
    recommended_allocation: dict[str, float]
    rebalance_note: str


async def main():
    #from peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)
    args = sys.argv[1:] + [None] * 3
    url, api_type, api_key = args[:3]
    await ChatBotManager.add_backend("local", url, api_type=api_type, api_key=api_key)

    analyzer = StockPortfolioAnalyzer()

    # Build portfolio
    await analyzer.invoke_agent(
        "Add holdings: AAPL 50 shares (Technology), GOOGL 20 shares (Technology), "
        "JNJ 30 shares (Healthcare), XOM 40 shares (Energy)."
    )
    await analyzer.invoke_agent(
        "Set prices: AAPL 175.50, GOOGL 141.80, JNJ 156.30, XOM 104.20."
    )
    await analyzer.invoke_agent(
        "Set returns: [0.5, -0.2, 0.8, -0.4, 0.3, 0.1, -0.6, 0.9, 0.2, -0.1]."
    )

    # Get structured performance metrics
    result = await analyzer.invoke_agent(
        "Calculate total portfolio value and performance metrics (average return, "
        "standard deviation, Sharpe ratio).",
        output_schema=PerformanceMetrics,
    )
    print("Performance metrics:", result)

    # Get structured sector allocation analysis
    result = await analyzer.invoke_agent(
        "Analyze the sector allocation and suggest a recommended diversification.",
        output_schema=SectorAllocation,
    )
    print("Sector allocation:", result)


if __name__ == "__main__":
    asyncio.run(main())
