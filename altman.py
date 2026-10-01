"""Altman Z-score calculation from public-company financial statements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class AltmanResult:
    score: float | None
    zone: str
    applicable: bool
    reason: str
    variables: dict[str, float | None]
    source: str
    period: str | None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return None if not np.isfinite(number) else number
    except (TypeError, ValueError):
        return None


def _row(frame: pd.DataFrame, names: tuple[str, ...]) -> float | None:
    if frame is None or frame.empty:
        return None
    normalized = {str(index).lower().replace(" ", ""): index for index in frame.index}
    for name in names:
        index = normalized.get(name.lower().replace(" ", ""))
        if index is not None:
            values = frame.loc[index].dropna()
            if not values.empty:
                return _number(values.iloc[0])
    return None


def _period(frame: pd.DataFrame) -> str | None:
    if frame is None or frame.empty:
        return None
    try:
        return pd.Timestamp(frame.columns[0]).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(frame.columns[0])


def _zone(score: float) -> str:
    if score > 2.99:
        return "Safe zone"
    if score >= 1.81:
        return "Grey zone"
    return "Distress zone"


def calculate_altman_z(ticker: str, info: dict | None = None) -> AltmanResult:
    """Original listed-company formula: 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + X5."""
    info = info or {}
    sector = str(info.get("sector") or "").lower()
    if any(value in sector for value in ("financial", "bank", "insurance")):
        return AltmanResult(None, "Not applicable", False,
                            "La formula originale non è adatta a banche e assicurazioni.",
                            {}, "yfinance", None)

    company = yf.Ticker(ticker)
    balance = company.balance_sheet
    income = company.income_stmt
    if balance is None or balance.empty or income is None or income.empty:
        return AltmanResult(None, "Unavailable", True,
                            "Bilanci sufficienti non disponibili da yfinance.", {}, "yfinance", None)

    total_assets = _row(balance, ("Total Assets",))
    current_assets = _row(balance, ("Current Assets",))
    current_liabilities = _row(balance, ("Current Liabilities",))
    retained_earnings = _row(balance, ("Retained Earnings",))
    total_liabilities = _row(balance, ("Total Liabilities Net Minority Interest", "Total Liabilities"))
    sales = _row(income, ("Total Revenue", "Operating Revenue"))
    ebit = _row(income, ("EBIT", "Operating Income"))
    fast_info = info.get("_fast", {}) or {}
    market_equity = _number(info.get("marketCap") or fast_info.get("market_cap"))
    values = {
        "working_capital": (current_assets - current_liabilities
                             if current_assets is not None and current_liabilities is not None else None),
        "total_assets": total_assets,
        "retained_earnings": retained_earnings,
        "ebit": ebit,
        "market_value_equity": market_equity,
        "total_liabilities": total_liabilities,
        "sales": sales,
    }
    if any(value is None for value in values.values()) or any(
        value == 0 for value in (total_assets, total_liabilities)
    ):
        return AltmanResult(None, "Unavailable", True,
                            "Mancano una o più voci di bilancio o la capitalizzazione di mercato.",
                            values, "yfinance", _period(balance))

    score = (1.2 * values["working_capital"] / total_assets
             + 1.4 * retained_earnings / total_assets
             + 3.3 * ebit / total_assets
             + 0.6 * market_equity / total_liabilities
             + sales / total_assets)
    return AltmanResult(float(score), _zone(float(score)), True, "", values,
                        "yfinance", _period(balance))