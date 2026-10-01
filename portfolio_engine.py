"""Backtest engine for MMM threshold portfolio simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Trade:
    ticker: str
    action: str
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    price: float
    shares: float
    value: float
    mmm: float


def _clip100(series: pd.Series) -> pd.Series:
    return series.clip(-100, 100)


def _squash(signal: pd.Series, scale: float) -> pd.Series:
    return _clip100(100.0 * np.tanh(signal / scale))


def _norm_z(series: pd.Series, window: int, saturation: float = 1.5) -> pd.Series:
    minimum = max(2, window // 3)
    mean = series.rolling(window, min_periods=minimum).mean()
    deviation = series.rolling(window, min_periods=minimum).std(ddof=0)
    z_score = (series - mean) / deviation.replace(0, np.nan)
    return _clip100(100.0 * np.tanh(z_score.fillna(0.0) / saturation))


def market_mood_meter(close: pd.Series, high: pd.Series, low: pd.Series,
                      volume: pd.Series, dpo_period: int = 20,
                      wyckoff_period: int = 20, speed_period: int = 30,
                      normalization_window: int = 100) -> pd.Series:
    shift = dpo_period // 2 + 1
    dpo_average = close.rolling(dpo_period, min_periods=max(2, dpo_period // 2)).mean()
    dpo = (close - dpo_average.shift(shift)) / close * 100
    dpo_score = _norm_z(dpo, normalization_window)

    price_range = (high - low).replace(0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / price_range
    money_flow_volume = (money_flow_multiplier * volume).fillna(0.0)
    cmf = (money_flow_volume.rolling(wyckoff_period, min_periods=max(2, wyckoff_period // 2)).sum()
           / volume.rolling(wyckoff_period, min_periods=max(2, wyckoff_period // 2)).sum())
    wyckoff_score = _squash(cmf, 0.45)

    daily_return = close.pct_change()
    volatility = daily_return.rolling(60, min_periods=20).std()
    speed_signal = close.pct_change(speed_period) / (volatility * np.sqrt(speed_period))
    speed_score = _squash(speed_signal, 0.5)

    return _clip100(((dpo_score + wyckoff_score + speed_score) / 3).ewm(span=3, adjust=False).mean())


def download_prices(tickers: Iterable[str], period: str) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if frame.empty:
            continue
        frame = frame[["Open", "High", "Low", "Close", "Volume"]].dropna()
        index = pd.to_datetime(frame.index)
        frame.index = index.tz_localize(None) if index.tz is not None else index
        prices[ticker] = frame
    return prices


def trim_prices(prices: dict[str, pd.DataFrame], period: str) -> dict[str, pd.DataFrame]:
    """Keep the requested backtest window after indicators have warm-up data."""
    offsets = {
        "5d": pd.Timedelta(days=7), "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3), "6mo": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1), "2y": pd.DateOffset(years=2),
        "5y": pd.DateOffset(years=5), "10y": pd.DateOffset(years=10),
    }
    offset = offsets.get(period)
    if offset is None:
        return prices
    trimmed: dict[str, pd.DataFrame] = {}
    for ticker, frame in prices.items():
        end = frame.index[-1]
        start = end - offset
        window = frame.loc[frame.index >= start]
        if not window.empty:
            trimmed[ticker] = window
    return trimmed


def market_cap_weights(tickers: Iterable[str]) -> dict[str, float]:
    """Return current market-cap weights for a scenario allocation.

    Current capitalization is deliberately exposed as a scenario assumption; it is
    not historical point-in-time data and should not be interpreted as such.
    """
    caps: dict[str, float] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            value = float(getattr(info, "market_cap", 0) or 0)
            if np.isfinite(value) and value > 0:
                caps[ticker] = value
        except (AttributeError, TypeError, ValueError, OSError):
            continue
    if not caps:
        raise ValueError("Capitalizzazioni di mercato non disponibili per i ticker selezionati.")
    total = sum(caps.values())
    return {ticker: caps.get(ticker, 0.0) / total for ticker in tickers}


def historical_market_cap_weights(prices: dict[str, pd.DataFrame],
                                  start_date: pd.Timestamp) -> tuple[dict[str, float], pd.DataFrame]:
    """Estimate point-in-time market caps at the beginning of the backtest.

    The estimate uses the first close in the selected window and the nearest
    historical shares-outstanding observation around that date. It avoids using
    today's market cap, which would introduce look-ahead bias.
    """
    records: list[dict] = []
    for ticker, frame in prices.items():
        window = frame.loc[frame.index >= start_date]
        if window.empty:
            continue
        price_date = window.index[0]
        price = float(window["Close"].iloc[0])
        shares_value = None
        shares_date = None
        try:
            shares = yf.Ticker(ticker).get_shares_full(
                start=price_date - pd.Timedelta(days=180),
                end=price_date + pd.Timedelta(days=180),
            )
            if shares is not None and not shares.empty:
                shares = pd.Series(shares).dropna()
                shares.index = pd.to_datetime(shares.index)
                if shares.index.tz is not None:
                    shares.index = shares.index.tz_localize(None)
                eligible = shares.loc[shares.index <= price_date]
                if eligible.empty:
                    # Yahoo often reports the first nearby share count after the date.
                    eligible = shares.loc[shares.index > price_date].sort_index().head(1)
                if not eligible.empty:
                    shares_date = eligible.index[-1]
                    shares_value = float(eligible.iloc[-1])
        except (AttributeError, KeyError, TypeError, ValueError, OSError):
            shares_value = None
        if shares_value is None or not np.isfinite(shares_value) or shares_value <= 0:
            continue
        records.append({
            "Ticker": ticker,
            "Data prezzo": price_date.strftime("%Y-%m-%d"),
            "Prezzo iniziale": price,
            "Data azioni": shares_date.strftime("%Y-%m-%d") if shares_date is not None else "N/D",
            "Azioni in circolazione": shares_value,
            "Market cap iniziale": price * shares_value,
        })
    if not records:
        raise ValueError("Azioni storiche non disponibili per calcolare i pesi market cap iniziali.")
    details = pd.DataFrame(records)
    total = details["Market cap iniziale"].sum()
    weights = dict(zip(details["Ticker"], details["Market cap iniziale"] / total))
    return weights, details


def buy_and_hold_portfolio(prices: dict[str, pd.DataFrame], capital: float,
                           start_date: pd.Timestamp,
                           allocation_weights: dict[str, float]) -> pd.Series:
    """Build a passive buy-and-hold equity curve with the same initial weights."""
    dates = sorted(set().union(*(frame.index for frame in prices.values())))
    dates = [date for date in dates if date >= start_date]
    if not dates:
        raise ValueError("Nessun dato buy-and-hold nell'intervallo selezionato.")
    curve = pd.DataFrame(index=pd.DatetimeIndex(dates))
    for ticker, frame in prices.items():
        window = frame.loc[frame.index >= start_date]
        if window.empty:
            continue
        first_close = float(window["Close"].iloc[0])
        sleeve = capital * allocation_weights[ticker]
        values = window["Close"] / first_close * sleeve
        curve[ticker] = values.reindex(curve.index).ffill().fillna(sleeve)
    return curve.sum(axis=1).rename("BuyHold")


def simulate_portfolio(prices: dict[str, pd.DataFrame], capital: float,
                       entry_threshold: float, exit_threshold: float,
                       dpo_period: int = 20, wyckoff_period: int = 20,
                       speed_period: int = 30, normalization_window: int = 100,
                       commission_rate: float = 0.0,
                       start_date: pd.Timestamp | None = None,
                       allocation_weights: dict[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[Trade]]:
    if capital <= 0:
        raise ValueError("Il capitale deve essere maggiore di zero.")
    if entry_threshold >= exit_threshold:
        raise ValueError("La soglia di ingresso deve essere inferiore alla soglia di uscita.")
    if not prices:
        raise ValueError("Nessun dato storico disponibile per i ticker selezionati.")
    if not 0 <= commission_rate < 1:
        raise ValueError("La commissione deve essere compresa tra 0% e 100%.")

    if allocation_weights is None:
        allocation_weights = {ticker: 1 / len(prices) for ticker in prices}
    if set(allocation_weights) != set(prices) or any(weight <= 0 for weight in allocation_weights.values()):
        raise ValueError("I pesi di allocazione devono essere positivi e presenti per ogni ticker.")
    weight_total = sum(allocation_weights.values())
    allocation_weights = {ticker: weight / weight_total for ticker, weight in allocation_weights.items()}
    all_dates = sorted(set().union(*(frame.index for frame in prices.values())))
    if start_date is not None:
        all_dates = [date for date in all_dates if date >= start_date]
    if not all_dates:
        raise ValueError("L'intervallo selezionato non contiene sedute valide.")
    equity = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
    trades: list[Trade] = []
    ending_cash: dict[str, float] = {}

    for ticker, frame in prices.items():
        mmm = market_mood_meter(frame["Close"], frame["High"], frame["Low"], frame["Volume"],
                                 dpo_period, wyckoff_period, speed_period, normalization_window)
        frame = frame.copy()
        frame["MMM"] = mmm
        allocation = capital * allocation_weights[ticker]
        cash = allocation
        shares = 0.0
        entry_price = None
        ticker_equity = pd.Series(index=frame.index, dtype=float)

        for position in range(len(frame)):
            row = frame.iloc[position]
            signal_date = frame.index[position]
            if start_date is not None and signal_date < start_date:
                continue
            execution_position = position + 1
            if execution_position < len(frame):
                execution_row = frame.iloc[execution_position]
                execution_date = frame.index[execution_position]
                if shares == 0 and row["MMM"] <= entry_threshold:
                    price = float(execution_row["Open"])
                    shares = cash / (price * (1 + commission_rate))
                    value = shares * price
                    cash -= value * (1 + commission_rate)
                    entry_price = price
                    trades.append(Trade(ticker, "BUY", signal_date, execution_date, price,
                                        shares, value, float(row["MMM"])))
                elif shares > 0 and row["MMM"] >= exit_threshold:
                    price = float(execution_row["Open"])
                    value = shares * price
                    cash += value * (1 - commission_rate)
                    trades.append(Trade(ticker, "SELL", signal_date, execution_date, price,
                                        shares, value, float(row["MMM"])))
                    shares = 0.0
                    entry_price = None
            ticker_equity.loc[signal_date] = cash + shares * float(row["Close"])

        last_date = frame.index[-1]
        if shares > 0:
            price = float(frame["Close"].iloc[-1])
            value = shares * price
            cash += value * (1 - commission_rate)
            trades.append(Trade(ticker, "LIQUIDATE", last_date, last_date, price,
                                shares, value, float(frame["MMM"].iloc[-1])))
            shares = 0.0
        ending_cash[ticker] = cash
        equity[ticker] = ticker_equity.reindex(equity.index).ffill().fillna(allocation)

    equity["Portfolio"] = equity[list(prices)].sum(axis=1)
    equity["Return"] = equity["Portfolio"] / capital - 1
    summary = pd.DataFrame({
        "Metric": ["Initial capital", "Final value", "Profit/Loss", "Return %", "Max drawdown %"],
        "Value": [capital, equity["Portfolio"].iloc[-1], equity["Portfolio"].iloc[-1] - capital,
                  equity["Return"].iloc[-1] * 100, _max_drawdown(equity["Portfolio"]) * 100],
    })
    return equity, summary, trades


def _max_drawdown(series: pd.Series) -> float:
    drawdown = series / series.cummax() - 1
    return float(drawdown.min())
