"""Interactive MMM threshold backtest page."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from portfolio_engine import (
    download_prices,
    historical_market_cap_weights,
    simulate_portfolio,
    trim_prices,
)


OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"
BENCHMARKS = {
    "S&P 500 Core EUR Acc · SXR8.DE": "SXR8.DE",
    "S&P 500 Core USD Acc · CSPX.L": "CSPX.L",
    "MSCI World EUR Acc · EUNL.DE": "EUNL.DE",
    "MSCI World USD Acc · SWDA.L": "SWDA.L",
}
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "NOVO-B.CO", "UNH",
    "8766.T", "LULU", "MRNA", "PFIZER.NS", "DPZ", "RACE", "CRM", "ADBE",
    "BIDU", "1211.HK", "BABA", "81810.HK", "JD", "005930.KS", "SKHY",
    "SNDK", "MU", "MRVL", "CGNX",
]
PERIODS = {
    "1 settimana": "5d", "1 mese": "1mo", "3 mesi": "3mo", "6 mesi": "6mo",
    "1 anno": "1y", "2 anni": "2y", "5 anni": "5y", "10 anni": "10y",
}
DOWNLOAD_PERIODS = {
    "5d": "1y", "1mo": "1y", "3mo": "1y", "6mo": "2y",
    "1y": "2y", "2y": "5y", "5y": "10y", "10y": "10y",
}

PORTFOLIO_ANALYSIS_PROMPT = """Sei un analista quantitativo senior. Analizza esclusivamente il risultato
backtest fornito, senza inventare dati, prezzi o notizie. Distingui risultati osservati,
calcoli e limiti del metodo. Confronta il portafoglio con l'ETF S&P 500 Core indicato nel
risultato, spiegando differenza di rendimento e profitto finale. Spiega il contributo dei
ticker, le operazioni generate dalle soglie MMM, il rendimento, il drawdown e i rischi di overfitting. Non presentare il risultato
come previsione né come consulenza personalizzata.

Rispondi in italiano con queste sezioni:
1. Sintesi del risultato
2. Contributo dei titoli e operazioni
3. Rischio e drawdown
4. Limiti statistici e operativi
5. Conclusione prudente

RISULTATO BACKTEST:
{result}
"""


def installed_models() -> list[str]:
    try:
        with urlopen(OLLAMA_TAGS_URL, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return sorted({item["name"] for item in payload.get("models", []) if item.get("name")})
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return []


def _json_default(value):
    """Serialize Pandas and NumPy scalar values used in the backtest payload."""
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def request_analysis(result: dict, model: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PORTFOLIO_ANALYSIS_PROMPT.format(
            result=json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
        )}],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    request = Request(OLLAMA_CHAT_URL, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=180) as response:
        content = json.loads(response.read().decode("utf-8")).get("message", {}).get("content")
    if not content:
        raise ValueError("Ollama ha restituito una risposta vuota.")
    return content


def render_chart(equity: pd.DataFrame, capital: float, benchmark_label: str) -> None:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=equity.index, y=equity["Portfolio"], mode="lines",
                                name="Portafoglio", line=dict(color="#4dabf7", width=2)))
    if "Benchmark" in equity:
        figure.add_trace(go.Scatter(x=equity.index, y=equity["Benchmark"], mode="lines",
                                    name=benchmark_label, line=dict(color="#ff922b", width=2)))
    figure.add_hline(y=capital, line_dash="dot", line_color="#adb5bd", annotation_text="Capitale iniziale")
    figure.update_layout(height=420, template="plotly_dark", paper_bgcolor="#0e1117",
                         plot_bgcolor="#0e1117", yaxis_title="Valore portafoglio",
                         margin=dict(l=20, r=20, t=35, b=20))
    st.plotly_chart(figure, use_container_width=True, theme=None)


st.set_page_config(page_title="Portfolio Simulation", layout="wide", page_icon="📊")
st.title("📊 Simulazione Portafoglio MMM")
st.caption("Backtest storico a soglie: non è una previsione e non costituisce consulenza finanziaria.")

with st.sidebar:
    st.header("Parametri simulazione")
    selected_tickers = st.multiselect("Ticker", TICKERS, default=["AAPL", "MSFT", "NVDA"], key="portfolio_tickers")
    capital = st.number_input("Capitale iniziale", min_value=1.0, value=10000.0, step=500.0)
    currency = st.selectbox("Valuta", ["EUR", "USD", "GBP", "CHF", "JPY"])
    benchmark_label = st.selectbox("ETF benchmark", list(BENCHMARKS), index=0)
    benchmark_ticker = BENCHMARKS[benchmark_label]
    period_label = st.selectbox("Arco temporale", list(PERIODS), index=3)
    allocation_mode = st.selectbox(
        "Distribuzione del capitale",
        ["Sleeve uguali per ticker", "Pesi market cap alla data iniziale"],
        help="La liquidità resta parcheggiata nella sleeve del ticker finché MMM non attiva l'acquisto.",
    )
    entry_threshold = st.slider("Soglia acquisto MMM", -100, 0, -75)
    exit_threshold = st.slider("Soglia vendita MMM", 0, 100, 70)
    commission_percent = st.number_input("Commissione per operazione (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.05)
    st.caption("Il capitale non investito resta parcheggiato; sono ammesse frazioni di azione.")
    run = st.button("Simula portafoglio", type="primary", use_container_width=True)
    st.caption(f"Benchmark selezionato: {benchmark_label}")

if not selected_tickers:
    st.info("Seleziona almeno un ticker nella barra laterale.")
    st.stop()

if run:
    try:
        with st.spinner("Scarico dati storici e calcolo i segnali MMM..."):
            selected_period = PERIODS[period_label]
            prices = download_prices(selected_tickers, DOWNLOAD_PERIODS[selected_period])
            window_prices = trim_prices(prices, selected_period)
            if not window_prices:
                raise ValueError("Nessun dato storico nell'intervallo selezionato.")
            start_date = min(frame.index[0] for frame in window_prices.values())
            if allocation_mode == "Pesi market cap alla data iniziale":
                allocation_weights, market_cap_details = historical_market_cap_weights(
                    prices, start_date
                )
            else:
                allocation_weights = {ticker: 1 / len(window_prices) for ticker in window_prices}
                market_cap_details = None
            benchmark_prices = trim_prices(
                download_prices([benchmark_ticker], DOWNLOAD_PERIODS[selected_period]),
                selected_period,
            )
            equity, summary, trades = simulate_portfolio(
                prices, capital, entry_threshold, exit_threshold,
                commission_rate=commission_percent / 100,
                start_date=start_date,
                allocation_weights=allocation_weights,
            )
            benchmark_frame = benchmark_prices.get(benchmark_ticker)
            if benchmark_frame is None or benchmark_frame.empty:
                st.warning(f"Dati non disponibili per il benchmark {benchmark_ticker}.")
            else:
                benchmark_value = benchmark_frame["Close"] / benchmark_frame["Close"].iloc[0] * capital
                equity["Benchmark"] = benchmark_value.reindex(equity.index).ffill()
                equity["Benchmark Return"] = equity["Benchmark"] / capital - 1
        if len(window_prices) != len(selected_tickers):
            missing = sorted(set(selected_tickers) - set(window_prices))
            st.warning(f"Dati non disponibili per: {', '.join(missing)}")
        st.session_state["portfolio_result"] = {
            "equity": equity, "summary": summary, "trades": trades,
            "capital": capital, "currency": currency, "period": period_label,
            "tickers": list(window_prices), "entry": entry_threshold, "exit": exit_threshold,
            "benchmark": benchmark_ticker,
            "benchmark_label": benchmark_label,
            "allocation_mode": allocation_mode,
            "allocation_weights": allocation_weights,
            "market_cap_details": market_cap_details,
        }
    except ValueError as exc:
        st.error(str(exc))

result = st.session_state.get("portfolio_result")
if not result:
    st.info("Imposta i parametri e premi **Simula portafoglio**.")
    st.stop()

summary = result["summary"].copy()
currency_metrics = summary["Metric"].isin(["Initial capital", "Final value", "Profit/Loss"])
# Keep numeric values in the engine; this copy is only a mixed-type display table.
summary["Value"] = summary["Value"].astype(object)
summary.loc[currency_metrics, "Value"] = summary.loc[currency_metrics, "Value"].map(
    lambda value: f"{value:,.2f} {result['currency']}"
)
st.dataframe(summary, hide_index=True, use_container_width=True)
equity = result["equity"]
portfolio_return = float(equity["Return"].iloc[-1] * 100)
benchmark_return = (float(equity["Benchmark Return"].dropna().iloc[-1] * 100)
                    if "Benchmark Return" in equity else None)
portfolio_profit = float(equity["Portfolio"].iloc[-1] - result["capital"])
benchmark_profit = (float(equity["Benchmark"].dropna().iloc[-1] - result["capital"])
                    if "Benchmark" in equity else None)
metric_columns = st.columns(4)
metric_columns[0].metric("Portafoglio finale", f"{portfolio_return:+.2f}%")
metric_columns[1].metric("Benchmark finale", f"{benchmark_return:+.2f}%" if benchmark_return is not None else "N/D")
metric_columns[2].metric("P/L portafoglio", f"{portfolio_profit:+,.2f} {result['currency']}")
metric_columns[3].metric("P/L benchmark", f"{benchmark_profit:+,.2f} {result['currency']}" if benchmark_profit is not None else "N/D")
if benchmark_return is not None:
    st.caption(f"Differenza rendimento vs benchmark: {portfolio_return - benchmark_return:+.2f} punti percentuali.")
selected_benchmark_label = result.get("benchmark_label", benchmark_label if "benchmark_label" in locals() else "ETF benchmark")
st.subheader("Allocazione iniziale")
allocation_table = pd.DataFrame({
    "Ticker": list(result["allocation_weights"]),
    "Peso %": [weight * 100 for weight in result["allocation_weights"].values()],
    "Capitale riservato": [result["capital"] * weight for weight in result["allocation_weights"].values()],
})
st.caption(f"{result.get('allocation_mode', 'Sleeve uguali per ticker')}. Il capitale riservato torna liquido alla vendita del ticker.")
st.dataframe(allocation_table, hide_index=True, use_container_width=True)
if result.get("market_cap_details") is not None:
    st.caption("Capitalizzazioni stimate alla prima seduta dell'intervallo selezionato.")
    st.dataframe(pd.DataFrame(result["market_cap_details"]), hide_index=True, use_container_width=True)
render_chart(equity, result["capital"], selected_benchmark_label)
st.caption("Il confronto usa rendimenti normalizzati sullo stesso capitale. La valuta della quota ETF può differire dalla valuta selezionata e l'effetto cambio non è convertito.")

trades = result["trades"]
if trades:
    trade_table = pd.DataFrame([trade.__dict__ for trade in trades])
    st.subheader("Ledger operazioni")
    st.dataframe(trade_table, hide_index=True, use_container_width=True)
else:
    st.info("Nessuna operazione ha rispettato entrambe le soglie nel periodo scelto.")

st.subheader("Analisi AI del backtest")
models = installed_models()
if models:
    model = st.selectbox("Modello locale", models, index=models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0)
    if st.button("Analizza risultato con Ollama"):
        serializable = {
            "summary": result["summary"].to_dict(orient="records"),
            "trades": [trade.__dict__ for trade in trades],
            "tickers": result["tickers"], "period": result["period"],
            "thresholds": {"entry": result["entry"], "exit": result["exit"]},
            "benchmark": {
                "ticker": result.get("benchmark", "non disponibile"),
                "label": result.get("benchmark_label", "ETF benchmark"),
                "return_percent": benchmark_return,
            },
            "allocation": {
                "mode": result.get("allocation_mode"),
                "weights": result.get("allocation_weights"),
                "market_caps_at_start": (
                    result["market_cap_details"].to_dict(orient="records")
                    if result.get("market_cap_details") is not None else None
                ),
            },
        }
        try:
            with st.spinner("Analizzo il backtest con Ollama..."):
                st.session_state["portfolio_ai"] = request_analysis(serializable, model)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            st.error(f"Analisi Ollama non disponibile: {exc}")
    if st.session_state.get("portfolio_ai"):
        st.markdown(st.session_state["portfolio_ai"])
else:
    st.info("Nessun modello Ollama disponibile. Installa un modello dalla pagina principale.")

st.caption("Fonte prezzi: Yahoo Finance tramite yfinance. Il backtest usa segnali a chiusura e esecuzione all'apertura successiva; costi, slippage, fiscalità e liquidità reale non sono modellati.")
