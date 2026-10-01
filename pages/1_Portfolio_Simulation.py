"""Interactive MMM threshold backtest page."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from portfolio_engine import download_prices, simulate_portfolio, trim_prices


OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"
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
calcoli e limiti del metodo. Spiega il contributo dei ticker, le operazioni generate dalle
soglie MMM, il rendimento, il drawdown e i rischi di overfitting. Non presentare il risultato
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


def request_analysis(result: dict, model: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PORTFOLIO_ANALYSIS_PROMPT.format(
            result=json.dumps(result, ensure_ascii=False, indent=2)
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


def render_chart(equity: pd.DataFrame, capital: float) -> None:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=equity.index, y=equity["Portfolio"], mode="lines",
                                name="Portafoglio", line=dict(color="#4dabf7", width=2)))
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
    period_label = st.selectbox("Arco temporale", list(PERIODS), index=3)
    entry_threshold = st.slider("Soglia acquisto MMM", -100, 0, -75)
    exit_threshold = st.slider("Soglia vendita MMM", 0, 100, 70)
    commission_percent = st.number_input("Commissione per operazione (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.05)
    st.caption("Il capitale viene ripartito equamente tra i ticker. Sono ammesse frazioni di azione.")
    run = st.button("Simula portafoglio", type="primary", use_container_width=True)

if not selected_tickers:
    st.info("Seleziona almeno un ticker nella barra laterale.")
    st.stop()

if run:
    try:
        with st.spinner("Scarico dati storici e calcolo i segnali MMM..."):
            selected_period = PERIODS[period_label]
            prices = download_prices(selected_tickers, DOWNLOAD_PERIODS[selected_period])
            prices = trim_prices(prices, selected_period)
            equity, summary, trades = simulate_portfolio(
                prices, capital, entry_threshold, exit_threshold,
                commission_rate=commission_percent / 100,
            )
        if len(prices) != len(selected_tickers):
            missing = sorted(set(selected_tickers) - set(prices))
            st.warning(f"Dati non disponibili per: {', '.join(missing)}")
        st.session_state["portfolio_result"] = {
            "equity": equity, "summary": summary, "trades": trades,
            "capital": capital, "currency": currency, "period": period_label,
            "tickers": list(prices), "entry": entry_threshold, "exit": exit_threshold,
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
render_chart(result["equity"], result["capital"])

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
