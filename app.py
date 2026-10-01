# -*- coding: utf-8 -*-
"""
Market Mood Meter (MMM) — App web locale
=========================================
Ricostruzione fedele alla DEFINIZIONE pubblica di Forecaster:
l'MMM è la media di tre indicatori (Advanced DPO, Wyckoff Causes/Effects, Speed),
ciascuno normalizzato in [-100, +100]. -100 = Panic Selling, +100 = FOMO.

NOTA DI ONESTA': le formule proprietarie esatte di Forecaster non sono pubbliche.
Questo modello riproduce la LOGICA dichiarata (detrend del prezzo, volume/prezzo alla
Wyckoff, momentum di velocita') e il comportamento qualitativo (estremi, divergenze,
crossing dello zero), non i decimali proprietari.

Avvio:  streamlit run app.py
"""

from datetime import datetime, timezone
import json
import shutil
import subprocess
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from altman import calculate_altman_z
from rag import (
    DEFAULT_EMBEDDING_MODEL,
    RAG_DIR,
    indexed_chunk_count,
    index_file,
    save_uploaded_file,
    search as search_rag,
)


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_DEFAULT_MODEL = "qwen3.5:4b"
AI_ANALYSIS_PROMPT = """Agisci come un Portfolio Manager istituzionale e un Analista Azionario Senior di Wall Street.

Analizza il titolo indicato usando esclusivamente il dossier dati fornito. Non inventare numeri,
fonti, notizie o stime. Se un dato non e' disponibile, scrivi "non disponibile". Distingui sempre
tra dati osservati, stime, calcoli del sistema e interpretazioni. Indica la data di riferimento
dei dati quando presente.

Segui esattamente queste sei sezioni, in italiano:
1. Identikit e Business Model (Il motore dei profitti)
2. Analisi Fondamentale e Salute Finanziaria
3. Contesto Settoriale e Macroeconomico
4. Analisi Tecnica e Momentum (Breve/Medio Termine)
5. Gestione del Rischio (Il Bear Case)
6. Verdetto dell'Esperto e Orizzonte Temporale

Nella sezione 6 indica il tipo di portafoglio e un verdetto Bullish, Bearish o Neutral per:
- breve termine: 1-6 mesi
- medio termine: 1-3 anni
- lungo termine: oltre 5 anni

Non dare consulenza personalizzata e non presentare il verdetto come certezza. Cita nella risposta
le fonti contenute nel dossier e nei documenti RAG, distinguendo fatti, inferenze e opinioni.
Se il contesto RAG non contiene la risposta, scrivi "non disponibile" e non inventare.
Segnala chiaramente i limiti della copertura dati.

DOSSIER DATI:
{dossier}

DOMANDA DELL'UTENTE:
{question}

CONTESTO DOCUMENTALE RAG:
{rag_context}
"""


@st.cache_resource
def ensure_ollama_service() -> bool:
    """Starts Ollama once for direct `streamlit run app.py` launches."""
    if shutil.which("ollama") is None:
        return False
    try:
        with urlopen("http://localhost:11434/api/tags", timeout=0.5):
            return True
    except Exception:
        pass

    terminal = shutil.which("x-terminal-emulator")
    command = (
        "ollama serve & server_pid=$!; "
        f"ollama pull {OLLAMA_DEFAULT_MODEL}; "
        "wait $server_pid; status=$?; "
        "echo; echo \"Ollama terminato (codice $status). Premi Invio per chiudere.\"; read -r"
    )
    try:
        if terminal:
            subprocess.Popen(
                [terminal, "-e", "bash", "-lc", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except OSError:
        return False


@st.cache_data(show_spinner=False, ttl=30)
def installed_ollama_models() -> list[str]:
    """Returns only models currently installed in the local Ollama registry."""
    try:
        with urlopen("http://localhost:11434/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = {
            item.get("name", "").strip()
            for item in payload.get("models", [])
            if item.get("name")
        }
        return sorted(names)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return []

# Define the list of available tickers extracted from README.md
AVAILABLE_TICKERS = [
    # Big 7 out of TESLA
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
    # DEFENSE & HEALTH
    "NOVO-B.CO", "UNH", "8766.T", "LULU", "MRNA", "PFIZER.NS", "DPZ",
    # Luxury
    "RACE",
    # Software
    "CRM", "ADBE",
    # CHINA MRVLCOME BACK
    "BIDU", "1211.HK", "BABA", "81810.HK", "JD",
    # RAM, MEMORY, HARDWARE FOR AI
    "005930.KS", "SKHY", "SNDK", "MU", "MRVL", "CGNX"
]

# ----------------------------------------------------------------------------
#  MOTORE DI CALCOLO MMM
# ----------------------------------------------------------------------------

def _clip100(s: pd.Series) -> pd.Series:
    return s.clip(-100, 100)


def _squash(sig: pd.Series, k: float) -> pd.Series:
    """Mappa in [-100,100] via tanh un segnale gia' centrato sullo ZERO strutturale
    (drift nullo / flusso neutro = 0). 'k' controlla quanto e' facile saturare."""
    return _clip100(100.0 * np.tanh(sig / k))


def _norm_z(series: pd.Series, win: int, sat: float = 1.1) -> pd.Series:
    """Z-score mobile riscalato in [-100,100] con saturazione tanh.
    Adatto SOLO a serie gia' centrate sullo zero (oscillatori detrended come il DPO):
    sottrae la media mobile, quindi misura lo scostamento dalla norma recente."""
    m = series.rolling(win, min_periods=max(2, win // 3)).mean()
    sd = series.rolling(win, min_periods=max(2, win // 3)).std(ddof=0)
    z = (series - m) / sd.replace(0, np.nan)
    z = z.fillna(0.0) / sat
    return _clip100(100.0 * np.tanh(z))


def advanced_dpo(close: pd.Series, period: int = 20, z_win: int = 100) -> pd.Series:
    """Advanced DPO — Detrended Price Oscillator (cicli di prezzo brevi).
    Prezzo meno SMA sfalsata, in % del prezzo (comparabile tra titoli). E' l'indicatore
    'reattivo': tende agli estremi durante pullback/rimbalzi. Serie centrata sullo
    zero -> z-score. NB: il 'Advanced DPO' proprietario di Forecaster non e' un DPO
    classico (confronto AAPL/ADBE: ordinamento invertito rispetto al detrend grezzo,
    non riproducibile da alcuna funzione monotona). 'sat=1.5' e' un compromesso tarato
    per bilanciare l'errore sui due titoli di riferimento noti."""
    shift = int(period / 2) + 1
    sma = close.rolling(period, min_periods=period // 2).mean()
    dpo = (close - sma.shift(shift)) / close * 100.0
    return _norm_z(dpo, z_win, sat=1.5)


def wyckoff_ce(close: pd.Series, high: pd.Series, low: pd.Series,
               volume: pd.Series, period: int = 20) -> pd.Series:
    """Wyckoff Causes/Effects — accumulazione/distribuzione via Chaikin Money Flow.
    Money-flow multiplier = dove chiude il prezzo nel range della barra, pesato per il
    volume. Centrato sullo ZERO: >0 accumulazione, <0 distribuzione. NIENTE z-score
    (il flusso non e' centrato a zero: lo z-score ne invertirebbe il segno)."""
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = (mfm * volume).fillna(0.0)
    cmf = (mfv.rolling(period, min_periods=period // 2).sum()
           / volume.rolling(period, min_periods=period // 2).sum())
    return _squash(cmf, 0.45)


def speed(close: pd.Series, period: int = 30, vol_win: int = 60) -> pd.Series:
    """Speed — velocita' direzionale (momentum) risk-adjusted.
    Drift su 'period' giorni diviso la volatilita' attesa: centrato sullo ZERO (drift
    positivo -> Speed positivo), comparabile tra titoli. NIENTE z-score: quello
    invertirebbe il segno quando il momentum si raffredda da un picco recente."""
    dret = close.pct_change()
    vol = dret.rolling(vol_win, min_periods=max(10, vol_win // 3)).std()
    sig = close.pct_change(period) / (vol * np.sqrt(period))
    return _squash(sig, 0.5)


def market_mood_meter(df: pd.DataFrame, dpo_p=20, wy_p=20, sp_p=30, z_win=100) -> pd.DataFrame:
    d = advanced_dpo(df["Close"], dpo_p, z_win)
    w = wyckoff_ce(df["Close"], df["High"], df["Low"], df["Volume"], wy_p)
    s = speed(df["Close"], sp_p)
    # media dei tre + lieve smussatura (come la linea MMM smooth di Forecaster)
    mmm = _clip100(((d + w + s) / 3.0).ewm(span=3, adjust=False).mean())
    return pd.DataFrame(
        {"Advanced_DPO": d, "Wyckoff": w, "Speed": s, "MMM": mmm}, index=df.index
    )


def detect_divergences(price: pd.Series, mmm: pd.Series, win: int = 20):
    """Divergenze semplici su finestra mobile: confronta pendenza prezzo vs MMM."""
    ps = price.diff(win)
    ms = mmm.diff(win)
    bull = (ps < 0) & (ms > 0) & (mmm < -30)   # prezzo giu', MMM su, in zona fear
    bear = (ps > 0) & (ms < 0) & (mmm > 30)    # prezzo su, MMM giu', in zona greed
    return bull, bear


def mood_label(v: float) -> tuple[str, str]:
    if v >= 75:  return "Extreme Greed (FOMO)", "#c0392b"
    if v >= 40:  return "Greed", "#e67e22"
    if v > -40:  return "Neutral", "#7f8c8d"
    if v > -75:  return "Fear", "#27ae60"
    return "Extreme Fear (Panic Selling)", "#145a32"


# ----------------------------------------------------------------------------
#  DATA
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=1800)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(period=period, auto_adjust=True)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# ----------------------------------------------------------------------------
#  QUOTE (riepilogo stile Yahoo Finance)
# ----------------------------------------------------------------------------

_CCY_SYM = {"USD": "$", "EUR": "€", "GBP": "£", "GBp": "p", "JPY": "¥",
            "CHF": "CHF ", "CAD": "C$", "AUD": "A$", "HKD": "HK$", "CNY": "¥"}


def _first(*vals):
    """Primo valore non-None e non-NaN."""
    for v in vals:
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return None


def _sym(ccy) -> str:
    return _CCY_SYM.get(ccy, (ccy + " ") if ccy else "")


def _fnum(x, dec=2):
    try:
        v = float(x)
        return "—" if np.isnan(v) else f"{v:,.{dec}f}"
    except (TypeError, ValueError):
        return "—"


def _fint(x):
    try:
        v = float(x)
        return "—" if np.isnan(v) else f"{int(v):,}"
    except (TypeError, ValueError):
        return "—"


def _fbig(x):
    """Grandi numeri: 1.23T / 45.60B / 789.00M / 12.30K."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(v):
        return "—"
    a = abs(v)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{v / div:,.2f}{suf}"
    return f"{v:,.0f}"


def _fdate(ts):
    """Da timestamp unix (s) o data a 'YYYY-MM-DD'."""
    if ts is None:
        return "—"
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return pd.to_datetime(ts).strftime("%Y-%m-%d")
    except Exception:
        return "—"


def _range(lo, hi, sym):
    if lo is None or hi is None:
        return "—"
    return f"{_fnum(lo)} - {_fnum(hi)} {sym}".strip()


def _bidask(v, size, sym):
    if v is None:
        return "—"
    s = f"{_fnum(v)} {sym}".strip()
    if size:
        try:
            s += f" x {int(size)}"
        except (TypeError, ValueError):
            pass
    return s


def _kv_table(pairs):
    rows = "".join(
        "<tr>"
        f"<td style='padding:7px 10px;color:#9aa5b1;border-bottom:1px solid #1f2530'>{k}</td>"
        f"<td style='padding:7px 10px;text-align:right;color:#e6e6e6;font-weight:600;"
        f"border-bottom:1px solid #1f2530'>{v}</td>"
        "</tr>"
        for k, v in pairs
    )
    return f"<table style='width:100%;border-collapse:collapse;font-size:0.9rem'>{rows}</table>"


@st.cache_data(show_spinner=False, ttl=900)
def load_quote(ticker: str) -> dict:
    """Riepilogo quotazione da Yahoo (via yfinance). Robusto a campi mancanti."""
    t = yf.Ticker(ticker)
    try:
        info = dict(t.info)
    except Exception:
        info = {}
    try:
        f = t.fast_info
        info["_fast"] = {k: getattr(f, k, None) for k in (
            "last_price", "previous_close", "open", "day_high", "day_low",
            "year_high", "year_low", "market_cap", "currency", "last_volume",
        )}
    except Exception:
        info["_fast"] = {}
    return info


def render_quote_panel(ticker: str, info: dict, df: pd.DataFrame) -> None:
    """Header (nome, prezzo, variazione, link Yahoo) + griglia metriche stile Yahoo."""
    fast = info.get("_fast", {}) or {}
    ccy = _first(info.get("currency"), fast.get("currency"))
    sym = _sym(ccy)
    yurl = f"https://it.finance.yahoo.com/quote/{quote_plus(ticker)}"

    price = _first(info.get("regularMarketPrice"), info.get("currentPrice"),
                   fast.get("last_price"), df["Close"].iloc[-1])
    prev = _first(info.get("regularMarketPreviousClose"), info.get("previousClose"),
                  fast.get("previous_close"),
                  df["Close"].iloc[-2] if len(df) > 1 else None)
    change = pct = None
    if price is not None and prev:
        change = float(price) - float(prev)
        pct = change / float(prev) * 100.0

    name = _first(info.get("longName"), info.get("shortName")) or ticker
    exch = _first(info.get("fullExchangeName"), info.get("exchange")) or ""

    h1, h2, h3 = st.columns([3, 2, 1.6])
    with h1:
        st.markdown(f"#### {name}")
        st.caption(" · ".join(x for x in (ticker, exch, ccy) if x))
    with h2:
        st.metric("Prezzo", f"{_fnum(price)} {sym}".strip(),
                  f"{change:+,.2f} ({pct:+.2f}%)" if change is not None else None)
    with h3:
        st.write("")
        st.link_button("🔗 Apri su Yahoo Finance", yurl, use_container_width=True)

    left = [
        ("Chiusura prec.", f"{_fnum(prev)} {sym}".strip()),
        ("Apertura", f"{_fnum(_first(info.get('regularMarketOpen'), info.get('open'), fast.get('open')))} {sym}".strip()),
        ("Bid", _bidask(info.get("bid"), info.get("bidSize"), sym)),
        ("Ask", _bidask(info.get("ask"), info.get("askSize"), sym)),
        ("Intervallo giorno", _range(_first(info.get("dayLow"), fast.get("day_low")),
                                      _first(info.get("dayHigh"), fast.get("day_high")), sym)),
        ("Intervallo 52 sett.", _range(_first(info.get("fiftyTwoWeekLow"), fast.get("year_low")),
                                        _first(info.get("fiftyTwoWeekHigh"), fast.get("year_high")), sym)),
        ("Volume", _fint(_first(info.get("volume"), info.get("regularMarketVolume"), fast.get("last_volume")))),
        ("Volume medio", _fint(_first(info.get("averageVolume"), info.get("averageDailyVolume10Day")))),
    ]

    # Dividendo & rendimento: calcolo il rendimento da rate/prezzo (non ambiguo).
    drate = _first(info.get("dividendRate"), info.get("trailingAnnualDividendRate"))
    div_txt = "—"
    if drate and price:
        div_txt = f"{_fnum(drate)} {sym} ({drate / float(price) * 100:.2f}%)".strip()
    elif info.get("trailingAnnualDividendYield"):
        div_txt = f"{info['trailingAnnualDividendYield'] * 100:.2f}%"
    elif info.get("dividendYield"):
        dy = info["dividendYield"]
        div_txt = f"{(dy * 100 if dy < 1 else dy):.2f}%"

    right = [
        ("Cap. di mercato", _fbig(_first(info.get("marketCap"), fast.get("market_cap")))),
        ("Beta (5A mens.)", _fnum(info.get("beta"))),
        ("Rapporto P/E (TTM)", _fnum(info.get("trailingPE"))),
        ("EPS (TTM)", f"{_fnum(info.get('trailingEps'))} {sym}".strip()),
        ("Data utili", _fdate(_first(info.get("earningsTimestamp"), info.get("earningsTimestampStart")))),
        ("Dividendo & rend.", div_txt),
        ("Data stacco div.", _fdate(info.get("exDividendDate"))),
        ("Stima target 1A", f"{_fnum(info.get('targetMeanPrice'))} {sym}".strip()),
    ]

    cL, cR = st.columns(2)
    cL.markdown(_kv_table(left), unsafe_allow_html=True)
    cR.markdown(_kv_table(right), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
#  GRAFICO PANORAMICA (prezzo con intervalli stile Yahoo)
# ----------------------------------------------------------------------------

_INTERVALS = {
    "1G":      ("1d",  "5m"),
    "1 sett.": ("5d",  "30m"),
    "1 mese":  ("1mo", "1d"),
    "3 mesi":  ("3mo", "1d"),
    "6 mesi":  ("6mo", "1d"),
    "1 anno":  ("1y",  "1d"),
    "5 anni":  ("5y",  "1wk"),
}


@st.cache_data(show_spinner=False, ttl=900)
def load_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Storico prezzi per il grafico panoramica (supporta l'intraday)."""
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df


def render_overview_chart(ticker: str) -> None:
    """Grafico prezzo con selettore d'intervallo (1G … 5 anni), stile Yahoo."""
    sel = st.segmented_control("Intervallo", list(_INTERVALS), default="6 mesi",
                               key="ov_interval") or "6 mesi"
    period, interval = _INTERVALS[sel]
    with st.spinner(f"Carico prezzi ({sel})…"):
        pdf = load_prices(ticker, period, interval)
    if pdf is None or pdf.empty:
        st.info(f"Nessun dato prezzo per l'intervallo {sel} (prova un intervallo più ampio).")
        return

    close = pdf["Close"]
    first, last = float(close.iloc[0]), float(close.iloc[-1])
    chg = last - first
    pct = chg / first * 100.0 if first else 0.0
    up = chg >= 0
    col = "#26a269" if up else "#e01b24"
    fill = "rgba(38,162,105,0.16)" if up else "rgba(224,27,36,0.16)"

    st.caption(f"Variazione **{sel}**: {'🟢' if up else '🔴'} {chg:+,.2f} ({pct:+.2f}%)")
    cfig = go.Figure()
    cfig.add_trace(go.Scatter(
        x=pdf.index, y=close, mode="lines", name="Prezzo",
        line=dict(color=col, width=1.8), fill="tozeroy", fillcolor=fill,
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:,.2f}<extra></extra>",
    ))
    dmin, dmax = float(close.min()), float(close.max())
    pad = (dmax - dmin) * 0.08 or (dmax * 0.02 or 1.0)
    cfig.update_layout(
        height=340, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e6e6e6",
        margin=dict(l=10, r=55, t=10, b=30), showlegend=False,
        yaxis=dict(range=[dmin - pad, dmax + pad], side="right", tickformat=",.2f"),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(cfig, use_container_width=True, theme=None)


# ----------------------------------------------------------------------------
#  OVERBOUGHT / OVERSOLD (gauge stile Forecaster)
# ----------------------------------------------------------------------------

def obos_state(v: float) -> tuple[str, str, str]:
    """Stessa logica di Forecaster: zone a ±50 sul valore MMM (media dei 3 componenti)."""
    if v >= 50:
        return "Overbought (FOMO)", "Il mercato è ipercomprato (euforia).", "#e74c3c"
    if v <= -50:
        return "Oversold (Panic Selling)", "Il mercato è ipervenduto (panico).", "#27ae60"
    return "Neutral", "Il mercato è stabile.", "#5b8def"


def _obos_gauge(value: float) -> go.Figure:
    g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        number=dict(font=dict(size=34, color="#e6e6e6")),
        gauge=dict(
            shape="angular",
            axis=dict(range=[-100, 100], tickmode="array",
                      tickvals=[-100, -50, 0, 50, 100],
                      ticktext=["-100", "-50", "0", "+50", "+100"],
                      tickcolor="#9aa5b1", tickfont=dict(color="#9aa5b1", size=12)),
            bar=dict(color="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[
                dict(range=[-100, -50], color="rgba(39,174,96,0.35)"),
                dict(range=[-50, 50],  color="rgba(91,141,239,0.30)"),
                dict(range=[50, 100],  color="rgba(231,76,60,0.35)"),
            ],
            threshold=dict(line=dict(color="#e6e6e6", width=5), thickness=0.9, value=value),
        ),
    ))
    g.update_layout(height=300, margin=dict(l=30, r=30, t=20, b=10),
                    paper_bgcolor="#0e1117", font_color="#e6e6e6")
    return g


def _comp_bar(value: float) -> str:
    pos = max(0.0, min(100.0, (value + 100) / 200 * 100))
    return (
        "<div style='position:relative;height:9px;border-radius:5px;margin-top:10px;"
        "background:linear-gradient(90deg,#27ae60 0%,#5b8def 50%,#e74c3c 100%)'>"
        f"<div style='position:absolute;left:{pos:.1f}%;top:-4px;width:3px;height:17px;"
        "background:#e6e6e6;transform:translateX(-50%);border-radius:2px;"
        "box-shadow:0 0 3px rgba(0,0,0,0.6)'></div></div>"
    )


def _comp_card(name: str, value: float) -> str:
    return (
        "<div style='background:#161b22;border:1px solid #1f2530;border-radius:12px;"
        "padding:14px 16px;text-align:center'>"
        f"<div style='color:#9aa5b1;font-size:0.9rem'>{name}</div>"
        f"<div style='color:#e6e6e6;font-size:1.7rem;font-weight:700;margin-top:2px'>{value:+.1f}</div>"
        f"{_comp_bar(value)}</div>"
    )


def _status_box(state: str, desc: str, color: str, value: float) -> str:
    return (
        "<div style='background:#161b22;border:1px solid #1f2530;border-radius:12px;"
        "padding:18px;text-align:center;margin-top:14px'>"
        "<div style='color:#9aa5b1;font-size:0.9rem'>Market Mood Meter</div>"
        f"<div style='color:{color};font-size:1.9rem;font-weight:800;margin:2px 0'>{state}</div>"
        f"<div style='color:#c9d1d9'>{desc} · valore <b>{value:+.1f}</b></div></div>"
    )


def render_obos_section(res: pd.DataFrame) -> None:
    st.subheader("🌡️ Overbought · Oversold")
    st.caption("Valutazione identica a Forecaster: media di Advanced DPO, Wyckoff e Speed → "
               "zone Oversold (≤ −50) · Neutral · Overbought (≥ +50).")
    mmm = float(res["MMM"].dropna().iloc[-1])
    dpo = float(res["Advanced_DPO"].dropna().iloc[-1])
    wyk = float(res["Wyckoff"].dropna().iloc[-1])
    spd = float(res["Speed"].dropna().iloc[-1])
    state, desc, color = obos_state(mmm)

    gcol, ccol = st.columns([1.05, 1.35])
    with gcol:
        st.plotly_chart(_obos_gauge(mmm), use_container_width=True, theme=None)
    with ccol:
        k1, k2, k3 = st.columns(3)
        k1.markdown(_comp_card("Advanced DPO", dpo), unsafe_allow_html=True)
        k2.markdown(_comp_card("Wyckoff", wyk), unsafe_allow_html=True)
        k3.markdown(_comp_card("Speed", spd), unsafe_allow_html=True)
        st.markdown(_status_box(state, desc, color, mmm), unsafe_allow_html=True)


def render_altman_section(ticker: str, info: dict) -> None:
    st.markdown("---")
    st.subheader("Salute finanziaria · Altman Z-score")
    st.caption(
        "Formula originale per società industriali quotate: capitale circolante, utili trattenuti, "
        "EBIT, valore di mercato del capitale e ricavi. Non è una probabilità di fallimento."
    )
    try:
        result = calculate_altman_z(ticker, info)
    except Exception as exc:
        st.warning(f"Z-score non disponibile: {exc}")
        return
    if not result.applicable:
        st.info(result.reason)
        return
    if result.score is None:
        st.warning(result.reason)
        if result.period:
            st.caption(f"Ultimo periodo bilancio considerato: {result.period}")
        return
    score_col, zone_col = st.columns(2)
    score_col.metric("Altman Z-score", f"{result.score:.2f}")
    zone_col.metric("Classificazione", result.zone)
    st.caption(
        f"Bilancio: {result.period or 'non disponibile'} · fonte automatica: {result.source}. "
        "Soglie originali: Safe > 2,99 · Grey 1,81–2,99 · Distress < 1,81."
    )
    with st.expander("Dettaglio dati usati"):
        st.json(result.variables)


def _num_or_none(value):
    try:
        number = float(value)
        return None if np.isnan(number) else number
    except (TypeError, ValueError):
        return None


def build_ai_dossier(ticker: str, period: str, df: pd.DataFrame,
                     info: dict, res: pd.DataFrame) -> dict:
    """Builds a dated, compact fact sheet for the local model."""
    fast = info.get("_fast", {}) or {}
    close = df["Close"]
    latest = float(close.iloc[-1])
    last_mmm = float(res["MMM"].dropna().iloc[-1])
    trend = {}
    for window in (20, 50, 200):
        average = close.rolling(window).mean().iloc[-1]
        trend[f"sma_{window}"] = _num_or_none(average)

    return {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "history_period": period,
        "market_data_source": "Yahoo Finance via yfinance; data may be delayed",
        "last_market_date": df.index[-1].strftime("%Y-%m-%d"),
        "price": latest,
        "price_currency": _first(info.get("currency"), fast.get("currency")),
        "market_cap": _num_or_none(_first(info.get("marketCap"), fast.get("market_cap"))),
        "trailing_pe": _num_or_none(info.get("trailingPE")),
        "forward_pe": _num_or_none(info.get("forwardPE")),
        "peg_ratio": _num_or_none(info.get("pegRatio")),
        "price_to_sales": _num_or_none(info.get("priceToSalesTrailing12Months")),
        "enterprise_to_ebitda": _num_or_none(info.get("enterpriseToEbitda")),
        "eps_ttm": _num_or_none(info.get("trailingEps")),
        "beta": _num_or_none(info.get("beta")),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "business_summary": info.get("longBusinessSummary"),
        "analyst_target_mean": _num_or_none(info.get("targetMeanPrice")),
        "earnings_date": _fdate(_first(info.get("earningsTimestamp"),
                                         info.get("earningsTimestampStart"))),
        "technical_snapshot": {
            "mmm": last_mmm,
            "mood": mood_label(last_mmm)[0],
            "advanced_dpo": float(res["Advanced_DPO"].dropna().iloc[-1]),
            "wyckoff": float(res["Wyckoff"].dropna().iloc[-1]),
            "speed": float(res["Speed"].dropna().iloc[-1]),
            "sma": trend,
        },
        "limitations": [
            "Le formule proprietarie Forecaster non sono disponibili.",
            "Il dossier non include automaticamente news live o filing SEC.",
            "Il sentiment istituzionale non viene trattato come dato real-time.",
        ],
    }


def request_ollama_analysis(dossier: dict, model: str, question: str = "",
                            rag_context: list[dict] | None = None) -> str:
    context = rag_context or []
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": AI_ANALYSIS_PROMPT.format(
                dossier=json.dumps(dossier, ensure_ascii=False, indent=2),
                question=question or "Nessuna domanda aggiuntiva.",
                rag_context=json.dumps(context, ensure_ascii=False, indent=2),
            ),
        }],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result.get("message", {}).get("content")
    if not content:
        raise ValueError("Ollama ha restituito una risposta vuota.")
    return content


def render_ai_analysis(ticker: str, period: str, df: pd.DataFrame,
                       info: dict, res: pd.DataFrame, model: str) -> None:
    st.subheader("Analisi AI locale")
    st.caption(f"Ollama · {model} · nessuna chiamata API a pagamento")
    question = st.text_area(
        "Domanda al modello",
        placeholder="Es. Quali rischi emergono dai documenti caricati per questo titolo?",
        height=90,
    )
    uploaded = st.file_uploader(
        "Documenti RAG",
        type=["txt", "md", "csv", "json", "pdf"],
        accept_multiple_files=True,
        help="I documenti vengono estratti e salvati localmente in rag_files/.",
    )
    embedding_model = st.text_input("Modello embedding Ollama", value=DEFAULT_EMBEDDING_MODEL)
    document_ticker = st.text_input("Metadato ticker documento", value=ticker)
    document_period = st.text_input("Metadato periodo", placeholder="Es. FY2025 o Q1 2026")
    document_source = st.text_input("Metadato fonte", placeholder="Es. Apple Investor Relations")
    document_date = st.text_input("Data documento", placeholder="YYYY-MM-DD")
    if uploaded and st.button("Salva e indicizza documenti", key="index_rag"):
        try:
            indexed = [save_uploaded_file(file) for file in uploaded]
            chunk_count = sum(
                index_file(path, embedding_model.strip() or DEFAULT_EMBEDDING_MODEL,
                           ticker=document_ticker, period=document_period,
                           source_name=document_source, document_date=document_date)
                for path in indexed
            )
            st.success(f"Indicizzati {len(indexed)} documenti ({chunk_count} chunk).")
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            st.error(f"Indicizzazione non riuscita: {exc}")

    st.caption("Filtri retrieval, lasciabili vuoti per cercare nell'intero archivio")
    filter_ticker = st.text_input("Filtra ticker", value=ticker, key="rag_filter_ticker")
    filter_period = st.text_input("Filtra periodo", key="rag_filter_period")
    filter_source = st.text_input("Filtra fonte", key="rag_filter_source")
    filter_date = st.text_input("Filtra data documento", key="rag_filter_date")

    request_key = f"{ticker}|{period}|{model}|{question}|{len(df)}|{float(res['MMM'].iloc[-1]):.4f}"
    stored = st.session_state.get("ai_analysis")

    if st.button("Genera analisi AI locale", type="primary", key="generate_ai_analysis"):
        dossier = build_ai_dossier(ticker, period, df, info, res)
        with st.spinner(f"Genero l'analisi di {ticker} con Ollama…"):
            try:
                rag_context = []
                if indexed_chunk_count() > 0:
                    rag_context = search_rag(
                        question or f"Analisi finanziaria di {ticker}",
                        embedding_model.strip() or DEFAULT_EMBEDDING_MODEL,
                        ticker=filter_ticker, period=filter_period,
                        source_name=filter_source, document_date=filter_date,
                    )
                content = request_ollama_analysis(dossier, model, question, rag_context)
                st.session_state["ai_analysis"] = {
                    "key": request_key,
                    "content": content,
                    "timestamp": dossier["analysis_timestamp_utc"],
                }
                stored = st.session_state["ai_analysis"]
            except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
                st.error(
                    "Analisi non riuscita. Verifica Ollama e i modelli configurati: "
                    f"`ollama pull {model}` per il modello generativo e "
                    f"`ollama pull {embedding_model.strip() or DEFAULT_EMBEDDING_MODEL}` "
                    f"per documenti RAG. Dettaglio: {exc}"
                )

    document_count = len({
        path.name for path in RAG_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".csv", ".json", ".pdf"}
    }) if RAG_DIR.exists() else 0
    st.caption(f"Archivio locale RAG: {document_count} documenti in {RAG_DIR.name}/")

    if stored and stored.get("key") == request_key:
        st.caption(f"Generata il {stored['timestamp']} · verifica sempre le fonti prima di usarla.")
        st.markdown(stored["content"])
    elif stored:
        st.info("Il report precedente appartiene a un altro ticker o a un altro set di parametri.")


# ----------------------------------------------------------------------------
#  UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Market Mood Meter", layout="wide", page_icon="📈")
ollama_available = ensure_ollama_service()
ollama_models = installed_ollama_models() if ollama_available else []

st.markdown(
    "<h1 style='margin-bottom:0'>📈 Analisi Titolo Azionario</h1>"
    "<p style='color:#7f8c8d;margin-top:2px'>Panoramica di mercato (Yahoo Finance) + Market Mood Meter · "
    "sentiment −100 (Panic) → +100 (FOMO)</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Parametri")
    ticker = st.selectbox("Ticker", options=AVAILABLE_TICKERS, index=0)
    period = st.selectbox("Periodo storico", ["1y", "2y", "3y", "5y", "10y", "max"], index=2)
    st.markdown("---")
    st.caption("Tuning indicatori")
    dpo_p = st.slider("DPO period", 10, 40, 20)
    wy_p  = st.slider("Wyckoff period", 10, 40, 20)
    sp_p  = st.slider("Speed period (momentum)", 10, 60, 30)
    z_win = st.slider("Finestra normalizzazione DPO", 40, 250, 100, step=10)
    show_components = st.checkbox("Mostra i 3 componenti", value=False)
    enable_ai = st.checkbox("Abilita analisi AI locale", value=bool(ollama_models),
                            disabled=not ollama_models)
    if st.button("Aggiorna modelli Ollama", use_container_width=True):
        installed_ollama_models.clear()
        st.rerun()
    if ollama_models:
        default_index = (ollama_models.index(OLLAMA_DEFAULT_MODEL)
                          if OLLAMA_DEFAULT_MODEL in ollama_models else 0)
        ollama_model = st.selectbox("Modello Ollama installato", ollama_models,
                                    index=default_index)
        st.caption(f"Modelli disponibili localmente: {len(ollama_models)}")
    else:
        ollama_model = ""
        st.warning("Nessun modello Ollama installato o servizio non raggiungibile.")
    run = st.button("Analizza", type="primary", use_container_width=True)
    if hasattr(st, "page_link"):
        st.page_link("pages/1_Portfolio_Simulation.py", label="Simulazione portafoglio", icon="📊")

if not run and "loaded" not in st.session_state:
    st.info("Seleziona un ticker nella barra laterale e premi **Analizza**.")
    st.stop()

st.session_state["loaded"] = True

with st.spinner(f"Scarico dati per {ticker}…"):
    df = load_data(ticker, period)

if df is None or df.empty:
    st.error(f"Nessun dato per '{ticker}'. Controlla il ticker (es. AAPL, MSFT, ENI.MI, BTC-USD).")
    st.stop()

# --- Panoramica quotazione (stile Yahoo Finance) ----------------------------
with st.spinner("Carico i dati di mercato…"):
    info = load_quote(ticker)
try:
    render_quote_panel(ticker, info, df)
except Exception as e:
    st.warning(f"Riepilogo quotazione non disponibile ora ({e}).")
    st.link_button("🔗 Apri su Yahoo Finance",
                   f"https://it.finance.yahoo.com/quote/{quote_plus(ticker)}")

render_altman_section(ticker, info)

render_overview_chart(ticker)

res = market_mood_meter(df, dpo_p=dpo_p, wy_p=wy_p, sp_p=sp_p, z_win=z_win)
bull, bear = detect_divergences(df["Close"], res["MMM"])

# --- Overbought / Oversold (gauge stile Forecaster) -------------------------
st.markdown("---")
render_obos_section(res)

if enable_ai and ollama_model:
    st.markdown("---")
    render_ai_analysis(ticker, period, df, info, res, ollama_model)

# --- Market Mood Meter (serie storica) --------------------------------------
st.markdown("---")
st.subheader("📊 Market Mood Meter")

# --- Stato corrente ---------------------------------------------------------
cur = float(res["MMM"].dropna().iloc[-1])
label, color = mood_label(cur)
c1, c2, c3, c4 = st.columns(4)
c1.metric("MMM attuale", f"{cur:+.1f}")
c2.markdown(f"<div style='padding-top:6px'><b>Stato:</b> "
            f"<span style='color:{color};font-weight:700'>{label}</span></div>", unsafe_allow_html=True)
c3.metric("Prezzo", f"{df['Close'].iloc[-1]:,.2f}")
c4.metric("Ultima data", df.index[-1].strftime("%Y-%m-%d"))

# --- Grafico principale (prezzo + volume) -----------------------------------
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
    vertical_spacing=0.03, subplot_titles=(f"{ticker} — Prezzo & Volume", "Market Mood Meter"),
)

# colore linea prezzo secondo il mood
mmm_al = res["MMM"].reindex(df.index)
seg_color = np.where(mmm_al > 40, "#e67e22", np.where(mmm_al < -40, "#27ae60", "#2e86de"))
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines",
                         line=dict(color="#2e86de", width=1.6), name="Close"), row=1, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color="rgba(39,174,96,0.55)",
                     name="Volume", yaxis="y3"), row=1, col=1)

# marker divergenze sul prezzo
fig.add_trace(go.Scatter(x=df.index[bull], y=df["Close"][bull], mode="markers",
                         marker=dict(symbol="triangle-up", size=9, color="#27ae60"),
                         name="Bullish div."), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index[bear], y=df["Close"][bear], mode="markers",
                         marker=dict(symbol="triangle-down", size=9, color="#c0392b"),
                         name="Bearish div."), row=1, col=1)

# --- Pannello MMM -----------------------------------------------------------
fig.add_trace(go.Scatter(x=res.index, y=res["MMM"], mode="lines",
                         line=dict(color="#eaecef", width=1.9), name="MMM"), row=2, col=1)
if show_components:
    for name, col in [("Advanced_DPO", "#b06ecc"), ("Wyckoff", "#1abc9c"), ("Speed", "#e67e22")]:
        fig.add_trace(go.Scatter(x=res.index, y=res[name], mode="lines",
                                 line=dict(width=0.9, color=col), opacity=0.6, name=name), row=2, col=1)

for lvl, col, dash in [(100, "#ff6b6b", "solid"), (50, "#ff6b6b", "dot"),
                       (0, "#9aa5b1", "solid"),
                       (-50, "#51cf66", "dot"), (-100, "#51cf66", "solid")]:
    fig.add_hline(y=lvl, line_color=col, line_width=1, line_dash=dash, row=2, col=1)

fig.update_yaxes(range=[-115, 115], row=2, col=1)
fig.update_layout(
    height=760, template="plotly_dark", bargap=0,
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e6e6e6",
    legend=dict(orientation="h", y=1.06, x=0),
    margin=dict(l=40, r=20, t=60, b=30),
    yaxis3=dict(overlaying="y", side="right", showgrid=False, range=[0, df["Volume"].max()*4]),
)
# theme=None: usiamo il nostro template scuro esplicito invece di quello di Streamlit,
# così il grafico è identico e leggibile a prescindere dal tema chiaro/scuro del sistema.
st.plotly_chart(fig, use_container_width=True, theme=None)

# --- Interpretazione --------------------------------------------------------
with st.expander("Come leggere il Market Mood Meter"):
    st.markdown(
        "- **> +50 / +100 (FOMO):** euforia, spesso vicino ai **massimi**. Zona di cautela / possibili top.\n"
        "- **< −50 / −100 (Panic Selling):** pessimismo estremo, spesso vicino ai **minimi**. Zona di opportunità.\n"
        "- **Divergenza** (prezzo nuovo minimo ma MMM no): esaurimento del trend → possibile inversione.\n"
        "- **Crossing dello zero** dopo una divergenza: alta probabilità che l'inversione sia confermata.\n"
        "- I livelli estremi passati del titolo tendono a fungere da supporto/resistenza del sentiment."
    )

st.caption(
    "Modello fedele alla definizione pubblica di Forecaster (media di DPO+Wyckoff+Speed, range −100/+100). "
    "Le formule proprietarie esatte non sono pubbliche: i valori qui sono una ricostruzione della logica, "
    "non i decimali ufficiali. Solo a scopo educativo, non è consulenza finanziaria."
)
