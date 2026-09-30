# Analisi Titolo Azionario — App locale

Inserisci un ticker (es. `AAPL`, `MSFT`, `ENI.MI`, `BTC-USD`) e ottieni:

1. **Panoramica di mercato (stile Yahoo Finance):** prezzo e variazione, chiusura
   precedente, apertura, bid/ask, intervallo giorno e 52 settimane, volume e volume
   medio, capitalizzazione, beta, P/E (TTM), EPS (TTM), data utili, dividendo e
   rendimento, stacco dividendo, stima target a 1 anno — più un pulsante
   **🔗 Apri su Yahoo Finance** che apre la pagina ufficiale del ticker.
   Include un **grafico prezzo** con selettore d'intervallo
   (`1G`, `1 sett.`, `1 mese`, `3 mesi`, `6 mesi`, `1 anno`, `5 anni`).
2. **Overbought · Oversold:** il gauge di Forecaster (media di Advanced DPO,
   Wyckoff, Speed) con zone **Oversold (≤ −50) · Neutral · Overbought (≥ +50)**,
   i tre componenti e lo stato corrente.
3. **Market Mood Meter:** grafico prezzo/volume + il sentiment da −100 (Panic) a
   +100 (FOMO), con marker di divergenza e linee dei livelli estremi.

I dati provengono da Yahoo Finance tramite `yfinance`.

## Avvio rapido

```bash
./run.sh
```

Al primo avvio crea un virtualenv e installa le dipendenze; poi apre l'app
nel browser (di default http://localhost:8501).

## Avvio manuale

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Come funziona il modello

L'MMM è la **media di tre indicatori**, ciascuno normalizzato in [−100, +100]:

| Componente     | Cosa misura                                          | Mappatura            |
|----------------|------------------------------------------------------|----------------------|
| Advanced DPO   | Cicli di prezzo di breve (detrended); reattivo ai pullback | z-score (centrato) |
| Wyckoff C/E    | Accumulazione / distribuzione via Chaikin Money Flow | assoluta (tanh)      |
| Speed          | Momentum direzionale risk-adjusted (drift / volatilità) | assoluta (tanh)   |

## Documenti locali e RAG

L'app include un archivio RAG locale per porre domande sui propri documenti di
testo insieme ai dati di mercato. Sono supportati `TXT`, `Markdown`, `CSV` e
`JSON`. I file caricati dalla pagina vengono salvati in `rag_files/`, divisi in
chunk e convertiti in embedding da Ollama. La domanda viene convertita nello
stesso spazio vettoriale, i chunk più simili vengono recuperati e passati al
modello generativo nel prompt.

Sono necessari due modelli distinti:

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:4b
```

Il progetto usa SQLite come vector store embedded: per un archivio locale
singolo è sufficiente, non richiede un servizio aggiuntivo e mantiene i dati
offline. Chroma o Qdrant diventano preferibili quando aumentano molto il numero
di documenti, gli utenti concorrenti o servono filtri e deployment separati.
L'LLM non recupera autonomamente i documenti: il retrieval è eseguito dal
vector store e il modello riceve soltanto i chunk selezionati.

I documenti e `rag_files/index.sqlite3` sono esclusi da Git per evitare di
pubblicare dati personali o proprietari. L'app è uno strumento di ricerca e
sintesi, non un consulente finanziario e non garantisce completezza o
correttezza delle fonti.


## Nome e indipendenza del progetto

Il repository pubblico usa il nome **Market Mood RAG** (`market-mood-rag`). È
un progetto indipendente e non è affiliato a Forecaster.biz. Il nome originale
è stato rimosso dal branding del progetto per ridurre il rischio di confusione
con una piattaforma finanziaria esistente. Questa scelta non costituisce un
parere legale: prima di una distribuzione commerciale è opportuno verificare
marchi, copyright, termini d'uso e la giurisdizione applicabile con un
professionista.



**Nota chiave sulla mappatura.** DPO è un oscillatore già *centrato sullo zero*
(prezzo − media sfalsata), quindi si normalizza con uno z-score mobile. **Speed e
Wyckoff invece NON sono centrati sullo zero** (in un trend rialzista il momentum e il
money-flow sono strutturalmente positivi): usare uno z-score li invertirebbe di segno
appena si raffreddano da un picco. Per questo sono mappati in modo *assoluto*
(`tanh` di un segnale centrato sullo zero strutturale), così restano positivi finché
il trend/accumulazione è positivo. Senza questa distinzione il modello dava
erroneamente *Oversold* per titoli in uptrend con un pullback recente (es. AAPL).

**Onestà sul reverse engineering:** Forecaster non pubblica le formule esatte dei tre
indicatori proprietari. Il modello riproduce lo **stato** (Neutral / Overbought /
Oversold) dei reading di riferimento noti:

| Titolo | Reale (DPO / Wyck / Speed → stato) | Modello (DPO / Wyck / Speed → stato) |
|--------|-----------------------------------|--------------------------------------|
| AAPL   | −97 / +40 / +56 → **Neutral**     | −77 / +45 / +59 → **Neutral** ✓      |
| ADBE   | +45 / +39 / +119 → **Overbought** | +70 / +40 / +99 → **Overbought** ✓   |

**Limite noto sul DPO.** Il "Advanced DPO" di Forecaster **non è** un DPO classico:
per AAPL il detrend grezzo è −5.4% (−0.9σ) ma il reading è −97 (estremo), mentre per
ADBE è +12.8% (+1.4σ) ma il reading è solo +45 — un *ordinamento invertito* rispetto
al detrend, non riproducibile da **nessuna** normalizzazione monotona (z-score, ATR,
min-max o smussatura: tutte verificate). Probabilmente è multi-periodo o combina più
segnali (da qui "Advanced"). La saturazione DPO (`sat=1.5`) è quindi tarata come
compromesso tra i due riferimenti; lo **stato** dell'MMM resta corretto in entrambi.
Wyckoff (CMF) e Speed (momentum risk-adjusted) invece combaciano da vicino. I
parametri sono regolabili dalla barra laterale.

Solo a scopo educativo. Non è consulenza finanziaria.




## Analisi AI locale opzionale

La pagina include un'opzione per generare un report finanziario con un modello
locale tramite Ollama. L'opzione è disattivabile dalla barra laterale e non
aggiunge chiamate API a pagamento né dipendenze Python obbligatorie.

1. Installa Ollama da [ollama.com](https://ollama.com/download).
2. Avvia il servizio Ollama.
3. Scarica un modello, ad esempio:

```bash
ollama pull qwen3.5:4b
```

4. Avvia l'app con `./run.sh`, abilita **Analisi AI locale** e premi
   **Genera analisi AI locale**.

Il ticker selezionato viene inserito automaticamente nel dossier inviato al
modello insieme ai dati disponibili nell'app, alla data dell'analisi e ai
limiti della ricostruzione MMM. Ollama non aggiorna autonomamente i dati
finanziari: il report è quindi una sintesi dei dati raccolti dall'app e non
costituisce consulenza finanziaria. Il modello predefinito 4B è scelto per
hardware con circa 6 GB di VRAM; `qwen3.5:9b` resta selezionabile manualmente
su macchine più potenti.

Il catalogo completo dei modelli Ollama, organizzato per requisiti VRAM fino a
500 GB, include anche Kimi K3 e i modelli Xiaomi MiMo:
[LOCAL_MODELS.md](LOCAL_MODELS.md).

## Lista Ticker Azioni Portafoglio Azionario

| Ticker | Nome Azienda |


# Big 7 out of TESLA
|  AAPL | Apple Inc. |
|  MSFT | Microsoft Corporation |
|  AMZN | Amazon.com Inc. |
|  GOOGL | Alphabet Inc. Class C |
|  META  | Meta Platforms, Inc. |
|  NVDA | NVIDIA Corporation |

# DEFENSE  & HEALTH
|  NOVO-B.CO  | Novo Nordisk B.V. |  
|  UNH  | UnitedHealth Group Inc. | 
|  8766.T | Tokio Marine Holdings, Inc. (Tokyo Stock Exchange; official ticker: 8766) |
|  LULU | Lululemon Athletica Inc. |
|  DPZ | Domino's Pizza & Restaurant Company |

# Luxury
|  RACE | Ferrari |

# Software
|  CRM   | Salesforce, Inc. |
|  ADBE | Adobe Inc. |

# CHINA MRVLCOME BACK
|  BIDU   | Baidu Inc. |
|  1211.HK  | BYD Ord Shs A |
|  BABA  | Alibaba Group Holding Limited |
|  81810.HK | Xiaomi Corporation |
|  JD | JD.com, Inc. (NASDAQ/GDR) |

# RAM, MEMORY, HARDWARE FOR AI 
|  005930.KS | Samsung Electronics Co., Ltd. |
|  SKHY  | Skhynix Holdings Co., Ltd. |
|  SNDK  | SanDisk Corporation |
|  MU    | Micron Technology, Inc. |
|  MRVL  | Marvell Technology, Inc. |