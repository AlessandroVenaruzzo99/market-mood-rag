#!/usr/bin/env bash
# Avvia l'app Market Mood Meter nel browser.
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "Creo ambiente virtuale…"
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
fi

./.venv/bin/pip install -r requirements.txt -q

python3 check_hardware.py
echo

if command -v ollama >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1 && curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "Ollama è già attivo."
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    echo "Avvio Ollama in un terminale separato…"
    x-terminal-emulator -e bash -lc \
      'ollama serve & server_pid=$!; ollama pull qwen3.5:4b; wait $server_pid; status=$?; echo; echo "Ollama terminato (codice $status). Premi Invio per chiudere."; read -r' \
      >/dev/null 2>&1 &
  else
    echo "Nessun terminale separato disponibile: avvio Ollama in background."
    ollama serve >.ollama.log 2>&1 &
  fi
else
  echo "Avviso: Ollama non è installato. L’app si avvierà senza analisi AI locale."
fi

exec ./.venv/bin/streamlit run app.py
