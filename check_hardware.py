#!/usr/bin/env python3
"""Diagnose local resources and recommend an Ollama model for financial analysis.

This script is intentionally dependency-free. It uses Linux procfs/free/nvidia-smi
when available and never downloads a model automatically.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass


OLLAMA_URL = "http://localhost:11434/api/tags"

# Approximate Q4/MXFP4 VRAM budgets. Keep headroom for the OS, Ollama and context.
MODELS = [
    ("qwen3.5:4b", 4.0, "best default for this app"),
    ("qwen3:8b", 6.0, "strong general reasoning"),
    ("qwen3.5:9b", 8.0, "stronger, slower"),
    ("mistral-nemo:12b", 9.0, "good multilingual alternative"),
    ("phi4:14b", 11.0, "compact reasoning model"),
    ("qwen3:14b", 11.0, "strong reasoning, slower"),
    ("deepseek-r1:14b", 11.0, "reasoning-focused, slower"),
    ("gpt-oss:20b", 16.0, "strong reasoning; MXFP4 build"),
    ("qwen3.5:27b", 20.0, "highest quality in this range"),
    ("gemma4:26b", 20.0, "multimodal/general alternative"),
    ("mistral-small3.2:24b", 20.0, "strong multilingual alternative"),
    ("deepseek-r1:32b", 23.0, "reasoning-focused; near the limit"),
    ("qwen3.5:35b", 28.0, "high-quality long-form analysis"),
    ("llama3.3:70b", 48.0, "large general-purpose model"),
    ("qwen3:72b", 50.0, "large reasoning model"),
    ("mistral-large:123b", 90.0, "large multilingual model"),
    ("gpt-oss:120b", 90.0, "large reasoning model; MXFP4 build"),
    ("qwen3:235b", 165.0, "mixture-of-experts, workstation/server class"),
    ("qwen3.5:122b", 86.0, "very large general-purpose model"),
    ("qwen3-coder:480b", 330.0, "near the upper 500 GB limit; coder-focused"),
]


@dataclass
class Hardware:
    ram_gb: float | None
    vram_gb: float | None
    gpu_name: str | None
    disk_free_gb: float | None


def run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def read_ram_gb() -> float | None:
    try:
        values = {}
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0])
        return values["MemTotal"] / 1024 / 1024
    except (OSError, KeyError, ValueError):
        return None


def read_gpu() -> tuple[str | None, float | None]:
    output = run([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return None, None
    name, memory = output.splitlines()[0].split(",", 1)
    try:
        return name.strip(), float(memory.strip()) / 1024
    except ValueError:
        return name.strip(), None


def read_disk_free_gb() -> float | None:
    try:
        return shutil.disk_usage(os.path.dirname(os.path.abspath(__file__))).free / 1024**3
    except OSError:
        return None


def ollama_models() -> set[str]:
    try:
        with urllib.request.urlopen(OLLAMA_URL, timeout=2) as response:
            payload = json.load(response)
        return {item.get("name", "") for item in payload.get("models", [])}
    except (OSError, ValueError):
        return set()


def format_value(value: float | None, unit: str = "GB") -> str:
    return f"{value:.1f} {unit}" if value is not None else "non disponibile"


def recommend(hardware: Hardware) -> tuple[str, float, str]:
    # CPU-only machines use a smaller model to avoid unusable latency.
    budget = hardware.vram_gb if hardware.vram_gb is not None else min(hardware.ram_gb or 8.0, 8.0)
    usable = max(3.0, budget - 1.5)
    candidates = [model for model in MODELS if model[1] <= usable]
    return candidates[-1] if candidates else MODELS[0]


def main() -> int:
    gpu_name, vram_gb = read_gpu()
    hardware = Hardware(
        ram_gb=read_ram_gb(),
        vram_gb=vram_gb,
        gpu_name=gpu_name,
        disk_free_gb=read_disk_free_gb(),
    )
    model, estimated_vram, note = recommend(hardware)
    installed = ollama_models()
    ollama_installed = shutil.which("ollama") is not None
    ollama_online = bool(installed)

    print("=== Forecaster.biz: controllo hardware AI locale ===")
    print(f"RAM:              {format_value(hardware.ram_gb)}")
    print(f"GPU:              {hardware.gpu_name or 'non rilevata'}")
    print(f"VRAM:             {format_value(hardware.vram_gb)}")
    print(f"Spazio libero:    {format_value(hardware.disk_free_gb)}")
    print(f"Ollama installato: {'si' if ollama_installed else 'no'}")
    print(f"Ollama online:     {'si' if ollama_online else 'no'}")
    print()
    print(f"Modello consigliato: {model}")
    print(f"VRAM stimata:        circa {estimated_vram:.1f} GB, {note}")
    if model in installed:
        print("Stato modello:       gia' installato")
    else:
        print(f"Per installarlo:     ollama pull {model}")
    if hardware.disk_free_gb is not None and hardware.disk_free_gb < estimated_vram * 1.3:
        print("AVVISO: lo spazio libero potrebbe non bastare per scaricare il modello.")
    print("Nota: i consumi reali dipendono da quantizzazione, contesto e GPU condivisa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
