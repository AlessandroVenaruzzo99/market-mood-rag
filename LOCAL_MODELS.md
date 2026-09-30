# Modelli Ollama per l'analisi finanziaria locale

Il catalogo seguente è una guida pratica per macchine con **fino a circa 500 GB di VRAM complessiva**. Le cifre sono stime conservative per versioni quantizzate disponibili in Ollama; il consumo effettivo cambia con quantizzazione, lunghezza del contesto, immagini e memoria condivisa. È necessario lasciare margine per il sistema operativo, Ollama e il contesto della richiesta.

Il controllo automatico si esegue con:

```bash
python3 check_hardware.py
```

Lo script rileva RAM, GPU/VRAM, spazio disco e modelli già installati, poi propone una scelta e stampa il relativo comando `ollama pull`. Non scarica modelli automaticamente.

## Catalogo consigliato fino a 24 GB di VRAM

| Modello Ollama | VRAM indicativa | Profilo | Comando |
|---|---:|---|---|
| `qwen3.5:4b` | ~4 GB | Scelta predefinita per il progetto; veloce e multilingue | `ollama pull qwen3.5:4b` |
| `qwen3:8b` | ~6 GB | Ragionamento generale | `ollama pull qwen3:8b` |
| `qwen3.5:9b` | ~8 GB | Qualità superiore, più lento | `ollama pull qwen3.5:9b` |
| `mistral-nemo:12b` | ~9 GB | Alternativa multilingue | `ollama pull mistral-nemo:12b` |
| `phi4:14b` | ~11 GB | Ragionamento compatto | `ollama pull phi4:14b` |
| `qwen3:14b` | ~11 GB | Ragionamento più robusto | `ollama pull qwen3:14b` |
| `deepseek-r1:14b` | ~11 GB | Specializzato nel ragionamento; più lento | `ollama pull deepseek-r1:14b` |
| `gpt-oss:20b` | ~16 GB | Ragionamento forte, build MXFP4 | `ollama pull gpt-oss:20b` |
| `qwen3.5:27b` | ~20 GB | Qualità massima consigliata in questa fascia | `ollama pull qwen3.5:27b` |
| `gemma4:26b` | ~20 GB | Modello generale/multimodale | `ollama pull gemma4:26b` |
| `mistral-small3.2:24b` | ~20 GB | Alternativa forte e multilingue | `ollama pull mistral-small3.2:24b` |
| `deepseek-r1:32b` | ~23 GB | Ragionamento intenso; vicino al limite | `ollama pull deepseek-r1:32b` |
| `qwen3.5:35b` | ~28 GB | Analisi lunga di qualità elevata | `ollama pull qwen3.5:35b` |
| `llama3.3:70b` | ~48 GB | Modello generale di grandi dimensioni | `ollama pull llama3.3:70b` |
| `qwen3:72b` | ~50 GB | Ragionamento di grandi dimensioni | `ollama pull qwen3:72b` |
| `qwen3.5:122b` | ~86 GB | Modello generale molto grande | `ollama pull qwen3.5:122b` |
| `mistral-large:123b` | ~90 GB | Modello multilingue molto grande | `ollama pull mistral-large:123b` |
| `gpt-oss:120b` | ~90 GB | Ragionamento grande, build MXFP4 | `ollama pull gpt-oss:120b` |
| `qwen3:235b` | ~165 GB | Mixture-of-experts, classe workstation/server | `ollama pull qwen3:235b` |
| `qwen3-coder:480b` | ~330 GB | Vicino al limite; orientato al coding, non ideale per finanza | `ollama pull qwen3-coder:480b` |

Questi sono i modelli pertinenti alla generazione di report testuali. Modelli vision-only, embedding e coder non sono inclusi perché non migliorano il compito principale dell'app.

## Regola pratica

- **4-6 GB VRAM:** `qwen3.5:4b`
- **6-8 GB VRAM:** `qwen3:8b` oppure `qwen3.5:9b` se resta memoria sufficiente
- **10-12 GB VRAM:** `phi4:14b`, `qwen3:14b` o `deepseek-r1:14b`
- **16-20 GB VRAM:** `gpt-oss:20b` oppure `qwen3.5:27b`
- **20-24 GB VRAM:** `mistral-small3.2:24b` o `deepseek-r1:32b`
- **24-32 GB VRAM:** `qwen3.5:35b`
- **40-64 GB VRAM:** `llama3.3:70b` o `qwen3:72b`
- **80-100 GB VRAM:** `qwen3.5:122b`, `mistral-large:123b` o `gpt-oss:120b`
- **150-200 GB VRAM:** `qwen3:235b`
- **300-500 GB VRAM:** `qwen3-coder:480b`, solo se il compito include anche coding; per l'analisi finanziaria è generalmente più sensato usare un 120B-235B di qualità.

Per il tuo progetto il criterio principale è la latenza: un 4B o 8B che risponde in modo stabile è preferibile a un 27B che esaurisce VRAM e usa pesantemente la CPU. Oltre 24 GB è importante avere anche RAM di sistema adeguata, alimentazione e raffreddamento; con più GPU la VRAM deve essere realmente disponibile a Ollama, non solo sommata nominalmente.

## Fonti e verifica

- [Libreria ufficiale Ollama](https://ollama.com/library)
- [Qwen3.5](https://ollama.com/library/qwen3.5)
- [Qwen3](https://ollama.com/library/qwen3)
- [Gemma4](https://ollama.com/library/gemma4)
- [Mistral Small](https://ollama.com/library/mistral-small3.2)
- [GPT-OSS](https://ollama.com/library/gpt-oss)
- [DeepSeek-R1](https://ollama.com/library/deepseek-r1)

Le dimensioni e i requisiti possono cambiare quando Ollama pubblica nuovi tag o quantizzazioni. Prima del download è consigliabile controllare il tag specifico con `ollama show nome-modello` oppure nella pagina ufficiale del modello.

## Inventario attuale delle famiglie Ollama

Questo è l'inventario delle famiglie esposte dalla libreria Ollama consultata il
29 settembre 2026. Non tutti i modelli sono adatti all'analisi finanziaria e non
tutti sono realmente open source: alcuni sono open-weight, cloud-only, vision,
embedding, safety o specializzati nel coding.

```text
alfred, all-minilm, athene-v2, aya, aya-expanse, bakllava,
bespoke-minicheck, bge-large, bge-m3, codebooga, codegeex4, codegemma,
codellama, codeqwen, codestral, codeup, cogito, cogito-2.1, command-a,
command-r, command-r7b, command-r7b-arabic, command-r-plus, dbrx, deepcoder,
deepscaler, deepseek-coder, deepseek-coder-v2, deepseek-llm, deepseek-ocr,
deepseek-r1, deepseek-v2, deepseek-v2.5, deepseek-v3, deepseek-v3.1,
deepseek-v4.1-flash, deepseek-v4-pro, devstral, devstral-2, devstral-small-2,
dolphin3, dolphincoder, dolphin-llama3, dolphin-mistral, dolphin-mixtral,
dolphin-phi, duckdb-nsql, embeddinggemma, everythinglm, exaone3.5,
exaone-deep, falcon, falcon2, falcon3, firefunction-v2, functiongemma, gemma,
gemma2, gemma3, gemma3n, gemma4, glm4, glm-4.7-flash, glm-5.2, glm-5.3,
glm-5.3-flash, glm-ocr, goliath, gpt-oss, gpt-oss-safeguard,
granite3.1-dense, granite3.1-moe, granite3.2, granite3.2-vision, granite3.3,
granite3-dense, granite3-guardian, granite3-moe, granite4, granite4.1,
granite4.1-guardian, granite4.2, granite-code, granite-embedding, hermes3,
internlm2, kimi-k2.6, kimi-k2.7-code, kimi-k3, laguna-s-2.1, laguna-xs.2,
laguna-xs-2.1, lfm2, lfm2.5, lfm2.5-thinking, llama2, llama2-chinese,
llama2-uncensored, llama3, llama3.1, llama3.2, llama3.2-vision, llama3.3,
llama3-chatqa, llama3-gradient, llama3-groq-tool-use, llama4, llama-guard3,
llama-pro, llava, llava-llama3, llava-phi3, magicoder, magistral, marco-o1,
mathstral, medgemma, medgemma1.5, meditron, medllama2, megadolphin,
minicpm-v, minicpm-v4.5, minicpm-v4.6, minimax-m2.7, minimax-m3, ministral-3,
mistral, mistral-large, mistral-large-3, mistrallite, mistral-medium-3.5,
mistral-nemo, mistral-openorca, mistral-small, mistral-small3.1,
mistral-small3.2, mixtral, moondream, muse-glimmer, mxbai-embed-large,
nemotron, nemotron3, nemotron-3.5-lightning, nemotron-3-nano,
nemotron-3-super, nemotron-3-ultra, nemotron-cascade-2, nemotron-mini,
neural-chat, nexusraven, nimble, nomic-embed-text, nomic-embed-text-v2-moe,
north-mini-code-1.0, notus, notux, nous-hermes, nous-hermes2,
nous-hermes2-mixtral, nuextract, olmo2, olmo-3, olmo-3.1, openchat,
opencoder, openhermes, open-orca-platypus2, openthinker, orca2, orca-mini,
ornith, ornith-1.5, paraphrase-multilingual, phi, phi3, phi3.5, phi4,
phi4-mini, phi4-mini-reasoning, phi4-reasoning, phind-codellama, qwen,
qwen2, qwen2.5, qwen2.5-coder, qwen2.5vl, qwen2-math, qwen3, qwen3.5,
qwen3.6, qwen3.8, qwen3.8-flash-next, qwen3-coder, qwen3-coder-next,
qwen3-embedding, qwen3-next, qwen3-vl, qwq, r1-1776, reader-lm, reflection,
rnj-1, sailor2, samantha-mistral, shieldgemma, smallthinker, smollm, smollm2,
snowflake-arctic-embed, snowflake-arctic-embed2, solar, solar-pro, sqlcoder,
stable-beluga, stable-code, stablelm2, stablelm-zephyr, starcoder, starcoder2,
starling-lm, tev1, tinydolphin, tinyllama, translategemma, tulu3, vicuna,
wizardcoder, wizardlm, wizardlm2, wizardlm-uncensored, wizard-math,
wizard-vicuna, wizard-vicuna-uncensored, xwinlm, yarn-llama2, yarn-mistral,
yi, yi-coder, zephyr
```

L'inventario include le famiglie, non ogni singolo tag dimensionale. Per vedere
le varianti realmente disponibili di una famiglia usare:

```bash
ollama show qwen3.5
ollama show gemma4
ollama show kimi-k3
```

## Kimi K3

`kimi-k3` è presente nella libreria Ollama come `kimi-k3:cloud`, con contesto da
1M token, ma la pagina ufficiale lo presenta come cloud-only. Quindi non è un
modello locale scaricabile con i normali tag di Ollama e richiede il servizio
cloud/autenticazione Ollama. Non va quindi proposto come alternativa gratuita
locale alla generazione dell'app.

```bash
ollama run kimi-k3:cloud
```

## Modelli Xiaomi MiMo

Le famiglie Xiaomi MiMo non risultano nella libreria ufficiale principale di
Ollama. Xiaomi pubblica però i checkpoint su Hugging Face e ModelScope; il
repository ufficiale documenta MiMo-7B-Base, MiMo-7B-SFT, MiMo-7B-RL-Zero e
MiMo-7B-RL. Esistono inoltre community models Ollama ricercabili con questi
comandi:

```bash
ollama pull alibayram/mimo-7b-rl
ollama pull XiaomiMiMo/MiMo-VL-7B-RL
ollama pull XiaomiMiMo/MiMo-VL-7B-SFT
ollama pull petronetto/mimo-v2.6-9b
```

Il primo è un community tag per MiMo-7B-RL, i due `MiMo-VL` sono varianti
vision, mentre `petronetto/mimo-v2.6-9b` è una conversione community del
checkpoint Distill-Qwen-9B. Questi tag possono essere rimossi, aggiornati o
non verificati da Ollama: controllare sempre la pagina del tag prima dell'uso.
Per l'app finanziaria consiglio un modello generale ufficiale, come
`qwen3.5:4b`, `qwen3.5:9b`, `gemma4:12b` o `gpt-oss:20b`, invece di un MiMo
orientato soprattutto a matematica, codice o visione.

Fonti Xiaomi:

- [Xiaomi MiMo su GitHub](https://github.com/XiaomiMiMo/MiMo)
- [Organizzazione XiaomiMiMo su Hugging Face](https://huggingface.co/XiaomiMiMo)
- [Ricerca Xiaomi su Ollama](https://ollama.com/search?q=xiaomi)

## Catalogo completo Ollama e requisiti hardware

La libreria Ollama contiene famiglie con più tag dimensionali. Le liste sotto
riportano tutte le famiglie rilevate il 29 settembre 2026; il requisito hardware
si riferisce alla **variante locale più grande normalmente disponibile** della
famiglia. Per una scelta precisa usa il tag completo con `ollama show TAG`.

Le stime VRAM sono conservative per quantizzazioni comuni e contesto moderato:

| Fascia | Hardware consigliato | Famiglie/modelli locali compatibili |
|---|---|---|
| Fino a 4 GB VRAM | 8-16 GB RAM | `all-minilm`, `embeddinggemma`, `functiongemma`, `granite3-moe`, `granite3.1-moe`, `granite3.1-dense`, `granite3-dense`, `granite3.2`, `granite3.3`, `granite4`, `lfm2.5-thinking`, `medgemma1.5`, `nemotron-3-nano`, `phi`, `phi3`, `phi3.5`, `phi4-mini`, `smollm`, `smollm2`, `stablelm2`, `tev1`, `tinyllama`, `tinydolphin` |
| 4-8 GB VRAM | 16 GB RAM | `aya`, `aya-expanse`, `bge-large`, `bge-m3`, `codegemma`, `codeqwen`, `dolphin3`, `falcon3`, `gemma`, `gemma2`, `gemma3`, `gemma3n`, `granite3.2-vision`, `granite4.1`, `granite4.1-guardian`, `hermes3`, `llama2`, `llama3`, `llama3.1`, `llama3.2`, `llama3-chatqa`, `llava`, `llava-llama3`, `llava-phi3`, `minicpm-v`, `mistral`, `mistral-nemo`, `mistral-openorca`, `moondream`, `nemotron-mini`, `neural-chat`, `nexusraven`, `nomic-embed-text`, `openchat`, `openhermes`, `opencoder`, `openthinker`, `orca2`, `phi4`, `qwen2`, `qwen2.5`, `qwen2.5-coder`, `qwen3`, `qwen3.5`, `qwen3-embedding`, `qwen3-vl`, `rnj-1`, `stable-code`, `stablelm-zephyr`, `starcoder`, `starcoder2`, `starling-lm`, `tulu3`, `vicuna`, `wizardlm`, `wizardlm2`, `xwinlm`, `yi`, `yi-coder`, `zephyr` |
| 8-16 GB VRAM | 24-32 GB RAM | `bespoke-minicheck`, `codegeex4`, `codellama`, `deepcoder`, `deepseek-coder`, `deepseek-llm`, `deepseek-ocr`, `deepseek-r1`, `deepscaler`, `dolphin-mistral`, `exaone3.5`, `exaone-deep`, `falcon`, `falcon2`, `gemma4`, `glm4`, `glm-ocr`, `granite4.2`, `internlm2`, `lfm2`, `lfm2.5`, `llama3.2-vision`, `magicoder`, `marco-o1`, `medgemma`, `medllama2`, `ministral-3`, `mistrallite`, `mixtral`, `olmo2`, `olmo-3`, `phi4-mini-reasoning`, `phi4-reasoning`, `phind-codellama`, `qwen2.5vl`, `qwen2-math`, `qwen3.6`, `qwen3.8`, `qwq`, `r1-1776`, `reader-lm`, `solar`, `sqlcoder`, `translategemma`, `wizardcoder`, `wizard-math`, `wizard-vicuna`, `wizard-vicuna-uncensored` |
| 16-32 GB VRAM | 32-64 GB RAM | `athene-v2`, `codebooga`, `codestral`, `cogito`, `deepseek-coder-v2`, `deepseek-v2`, `devstral-small-2`, `dolphin-mixtral`, `granite-code`, `magistral`, `mathstral`, `mistral-small`, `mistral-small3.1`, `mistral-small3.2`, `nemotron-3.5-lightning`, `olmo-3.1`, `qwen3-coder`, `qwen3-coder-next`, `qwen3`, `qwen3.5`, `qwen3-vl`, `solar-pro`, `yi` |
| 32-64 GB VRAM | 64-128 GB RAM | `command-r`, `deepseek-r1`, `deepseek-v2.5`, `devstral`, `goliath`, `llama3.3`, `llama4`, `mistral-large`, `mixtral`, `nemotron`, `nemotron-3-super`, `nemotron-cascade-2`, `qwen3-next`, `reflection`, `command-r-plus`, `nous-hermes2-mixtral` |
| 64-128 GB VRAM | 128-256 GB RAM | `command-a`, `dbrx`, `deepseek-v3`, `deepseek-v3.1`, `deepseek-v4-pro`, `gpt-oss`, `laguna-s-2.1`, `megadolphin`, `mistral-large-3`, `nemotron-3-ultra`, `qwen3.5`, `qwen3-next` |
| 128-256 GB VRAM | 256-512 GB RAM | `deepseek-v2:236b`, `deepseek-v2.5:236b`, `qwen3:235b`, `qwen3-vl:235b`, `dbrx:132b`, `command-a:111b`, `mistral-medium-3.5:128b`, `deepseek-v3`, `nemotron-3-ultra` |
| 256-500 GB VRAM | workstation/server multi-GPU | `qwen3-coder:480b`, `deepseek-v3`, `deepseek-v3.1`, `cogito-2.1`, `ornith-1.5`, `llama4:128x17b` |

### Famiglie specialistiche escluse dal consiglio finanziario

Sono disponibili su Ollama, ma non sono modelli generali adatti a redigere il
report dell'app:

```text
Embedding/RAG:
bge-large, bge-m3, embeddinggemma, granite-embedding, mxbai-embed-large,
nomic-embed-text, nomic-embed-text-v2-moe, paraphrase-multilingual,
snowflake-arctic-embed, snowflake-arctic-embed2, qwen3-embedding, all-minilm

Vision/OCR/documenti:
bakllava, deepseek-ocr, gemma3-vision, glm-ocr, granite3.2-vision, llava,
llava-llama3, llava-phi3, medgemma, medgemma1.5, minicpm-v, minicpm-v4.5,
minicpm-v4.6, moondream, qwen2.5vl, qwen3-vl, llama3.2-vision

Coding/SQL/agenti:
codebooga, codegeex4, codellama, codegemma, codeqwen, codestral, deepcoder,
deepseek-coder, deepseek-coder-v2, devstral, devstral-2, devstral-small-2,
granite-code, magicoder, opencoder, phind-codellama, qwen2.5-coder,
qwen3-coder, qwen3-coder-next, sqlcoder, starcoder, starcoder2, wizardcoder,
yi-coder, duckdb-nsql

Safety/classificazione/utility:
bespoke-minicheck, granite3-guardian, granite4.1-guardian, llama-guard3,
shieldgemma, functiongemma, tev1, nimble, nuextract, reader-lm
```

### Modelli non locali o oltre il limite

```text
kimi-k3:cloud, gemma4:cloud, gemma4:31b-cloud, gpt-oss:20b-cloud,
gpt-oss:120b-cloud, minimax-m3:cloud, deepseek-r1:671b, deepseek-v3:671b,
deepseek-v3.1:671b, cogito-2.1:671b
```

I tag `:cloud` richiedono il servizio cloud Ollama e non sono equivalenti a un
modello scaricato localmente. I modelli da 400B-700B richiedono infrastruttura
server multi-GPU e non sono realistici su una workstation ordinaria.

### Xiaomi MiMo e tag community

Xiaomi MiMo non è una famiglia ufficiale della libreria principale Ollama. Sono
però reperibili tag community e checkpoint Xiaomi convertiti:

```bash
ollama pull alibayram/mimo-7b-rl
ollama pull XiaomiMiMo/MiMo-VL-7B-RL
ollama pull XiaomiMiMo/MiMo-VL-7B-SFT
ollama pull petronetto/mimo-v2.6-9b
```

I primi tre sono rispettivamente MiMo-7B-RL e varianti MiMo-VL; l'ultimo è una
conversione community di MiMo-V2.6 Distill-Qwen-9B. Verificare sempre il tag
prima del download, perché i repository community non hanno la stessa garanzia
di manutenzione dei tag ufficiali Ollama.

## Catalogo completo per limite massimo VRAM

Le sezioni seguenti includono le famiglie locali scaricabili dalla libreria
ufficiale Ollama. Il valore indicato è il limite massimo pratico stimato per la
variante più grande della famiglia in quantizzazione comune, non la dimensione
del file e non una garanzia di velocità. Per le famiglie con molte varianti,
controllare il tag preciso con `ollama show famiglia`.

### Fino a 4 GB

```text
all-minilm, embeddinggemma, functiongemma, granite3-moe, granite3.1-moe,
granite3.1-dense, granite3-dense, granite3.2, granite3.3, granite4,
lfm2.5-thinking, medgemma1.5, nemotron-3-nano, phi, phi3, phi3.5,
phi4-mini, phi4-mini-reasoning, qwen, qwen2, qwen2.5, qwen3, smollm,
smollm2, stablelm2, tev1, tinyllama, tinydolphin
```

### Oltre 4 fino a 8 GB

```text
aya, aya-expanse, bge-large, bge-m3, codegemma, codeqwen, dolphin3,
falcon3, gemma, gemma2, gemma3, gemma3n, granite3.2-vision, granite4.1,
granite4.1-guardian, hermes3, llama2, llama3, llama3.1, llama3.2,
llama3-chatqa, llava, llava-llama3, llava-phi3, minicpm-v, mistral,
mistral-nemo, mistral-openorca, moondream, nemotron-mini, neural-chat,
nexusraven, nomic-embed-text, nomic-embed-text-v2-moe, openchat,
openhermes, opencoder, openthinker, orca2, orca-mini, phi4, phi4-reasoning,
qwen2.5-coder, qwen2.5vl, qwen3.5, qwen3-vl, qwen3-embedding, rnj-1,
stable-code, stablelm-zephyr, starcoder, starcoder2, starling-lm, tulu3,
vicuna, wizardlm, wizardlm2, xwinlm, yi, yi-coder, zephyr
```

### Oltre 8 fino a 16 GB

```text
bespoke-minicheck, codegeex4, codellama, deepcoder, deepseek-coder,
deepseek-llm, deepseek-ocr, deepseek-r1, deepscaler, dolphin-mistral,
exaone3.5, exaone-deep, falcon, falcon2, gemma4, glm4, glm-ocr, granite4.2,
internlm2, lfm2, lfm2.5, llama3.2-vision, magicoder, marco-o1, medgemma,
medllama2, ministral-3, mistrallite, mixtral, nemotron3, olmo2, olmo-3,
orca-mini, phi4-mini-reasoning, phind-codellama, qwen2.5, qwen2.5-math,
qwen2-math, qwen3.5, qwen3.6, qwen3.8, qwq, r1-1776, reader-lm, solar, sqlcoder,
translategemma, wizardcoder, wizard-math, wizard-vicuna, wizard-vicuna-uncensored
```

### Oltre 16 fino a 32 GB

```text
athene-v2, codebooga, codestral, cogito, deepseek-coder-v2, deepseek-v2,
devstral-small-2, dolphin-mixtral, exaone-deep, gemma4, granite-code,
granite4.2, lfm2, magistral, mathstral, mistral-small, mistral-small3.1,
mistral-small3.2, nemotron-3.5-lightning, olmo-3.1, openthinker, qwen3,
qwen3.5, qwen3-coder, qwen3-coder-next, qwen3-vl, qwen3.8, solar-pro,
wizardlm-uncensored, yi
```

### Oltre 32 fino a 64 GB

```text
command-r, deepseek-r1, deepseek-v2.5, devstral, devstral-small-2,
goliath, llama3.3, llama4, mistral-large, mistral-medium-3.5, mixtral,
nemotron, nemotron-3-super, nemotron-cascade-2, olmo-3.1, qwen3,
qwen3-next, qwen3-vl, qwen3-coder, qwen3.8, r1-1776, reflection,
command-r-plus, deepseek-llm, nous-hermes2-mixtral
```

### Oltre 64 fino a 128 GB

```text
command-a, dbrx, deepseek-v3, deepseek-v3.1, deepseek-v4-pro,
gpt-oss:120b, laguna-s-2.1, megadolphin, mistral-large-3,
nemotron-3-ultra, qwen3.5:122b, qwen3:235b, qwen3-next,
qwen3-coder:480b, llama3.1:405b, llama3.1:405b, hermes3:405b
```

### Oltre 128 fino a 256 GB

```text
deepseek-v2:236b, deepseek-v2.5:236b, qwen3:235b, qwen3-vl:235b,
dbrx:132b, command-a:111b, mistral-medium-3.5:128b, goliath,
nemotron-3-ultra, deepseek-v3, deepseek-v3.1
```

### Oltre 256 fino a 500 GB

```text
qwen3-coder:480b, qwen3-coder-next, deepseek-v3, deepseek-v4-pro,
deepseek-v3.1, cogito-2.1, ornith-1.5, llama4:128x17b
```

### Oltre 500 GB o cloud-only

Questi modelli non rientrano in un normale PC/workstation con massimo 500 GB
di VRAM oppure non sono scaricabili localmente tramite il tag mostrato:

```text
deepseek-r1:671b, deepseek-v3:671b, deepseek-v3.1:671b,
cogito-2.1:671b, kimi-k3:cloud, gemma4:cloud, gpt-oss:120b-cloud,
gpt-oss:20b-cloud, minimax-m3:cloud, deepseek-v4-pro:cloud
```

> Nota: alcuni nomi sopra rappresentano varianti/tag della stessa famiglia e
> alcuni modelli MoE hanno un numero elevato di parametri totali ma attivano
> solo una parte degli esperti. La VRAM richiesta dipende dal checkpoint
> specifico. Il catalogo è quindi una mappa per scegliere il tag, non una
> certificazione hardware: il controllo definitivo resta `ollama show TAG`.
