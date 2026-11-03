# agy-local

Esegue la CLI ufficiale di Antigravity (`agy`) su un terminale separato utilizzando i modelli locali esposti da **LM Studio** (`http://localhost:1234/v1`), mantenendo il **100% di tool, skill, comandi slash (`/model`, `/context`, `/clear`, `/compact`, ecc.) e riferimenti `@file`**, senza consumare token del tuo abbonamento Cloud e senza interferire con le sessioni `agy` Cloud aperte in altri terminali.

## Come funziona

1. **Terminale 1 (Cloud Session)**: esegui `agy` normalmente e con `/model` selezioni un modello del tuo abbonamento (es. `Gemini 4 Argon (Medium)`).
2. **Terminale 2 (Local Session)**: esegui `agy-local` e con `/model` selezioni uno dei modelli LLM presenti nel tuo LM Studio (es. `qwen3.6-35b-a3b`, `gemma-4-31b-it-mlx`, `gemma-4-e4b-it-mlx@4bit`).
3. **Translation Proxy automatico**: `agy-local` avvia e supervisiona automaticamente in background un **Translation Proxy** locale (`http://127.0.0.1:47130`) che traduce il protocollo `v1internal` di `agy` verso le API di LM Studio (`/api/v1/models` e `/v1/chat/completions`), isolando le impostazioni della sessione locale in `~/.gemini/agy-local/settings-local.json`.

## Comandi Principali

```bash
# Avvia la CLI interattiva di Antigravity collegata a LM Studio
agy-local

# Avvia agy-local specificando direttamente un modello locale
agy-local --model qwen3.6-35b-a3b

# Elenca tutti i modelli LLM locali rilevati da LM Studio (con stato [Loaded])
agy-local models

# Configura un Auxiliary Local Model leggero (es. Gemma 4 E4B) per i task secondari
# (titolo conversazione, /compact, commit message, subagent flash_lite)
agy-local aux gemma-4-e4b-it-mlx@4bit

# Disattiva l'Auxiliary Local Model (usa solo il Primary Local Model)
agy-local aux off

# Controlla lo stato del proxy, dei modelli caricati in VRAM e della sicurezza Fail-Closed
agy-local status

# Visualizza gli ultimi log del Translation Proxy o riavvialo/fermalo
agy-local logs -n 50
agy-local restart
agy-local stop
```

## Sicurezza Fail-Closed e Pinning del Binario

- Ad ogni avvio, `agy-local` verifica che il binario `agy` contenga gli hook per `CLOUD_CODE_URL` e salva automaticamente una copia di backup funzionante in `~/.gemini/agy-local/bin/agy-pinned`.
- Se un futuro `agy update` dovesse modificare o rimuovere `CLOUD_CODE_URL`, `agy-local` passa automaticamente al binario `agy-pinned` salvato (o blocca l'avvio in modalità fail-closed), garantendo **zero consumo involontario di token Cloud**.
- **Anti-Thrashing VRAM**: quando usi sia un **Primary Local Model** sia un **Auxiliary Local Model**, il Translation Proxy monitora `loaded_instances` tramite `/api/v1/models` di LM Studio e ripiega automaticamente sul modello primario se rileva che LM Studio sta scaricando il modello primario dalla VRAM.

## Documentazione di Dominio e Architettura

- [GLOSSARY.md](./GLOSSARY.md): definizioni canoniche (`Local Session`, `Cloud Session`, `Model Host`, `Translation Proxy`, `Primary Local Model`, `Auxiliary Local Model`).
- [ADR 0001](./docs/adr/0001-cloudcode-translation-proxy.md): decisione architetturale sull'uso di `CLOUD_CODE_URL` e del Translation Proxy.
