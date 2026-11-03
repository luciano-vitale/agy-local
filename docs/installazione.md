# Manuale di installazione e utilizzo di agy-local

Questa guida accompagna passo passo un nuovo utente dal repository su GitHub fino all'installazione e all'esecuzione di `agy-local` nel terminale integrato di Google Antigravity, in affiancamento alla CLI ufficiale `agy`.

---

## Indice

1. [Prerequisiti di sistema](#1-prerequisiti-di-sistema)
2. [Configurazione e avvio di LM Studio](#2-configurazione-e-avvio-di-lm-studio)
3. [Clonazione del repository da GitHub](#3-clonazione-del-repository-da-github)
4. [Installazione del comando agy-local](#4-installazione-del-comando-agy-local)
5. [Verifica dell'installazione e dello stato del sistema](#5-verifica-dellinstallazione-e-dello-stato-del-sistema)
6. [Esecuzione su doppio terminale in Google Antigravity](#6-esecuzione-su-doppio-terminale-in-google-antigravity)
7. [Utilizzo dei comandi slash, dei tool e delle skill](#7-utilizzo-dei-comandi-slash-dei-tool-e-delle-skill)
8. [Configurazione del modello ausiliario per ottimizzare il carico](#8-configurazione-del-modello-ausiliario-per-ottimizzare-il-carico)
9. [Manutenzione, log e risoluzione dei problemi](#9-manutenzione-log-e-risoluzione-dei-problemi)

---

## 1. Prerequisiti di sistema

Prima di installare `agy-local`, verifica di avere sul tuo sistema (macOS o Linux):

- **Google Antigravity e la CLI `agy`**: il binario ufficiale `agy` deve essere installato (tipicamente in `~/.local/bin/agy`) e autenticato almeno una volta. Puoi verificarlo aprendo un terminale ed eseguendo:
  ```bash
  which agy
  ```
- **Python 3.10 o superiore**: `agy-local` utilizza esclusivamente la libreria standard di Python 3 e non richiede l'installazione di pacchetti esterni tramite `pip`. Verifica la versione con:
  ```bash
  python3 --version
  ```
- **LM Studio**: installato sul computer con almeno un modello LLM scaricato (ad esempio modelli della famiglia Qwen o Gemma con supporto al tool calling).
- **Git**: per clonare il repository da GitHub.

---

## 2. Configurazione e avvio di LM Studio

Per consentire a `agy-local` di scoprire e utilizzare i modelli locali:

1. Apri **LM Studio**.
2. Assicurati di aver scaricato almeno un modello LLM nella tua libreria locale.
3. Entra nella sezione **Developer** (icona del terminale/server nella barra laterale sinistra di LM Studio).
4. Attiva il server locale cliccando su **Start server** (oppure avvialo da terminale con `lms server start`).
5. Verifica che l'endpoint sia in ascolto sulla porta predefinita `1234` (`http://localhost:1234/v1`).
6. Puoi testare che il server di LM Studio risponda eseguendo da terminale:
   ```bash
   curl http://localhost:1234/v1/models
   ```
   > **Nota**: non è obbligatorio pre-caricare manualmente un modello in memoria prima di avviare `agy-local`; se in LM Studio è attivo il caricamento *just-in-time* (JIT), il modello selezionato verrà caricato automaticamente al primo messaggio.

---

## 3. Clonazione del repository da GitHub

Apri il terminale e clona il repository in una cartella a tua scelta:

```bash
git clone https://github.com/luciano-vitale/agy-local.git
cd agy-local
```

---

## 4. Installazione del comando agy-local

Puoi installare `agy-local` nel tuo `PATH` usando la procedura automatica (consigliata) oppure quella manuale.

### Procedura automatica (consigliata)

Dalla cartella principale del repository appena clonato, esegui:

```bash
./bin/agy-local install
```

Questo comando esegue automaticamente le seguenti operazioni:
- imposta i permessi di esecuzione su `bin/agy-local`;
- crea il collegamento simbolico (`symlink`) in `~/.local/bin/agy-local`;
- verifica che il binario ufficiale `agy` supporti l'override `CLOUD_CODE_URL` e ne salva una copia di sicurezza funzionante in `~/.gemini/agy-local/bin/agy-pinned`;
- prepara il file di configurazione isolato `~/.gemini/agy-local/settings-local.json` sincronizzando i tuoi permessi e i tuoi workspace fidati;
- controlla che `~/.local/bin` sia presente nella variabile d'ambiente `PATH` e verifica la connessione a LM Studio.

### Procedura manuale equivalente

Se preferisci eseguire i passaggi manualmente da terminale:

```bash
chmod +x bin/agy-local
mkdir -p ~/.local/bin
ln -sf "$(pwd)/bin/agy-local" ~/.local/bin/agy-local
```

### Configurazione del PATH (se necessaria)

Se `~/.local/bin` non è ancora incluso nel tuo `PATH`, aggiungilo alla configurazione della tua shell (su macOS la shell predefinita è `zsh`):

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verifica che il comando sia visibile da qualsiasi directory:

```bash
which agy-local
```

---

## 5. Verifica dell'installazione e dello stato del sistema

Prima di aprire la sessione interattiva, controlla che tutti i componenti siano operativi:

```bash
agy-local status
```

Otterrai un riepilogo simile al seguente:

```text
=== agy-local Status ===
• System agy binary : /Users/tuo-utente/.local/bin/agy (CLOUD_CODE_URL OK)
• Pinned backup agy : /Users/tuo-utente/.gemini/agy-local/bin/agy-pinned (READY)
• Translation Proxy : RUNNING on http://127.0.0.1:47130 (requests=2)
• Model Host        : http://localhost:1234 (ONLINE, 23 LLM models)
• Loaded in VRAM    : gemma-4-e4b-it-mlx@4bit
• Primary Model     : qwen3.6-35b-a3b
• Auxiliary Model   : gemma-4-e4b-it-mlx@4bit
```

Per visualizzare l'elenco completo dei modelli LLM locali rilevati da LM Studio (con esclusione automatica dei modelli di solo embedding):

```bash
agy-local models
```

I modelli già caricati nella RAM/VRAM di LM Studio sono contrassegnati dal simbolo `●` e dall'etichetta `[Loaded]`.

---

## 6. Esecuzione su doppio terminale in Google Antigravity

L'obiettivo di `agy-local` è permetterti di alternare nella stessa finestra di Google Antigravity i modelli del tuo abbonamento cloud e i modelli locali di LM Studio per risparmiare token sui task più semplici o ripetitivi.

Segui questi passaggi all'interno di **Google Antigravity**:

1. **Apri Google Antigravity** sul tuo progetto o workspace di lavoro.
2. **Apri il primo terminale integrato (sessione cloud)**:
   - avvia la CLI standard digitando:
     ```bash
     agy
     ```
   - digita `/model` e premi Invio per selezionare uno dei modelli inclusi nel tuo abbonamento cloud (ad esempio `Gemini 4 Argon (Medium)` o `Claude Sonnet 5.5`).
3. **Apri un secondo terminale integrato separato (sessione locale)**:
   - clicca sul pulsante **`+`** del pannello terminale di Google Antigravity per aprire una nuova scheda terminale;
   - avvia la CLI locale digitando:
     ```bash
     agy-local
     ```
   - se il proxy locale non è ancora attivo, `agy-local` lo avvia automaticamente in background in pochi millisecondi prima di mostrare l'interfaccia interattiva;
   - digita `/model` e premi Invio: vedrai elencati esclusivamente i modelli locali disponibili nel tuo LM Studio;
   - seleziona il modello locale desiderato e premi Invio.

In questo modo:
- i due terminali operano sullo **stesso codice e workspace**, ma mantengono impostazioni del modello indipendenti (`~/.gemini/antigravity-cli/settings.json` per `agy` cloud e `~/.gemini/agy-local/settings-local.json` per `agy-local`);
- puoi passare in qualsiasi momento dal terminale `agy` (per analisi architetturali complesse in cloud) al terminale `agy-local` (per generazione codice, test, refactoring o domande rapide a costo zero di token cloud).

---

## 7. Utilizzo dei comandi slash, dei tool e delle skill

Poiché `agy-local` riutilizza il binario ufficiale `agy` collegandolo al proxy di traduzione locale, nel terminale `agy-local` hai a disposizione tutte le funzionalità native di Google Antigravity:

- **`/model`**: mostra il selettore interattivo dei modelli LLM presenti in LM Studio, indicando con `● ... [Loaded]` quelli già in memoria e ricordando la tua scelta per le sessioni `agy-local` successive.
- **`/context`**: mostra il dettaglio della finestra di contesto usando la reale capienza massima (`max_context_length`) comunicata da LM Studio per il modello locale selezionato.
- **`/clear`**: svuota la conversazione corrente mantenendo il modello locale attivo.
- **`/compact`**: comprime e riassume la cronologia della sessione locale per liberare spazio nella finestra di contesto.
- **Riferimenti `@file` e tool nativi**: puoi referenziare qualsiasi file del workspace (es. `@README.md`, `@src/agy_local/proxy.py`) e il modello locale può invocare tutti i tool (`read_file`, `edit_file`, `write_file`, `run_command`, ricerca nel codice) e le tue skill personalizzate (es. `/grill-with-docs`, `/code-review`, `/tdd`).

---

## 8. Configurazione del modello ausiliario per ottimizzare il carico

Quando avvii una conversazione, `agy` effettua dietro le quinte alcune chiamate leggere (come la generazione del titolo della chat, la sintesi di `/compact` o i sotto-agenti veloci `flash_lite`).

Con `agy-local` puoi affiancare al tuo modello principale (**Primary Local Model**, scelto con `/model`) un secondo modello locale più piccolo e rapido (**Auxiliary Local Model**, ad esempio un modello da 4B parametri come `gemma-4-e4b-it-mlx@4bit`):

```bash
# Imposta un modello leggero per i task secondari
agy-local aux gemma-4-e4b-it-mlx@4bit

# Verifica i modelli disponibili e quale è impostato come ausiliario
agy-local aux

# Disattiva il modello ausiliario (usa il modello principale per tutte le chiamate)
agy-local aux off
```

> **Protezione anti-thrashing della VRAM**: se LM Studio non ha abbastanza memoria per mantenere caricati contemporaneamente sia il modello principale sia quello ausiliario (e quindi scarica il modello principale), `agy-local` rileva automaticamente lo scaricamento tramite le API di LM Studio e reindirizza i task secondari sul modello principale già caricato, evitando continui ricaricamenti da disco.

---

## 9. Manutenzione, log e risoluzione dei problemi

### Comandi di gestione del proxy

```bash
# Mostra lo stato completo di binario, proxy, LM Studio e modelli in memoria
agy-local status

# Mostra gli ultimi 50 log del proxy di traduzione
agy-local logs -n 50

# Riavvia il proxy di traduzione in background
agy-local restart

# Ferma il proxy di traduzione in background
agy-local stop
```

### Esecuzione dei test automatici del progetto

Dalla directory del repository puoi eseguire l'intera suite di test unitari e di integrazione:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

### Problemi comuni e soluzioni

- **`agy-local` segnala che LM Studio non è raggiungibile**:
  verifica che l'applicazione LM Studio sia aperta e che il server locale nella scheda **Developer** sia avviato sulla porta `1234` (oppure imposta `LM_STUDIO_URL=http://localhost:<porta>` se usi una porta personalizzata).
- **Il comando `agy-local` non viene trovato nel terminale**:
  esegui `./bin/agy-local install` dalla cartella del repository e assicurati che `$HOME/.local/bin` sia nel tuo `PATH` ricaricando il terminale con `source ~/.zshrc`.
- **Aggiornamento futuro di `agy` (`agy update`)**:
  se un futuro aggiornamento del binario ufficiale `agy` dovesse modificare o rimuovere il supporto alla variabile `CLOUD_CODE_URL`, `agy-local` non invierà mai richieste al cloud per errore (modalità *fail-closed*) e passerà automaticamente alla copia compatibile salvata in `~/.gemini/agy-local/bin/agy-pinned`, avvisandoti a schermo.
