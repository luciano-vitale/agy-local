# agy-local

Allows running the official Antigravity CLI (`agy`) against locally hosted models in LM Studio alongside standard cloud-backed `agy` sessions.

## Language

**Local Session**:
An interactive `agy` CLI session started via `agy-local` whose model discovery and inference requests are routed to a local model host instead of consuming cloud subscription quota.
_Avoid_: Offline session, secondary chat, custom CLI

**Cloud Session**:
An interactive `agy` CLI session started via standard `agy` that authenticates and executes inference against Google's remote Cloud Code backend using the user's subscription.
_Avoid_: Standard session, main terminal

**Model Host**:
The local inference server (such as LM Studio on `http://localhost:1234/v1`) that serves locally downloaded LLMs via an OpenAI-compatible API.
_Avoid_: Local backend, inference server

**Translation Proxy**:
The local intermediary service that intercepts `agy`'s Cloud Code (`v1internal`) requests and translates model metadata, chat, and tool-calling streams to and from the Model Host's APIs.
_Avoid_: Bridge, wrapper, adapter

**Primary Local Model**:
The main local LLM selected by the user via `/model` in a Local Session to execute core agent reasoning, coding, and tool calls.
_Avoid_: Default model, heavy model

**Auxiliary Local Model**:
An optional, lighter local LLM (such as `gemma-4-e4b-it-mlx@4bit`) used by the Translation Proxy to offload background and lightweight tasks (conversation titling, fast subagents, context compaction) from the Primary Local Model.
_Avoid_: Secondary model, helper model

