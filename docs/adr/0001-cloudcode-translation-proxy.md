# Reuse official `agy` binary via `CLOUD_CODE_URL` and a local Translation Proxy

We will implement `agy-local` by launching the official `agy` CLI with `CLOUD_CODE_URL` pointed at a local Translation Proxy that translates Cloud Code `v1internal` calls (`fetchAvailableModels`, `streamGenerateContent`, `generateContent`, `countTokens`) to and from LM Studio's OpenAI-compatible API (`http://localhost:1234/v1`), while isolating local session settings via `--settings`.

Building a standalone CLI from scratch would fail to replicate `agy`'s built-in tools, skills, slash commands (`/model`, `/context`, `/clear`, `/compact`), and `@file` workspace references, whereas intercepting `CLOUD_CODE_URL` preserves 100% of the native `agy` UX in a separate terminal. Because `CLOUD_CODE_URL` is an internal endpoint-override hook in the `agy` binary, `agy-local` must fail closed at startup if a future `agy` update removes or ignores `CLOUD_CODE_URL`, preventing silent fallback to cloud quota.
