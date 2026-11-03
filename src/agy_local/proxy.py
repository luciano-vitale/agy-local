"""Local Translation Proxy implementing Cloud Code v1internal against LM Studio."""

from __future__ import annotations

import gzip
import http.server
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Iterator, Optional

from .config import AgyLocalConfig, get_proxy_log_path
from .model_host import DualModelRouter
from .translator import (
    _format_gemini_sse_event,
    convert_gemini_request_to_openai,
    extract_workspace_root,
    stream_openai_to_gemini_sse,
)


PROXY_SERVICE_SIGNATURE = "agy-local-translation-proxy"

DEFAULT_EXPERIMENTS_RESPONSE: dict[str, Any] = {
    "experimentIds": [],
    "flags": [
        {"name": "enable-customization-skills", "boolValue": True},
        {"name": "enable-customization-kind-skill", "boolValue": True},
        {"name": "enable-customization-kind-rule", "boolValue": True},
        {"name": "enable-customization-kind-hooks", "boolValue": True},
        {"name": "enable-customization-kind-workflow", "boolValue": True},
        {"name": "enable-customization-kind-plugin", "boolValue": True},
        {"name": "enable-customization-kind-mcp", "boolValue": True},
        {"name": "enable-customization-kind-agent", "boolValue": True},
        {"name": "enable-markdown-agents", "boolValue": True},
        {"name": "enable-learn-slash-command", "boolValue": True},
        {"name": "enable-tasks", "boolValue": True},
        {"name": "enable-ask-question-tool", "boolValue": True},
        {"name": "enable-image-generator-subagent", "boolValue": True},
        {"name": "send-subagent-initial-prompt-as-message", "boolValue": True},
        {"name": "cascade-enable-messaging", "boolValue": True},
        {"name": "json-hooks-enabled", "boolValue": True},
        {"name": "mcp-lazy-load-tools", "boolValue": True},
        {"name": "enable-ripgrep-path-search", "boolValue": True},
        {"name": "enable-file-diff-accumulator", "boolValue": True},
        {"name": "enable-sqlite-trajectory", "boolValue": True},
        {"name": "show-model-selection-change", "boolValue": True},
        {"name": "use-model-reported-token-count", "boolValue": True},
        {"name": "disable-loop-detection", "boolValue": True},
        {"name": "Chat__enable_agent_mode_slash_commands", "boolValue": True},
        {"name": "Chat__enable_custom_prompts", "boolValue": True},
        {"name": "DuetAICodeTransform__custom_slash_commands", "boolValue": True},
        {"name": "customization-token-budget", "intValue": "20000"},
        {"name": "rules-token-budget", "intValue": "20000"},
        {"name": "max-tokens-per-step", "intValue": "16384"},
        {"name": "tool-output-max-bytes", "intValue": "10240"},
        {"name": "default_subagent_interaction_timeout_seconds", "intValue": "60"},
        {"name": "max_injected_memories", "intValue": "-1"},
        {"name": "max_injected_skills", "intValue": "-1"},
        {
            "name": "cascade-conversation-history-config",
            "stringValue": '{"enabled": true, "max_conversations": 20}',
        },
        {
            "name": "log-artifacts-config",
            "stringValue": '{"enabled": true, "hideNominalToolSteps": false, "hidePlannerResponseText": false, "maxBytesPerStep": 4096, "maxBytesPerToolArg": 2048, "hideSystemSteps": false, "hideUserImplicitSteps": false}',
        },
        {
            "name": "CASCADE_USE_REPLACE_CONTENT_EDIT_TOOL",
            "stringValue": '{"max_fuzzy_edit_distance_fraction": 0.001, "allow_partial_replacement_success": true, "use_line_range": true, "fast_apply_fallback_config": {"enabled": false, "prompt_unchanged_threshold": 5, "content_view_radius_lines": 200, "content_edit_radius_lines": 5}}',
        },
    ],
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("agy_local.proxy")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        log_path = get_proxy_log_path()
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
    return logger


def _read_http_body(handler: http.server.BaseHTTPRequestHandler) -> bytes:
    transfer_encoding = (handler.headers.get("Transfer-Encoding") or "").lower()
    if "chunked" in transfer_encoding:
        chunks: list[bytes] = []
        while True:
            line = handler.rfile.readline().strip()
            if not line:
                break
            chunk_len = int(line.split(b";")[0], 16)
            if chunk_len == 0:
                handler.rfile.readline()
                break
            chunks.append(handler.rfile.read(chunk_len))
            handler.rfile.readline()
        raw = b"".join(chunks)
    else:
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length > 0 else b""

    content_encoding = (handler.headers.get("Content-Encoding") or "").lower()
    if "gzip" in content_encoding and raw:
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass
    return raw


def _iter_urlopen_lines(resp: Any) -> Iterator[bytes]:
    while True:
        line = resp.readline()
        if not line:
            break
        yield line


class TranslationProxyServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        config: AgyLocalConfig,
        router: Optional[DualModelRouter] = None,
    ) -> None:
        self.config = config
        self.router = router or DualModelRouter(
            lm_studio_url=config.lm_studio_url,
            primary_model=config.primary_model,
            auxiliary_model=config.auxiliary_model,
            anti_thrashing_enabled=config.anti_thrashing,
        )
        self.logger = setup_logger()
        self.started_at = time.time()
        self.requests_served = 0
        self._model_locks_guard = threading.Lock()
        self._model_locks: dict[str, threading.Lock] = {}
        super().__init__(server_address, TranslationProxyHandler)

    def get_model_lock(self, model_id: str) -> threading.Lock:
        with self._model_locks_guard:
            lock = self._model_locks.get(model_id)
            if lock is None:
                lock = threading.Lock()
                self._model_locks[model_id] = lock
            return lock


class TranslationProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def proxy_server(self) -> TranslationProxyServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: Any) -> None:
        self.proxy_server.logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/__agy_local/health"):
            router = self.proxy_server.router
            models = router.fetch_models(timeout=1.5)
            self._send_json(
                200,
                {
                    "service": PROXY_SERVICE_SIGNATURE,
                    "status": "ok",
                    "uptime_seconds": round(time.time() - self.proxy_server.started_at, 1),
                    "requests_served": self.proxy_server.requests_served,
                    "lm_studio_url": router.lm_studio_url,
                    "lm_studio_reachable": len(models) > 0,
                    "available_models_count": len(models),
                    "loaded_models": [m.id for m in models if m.is_loaded],
                    "primary_model": router.primary_model,
                    "auxiliary_model": router.auxiliary_model,
                    "anti_thrashing_enabled": router.anti_thrashing_enabled,
                    "thrashing_detected": router.thrashing_detected,
                    "thrashing_reason": router.thrashing_reason,
                },
            )
            return

        self._send_json(404, {"error": f"Unknown GET endpoint: {self.path}"})

    def do_POST(self) -> None:
        self.proxy_server.requests_served += 1
        raw_body = _read_http_body(self)

        if self.path.startswith("/__agy_local/config"):
            try:
                updates = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception:
                updates = {}
            router = self.proxy_server.router
            cfg = self.proxy_server.config
            if "primary_model" in updates:
                router.primary_model = updates["primary_model"]
                cfg.primary_model = updates["primary_model"]
            if "auxiliary_model" in updates:
                router.auxiliary_model = updates["auxiliary_model"]
                cfg.auxiliary_model = updates["auxiliary_model"]
                router.thrashing_detected = False
                router.thrashing_reason = None
            cfg.save()
            self._send_json(
                200,
                {
                    "status": "updated",
                    "primary_model": router.primary_model,
                    "auxiliary_model": router.auxiliary_model,
                },
            )
            return

        if "loadCodeAssist" in self.path:
            self._send_json(
                200,
                {
                    "currentTier": {
                        "id": "free-tier",
                        "name": "Antigravity Local (LM Studio)",
                        "description": "Local inference via LM Studio Translation Proxy",
                    },
                    "allowedTiers": [
                        {
                            "id": "free-tier",
                            "name": "Antigravity Local (LM Studio)",
                            "description": "Local inference via LM Studio Translation Proxy",
                            "isDefault": True,
                        }
                    ],
                    "cloudaicompanionProject": "aicode-consumers",
                    "gcpManaged": False,
                    "paidTier": {
                        "id": "g1-pro-tier",
                        "name": "LM Studio Local",
                        "description": "Zero cloud token consumption",
                    },
                },
            )
            return

        if "fetchAvailableModels" in self.path:
            catalog = self.proxy_server.router.build_fetch_available_models_response()
            self._send_json(200, catalog)
            return

        if "retrieveUserQuotaSummary" in self.path or "retrieveUserQuota" in self.path:
            self._send_json(
                200,
                {
                    "groups": [
                        {
                            "displayName": "LM Studio Local Models",
                            "description": "Local inference (unlimited local tokens)",
                            "buckets": [
                                {
                                    "bucketId": "local-unlimited",
                                    "displayName": "Local Quota Remaining",
                                    "window": "unlimited",
                                    "resetTime": "2099-12-31T23:59:59Z",
                                    "description": "Running locally on LM Studio (0 cloud tokens used).",
                                    "remainingFraction": 1.0,
                                }
                            ],
                        }
                    ]
                },
            )
            return

        if "fetchUserInfo" in self.path:
            self._send_json(200, {"userSettings": {}, "regionCode": "IT"})
            return

        if "listExperiments" in self.path:
            self._send_json(200, DEFAULT_EXPERIMENTS_RESPONSE)
            return

        if "countTokens" in self.path:
            est_tokens = max(1, len(raw_body) // 4)
            self._send_json(200, {"totalTokens": est_tokens})
            return

        if "streamGenerateContent" in self.path:
            self._handle_stream_generate_content(raw_body)
            return

        if "generateContent" in self.path:
            self._handle_generate_content(raw_body)
            return

        # All other service/telemetry calls (listExperiments, fetchAdminControls, recordTrajectoryAnalytics, etc.)
        self._send_json(200, {})

    def _handle_stream_generate_content(self, raw_body: bytes) -> None:
        router = self.proxy_server.router
        cfg = self.proxy_server.config
        logger = self.proxy_server.logger

        try:
            envelope = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception as exc:
            logger.error("Failed to parse streamGenerateContent request JSON: %s", exc)
            envelope = {}

        requested_model = envelope.get("model")
        inner_request = envelope.get("request") or {}

        prev_primary = router.primary_model
        target_model, used_aux = router.select_target_model(requested_model, inner_request)
        if router.primary_model != prev_primary:
            cfg.primary_model = router.primary_model
            cfg.save()
            logger.info("Updated Primary Local Model to '%s'", router.primary_model)

        primary_was_loaded = False
        if used_aux and router.anti_thrashing_enabled and router.primary_model:
            loaded_before = router.get_loaded_model_ids(timeout=1.5)
            primary_was_loaded = router.primary_model in loaded_before

        openai_payload = convert_gemini_request_to_openai(
            inner_request=inner_request,
            target_model=target_model,
            stream=True,
        )

        role_tag = "AUX" if used_aux else "PRIMARY"
        logger.info(
            "Dispatching %s request (requested=%s -> target=%s, messages=%d, tools=%d)",
            role_tag,
            requested_model,
            target_model,
            len(openai_payload.get("messages", [])),
            len(openai_payload.get("tools", [])),
        )

        lm_url = f"{router.lm_studio_url}/v1/chat/completions"
        req_bytes = json.dumps(openai_payload, ensure_ascii=False).encode("utf-8")

        resp = None
        last_http_err: Optional[urllib.error.HTTPError] = None
        last_err_detail = ""
        model_lock = self.proxy_server.get_model_lock(target_model)

        # Serialize request dispatch per model and retry transient HTTP 500 during cold JIT model loading
        for attempt in range(3):
            try:
                with model_lock:
                    req = urllib.request.Request(
                        lm_url,
                        data=req_bytes,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    resp = urllib.request.urlopen(req, timeout=600)
                break
            except urllib.error.HTTPError as http_err:
                last_http_err = http_err
                last_err_detail = http_err.read().decode("utf-8", errors="replace")
                if http_err.code == 500 and attempt < 2:
                    logger.warning(
                        "LM Studio transient HTTP 500 on %s (attempt %d/3, likely concurrent JIT load); retrying...",
                        target_model,
                        attempt + 1,
                    )
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
            except Exception as exc:
                logger.error("LM Studio connection error for %s: %s", lm_url, exc)
                self._send_sse_error_message(
                    target_model,
                    f"⚠️ Impossibile contattare LM Studio su `{router.lm_studio_url}` ({exc}). Verifica che il server locale di LM Studio sia avviato.",
                )
                return

        if resp is None and last_http_err is not None:
            logger.error("LM Studio HTTP %d for model %s: %s", last_http_err.code, target_model, last_err_detail)
            self._send_sse_error_message(
                target_model,
                f"⚠️ Errore LM Studio (HTTP {last_http_err.code}) sul modello `{target_model}`: {last_err_detail[:400]}",
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        workspace_root = extract_workspace_root(inner_request)
        try:
            with resp:
                for sse_chunk in stream_openai_to_gemini_sse(
                    _iter_urlopen_lines(resp),
                    model_version=target_model,
                    workspace_root=workspace_root,
                ):
                    self.wfile.write(sse_chunk)
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Client closed SSE stream early")
        except Exception as exc:
            logger.error("Error while streaming SSE from LM Studio: %s", exc)

        if used_aux:
            warning = router.record_execution_and_check_thrashing(
                used_auxiliary=True,
                primary_was_loaded_before=primary_was_loaded,
            )
            if warning:
                logger.warning("[Anti-Thrashing] %s", warning)

    def _handle_generate_content(self, raw_body: bytes) -> None:
        router = self.proxy_server.router
        logger = self.proxy_server.logger

        try:
            envelope = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            envelope = {}

        requested_model = envelope.get("model")
        inner_request = envelope.get("request") or {}
        target_model, _ = router.select_target_model(requested_model, inner_request)

        openai_payload = convert_gemini_request_to_openai(
            inner_request=inner_request,
            target_model=target_model,
            stream=False,
        )

        lm_url = f"{router.lm_studio_url}/v1/chat/completions"
        req = urllib.request.Request(
            lm_url,
            data=json.dumps(openai_payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                lm_data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.error("Non-streaming generateContent failed: %s", exc)
            self._send_json(
                200,
                {
                    "response": {
                        "candidates": [
                            {
                                "content": {"role": "model", "parts": [{"text": f"Error: {exc}"}]},
                                "finishReason": "STOP",
                            }
                        ],
                        "modelVersion": target_model,
                    }
                },
            )
            return

        choice = (lm_data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        parts: list[dict[str, Any]] = []
        if msg.get("reasoning_content"):
            parts.append({"text": str(msg["reasoning_content"]), "thought": True})
        if msg.get("content"):
            parts.append({"text": str(msg["content"])})
        for idx, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            parts.append(
                {
                    "functionCall": {
                        "id": tc.get("id") or f"call_{idx + 1}",
                        "name": fn.get("name", ""),
                        "args": args,
                    }
                }
            )
        if not parts:
            parts.append({"text": " "})

        self._send_json(
            200,
            {
                "response": {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": parts},
                            "finishReason": "STOP",
                        }
                    ],
                    "modelVersion": target_model,
                }
            },
        )

    def _send_sse_error_message(self, model_version: str, message: str) -> None:
        chunk = _format_gemini_sse_event(
            parts=[{"text": message}],
            model_version=model_version,
            response_id="local-err",
            trace_id="local-err",
            finish_reason="STOP",
            usage_metadata={"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(chunk)
