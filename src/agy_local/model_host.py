"""Model Host (LM Studio) discovery, metadata mapping, and Dual-Model routing."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional


INTERNAL_HELPER_MODEL_IDS = frozenset({
    "gemini-3.5-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-thinking",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash-tiered",
    "chat_20706",
    "chat_23310",
    "MODEL_PLACEHOLDER_M50",
    "MODEL_PLACEHOLDER_M18",
    "MODEL_PLACEHOLDER_M198",
})


DEFAULT_SUPPORTED_MIME_TYPES = {
    "text/plain": True,
    "text/markdown": True,
    "text/html": True,
    "text/css": True,
    "text/csv": True,
    "text/xml": True,
    "text/javascript": True,
    "text/x-python": True,
    "text/x-python-script": True,
    "text/x-typescript": True,
    "application/json": True,
    "application/x-javascript": True,
    "application/x-typescript": True,
    "application/x-python-code": True,
    "application/x-ipynb+json": True,
    "application/pdf": True,
}

VISION_MIME_TYPES = {
    "image/png": True,
    "image/jpeg": True,
    "image/webp": True,
    "image/heic": True,
    "image/heif": True,
}


@dataclass
class LocalModelMetadata:
    id: str
    display_name: str
    max_context_length: int = 131072
    supports_images: bool = False
    supports_thinking: bool = False
    supports_tool_use: bool = True
    is_loaded: bool = False
    quantization: Optional[str] = None
    params_string: Optional[str] = None
    model_type: str = "llm"

    def formatted_label(self, is_aux: bool = False) -> str:
        prefix = "●" if self.is_loaded else "○"
        base = self.display_name or self.id
        details: list[str] = []
        if self.params_string and self.params_string.lower() not in base.lower():
            details.append(self.params_string)
        if self.quantization and self.quantization.lower() not in base.lower():
            details.append(self.quantization)
        suffix = f" ({', '.join(details)})" if details else ""
        tags = ""
        if self.is_loaded:
            tags += " [Loaded]"
        if is_aux:
            tags += " [Aux]"
        return f"{prefix} {base}{suffix}{tags}"


@dataclass
class DualModelRouter:
    """Routes requests between the Primary Local Model and optional Auxiliary Local Model.

    Includes VRAM anti-thrashing protection: if executing the Auxiliary Local Model
    causes the Model Host (LM Studio) to evict the Primary Local Model from memory,
    the router automatically disables auxiliary offloading for the session and falls
    back to the Primary Local Model.
    """

    lm_studio_url: str
    primary_model: Optional[str] = None
    auxiliary_model: Optional[str] = None
    anti_thrashing_enabled: bool = True
    thrashing_detected: bool = False
    thrashing_reason: Optional[str] = None
    _last_known_models: list[LocalModelMetadata] = field(default_factory=list)

    def fetch_models(self, timeout: float = 3.0) -> list[LocalModelMetadata]:
        """Query LM Studio's /api/v1/models (with fallback to /api/v0/models and /v1/models)."""
        models = (
            self._fetch_api_v1_models(timeout)
            or self._fetch_api_v0_models(timeout)
            or self._fetch_openai_v1_models(timeout)
        )
        llm_models = [
            m for m in models
            if m.model_type in ("llm", "vlm", "chat") and "embed" not in m.id.lower()
        ]
        if llm_models:
            self._last_known_models = llm_models
            if not self.primary_model or not any(m.id == self.primary_model for m in llm_models):
                # Prefer an already-loaded model if available, otherwise the first LLM
                loaded = [m for m in llm_models if m.is_loaded]
                self.primary_model = loaded[0].id if loaded else llm_models[0].id
        return llm_models

    def get_loaded_model_ids(self, timeout: float = 2.0) -> set[str]:
        models = self.fetch_models(timeout=timeout)
        return {m.id for m in models if m.is_loaded}

    def is_auxiliary_workload(self, requested_model: Optional[str], inner_request: dict[str, Any]) -> bool:
        """Determine whether an agy v1internal generation call is an auxiliary workload."""
        tools = inner_request.get("tools")
        has_tools = bool(tools)
        if not has_tools:
            # Title generation, commit message generation, or /compact summarization
            return True
        if requested_model and requested_model in INTERNAL_HELPER_MODEL_IDS:
            return True
        return False

    def select_target_model(
        self,
        requested_model: Optional[str],
        inner_request: dict[str, Any],
    ) -> tuple[str, bool]:
        """Return (target_model_id, used_auxiliary_model).

        Also updates primary_model when the user selects a new local model via /model.
        """
        known_ids = {m.id for m in self._last_known_models}
        if not known_ids:
            self.fetch_models()
            known_ids = {m.id for m in self._last_known_models}

        is_aux = self.is_auxiliary_workload(requested_model, inner_request)

        # If this is a primary turn and requested_model is a valid local model, update primary_model
        if not is_aux and requested_model and (not known_ids or requested_model in known_ids):
            if requested_model not in INTERNAL_HELPER_MODEL_IDS:
                self.primary_model = requested_model

        primary = self.primary_model
        if not primary and self._last_known_models:
            primary = self._last_known_models[0].id
            self.primary_model = primary
        primary = primary or requested_model or "local-model"

        if (
            is_aux
            and self.auxiliary_model
            and self.auxiliary_model != primary
            and not self.thrashing_detected
            and (not known_ids or self.auxiliary_model in known_ids)
        ):
            return self.auxiliary_model, True

        # If requested_model is an explicit local model chosen by user, honor it
        if requested_model and requested_model in known_ids and not is_aux:
            return requested_model, False

        return primary, False

    def record_execution_and_check_thrashing(
        self,
        used_auxiliary: bool,
        primary_was_loaded_before: bool,
    ) -> Optional[str]:
        """Check if running the Auxiliary Local Model evicted the Primary Local Model from VRAM."""
        if (
            not used_auxiliary
            or not self.anti_thrashing_enabled
            or self.thrashing_detected
            or not primary_was_loaded_before
            or not self.primary_model
            or not self.auxiliary_model
        ):
            return None

        loaded_after = self.get_loaded_model_ids(timeout=2.0)
        if self.auxiliary_model in loaded_after and self.primary_model not in loaded_after:
            self.thrashing_detected = True
            self.thrashing_reason = (
                f"Il caricamento dell'Auxiliary Local Model '{self.auxiliary_model}' ha scaricato "
                f"il Primary Local Model '{self.primary_model}' dalla RAM/VRAM di LM Studio. "
                f"Per evitare continui ricaricamenti (VRAM thrashing), i task ausiliari "
                f"verranno instradati direttamente su '{self.primary_model}'."
            )
            return self.thrashing_reason
        return None

    def build_fetch_available_models_response(self) -> dict[str, Any]:
        """Build the exact Cloud Code /v1internal:fetchAvailableModels response from LM Studio metadata."""
        llm_models = self.fetch_models()
        models_map: dict[str, Any] = {}
        model_ids: list[str] = []

        for idx, meta in enumerate(llm_models):
            model_ids.append(meta.id)
            mime_types = dict(DEFAULT_SUPPORTED_MIME_TYPES)
            if meta.supports_images:
                mime_types.update(VISION_MIME_TYPES)

            max_out = min(65535, max(8192, meta.max_context_length // 4))
            models_map[meta.id] = {
                "displayName": meta.formatted_label(is_aux=(meta.id == self.auxiliary_model)),
                "supportsImages": meta.supports_images,
                "supportsThinking": meta.supports_thinking,
                "minThinkingBudget": 128 if meta.supports_thinking else 0,
                "recommended": True,
                "maxTokens": meta.max_context_length,
                "maxOutputTokens": max_out,
                "quotaInfo": {
                    "remainingFraction": 1.0,
                    "resetTime": "2099-12-31T23:59:59Z",
                },
                "model": f"MODEL_PLACEHOLDER_M{500 + idx}",
                "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
                "modelProvider": "MODEL_PROVIDER_GOOGLE",
                "supportsVideo": False,
                "supportedMimeTypes": mime_types,
                "thinkingLevel": 1 if meta.supports_thinking else 0,
            }

        default_primary = self.primary_model or (model_ids[0] if model_ids else "local-model")
        aux_target = (
            self.auxiliary_model
            if (self.auxiliary_model and self.auxiliary_model in models_map and not self.thrashing_detected)
            else default_primary
        )

        # Ensure internal helper IDs resolve in models_map without appearing in agentModelSorts
        base_template = models_map.get(aux_target) or models_map.get(default_primary) or {
            "displayName": "Local Helper",
            "supportsImages": True,
            "supportsThinking": True,
            "maxTokens": 131072,
            "maxOutputTokens": 16384,
            "quotaInfo": {"remainingFraction": 1.0},
            "model": "MODEL_PLACEHOLDER_M50",
            "apiProvider": "API_PROVIDER_GOOGLE_GEMINI",
            "modelProvider": "MODEL_PROVIDER_GOOGLE",
        }
        for helper_id in ("gemini-3.5-flash-lite", "gemini-3-flash", "gemini-3.8-flash-tiered", "gemini-2.5-flash"):
            if helper_id not in models_map:
                helper_copy = dict(base_template)
                helper_copy["model"] = "MODEL_PLACEHOLDER_M50"
                models_map[helper_id] = helper_copy

        return {
            "models": models_map,
            "defaultAgentModelId": default_primary,
            "agentModelSorts": [
                {
                    "displayName": "LM Studio Local Models",
                    "groups": [{"modelIds": model_ids}],
                }
            ],
            "commandModelIds": [aux_target],
            "mqueryModelIds": [aux_target],
            "webSearchModelIds": [aux_target],
            "commitMessageModelIds": [aux_target],
            "tieredModelIds": {
                "flashLite": [aux_target],
                "flash": [aux_target],
                "pro": [default_primary],
            },
        }

    def _fetch_api_v1_models(self, timeout: float) -> list[LocalModelMetadata]:
        url = f"{self.lm_studio_url}/api/v1/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []

        items = data.get("models")
        if not isinstance(items, list):
            return []

        result: list[LocalModelMetadata] = []
        for item in items:
            model_id = item.get("key") or item.get("id")
            if not model_id:
                continue
            mtype = str(item.get("type", "llm")).lower()
            caps = item.get("capabilities") or {}
            loaded_instances = item.get("loaded_instances") or []
            is_loaded = len(loaded_instances) > 0

            ctx_len = int(item.get("max_context_length") or 131072)
            # If loaded instance specifies a configured context_length, prefer it
            if is_loaded and isinstance(loaded_instances[0], dict):
                inst_cfg = loaded_instances[0].get("config") or loaded_instances[0]
                if isinstance(inst_cfg, dict) and inst_cfg.get("context_length"):
                    ctx_len = int(inst_cfg["context_length"])

            quant_obj = item.get("quantization")
            quant_name = quant_obj.get("name") if isinstance(quant_obj, dict) else (str(quant_obj) if quant_obj else None)

            has_reasoning = bool(caps.get("reasoning")) or any(
                tok in model_id.lower() for tok in ("qwen3", "gemma-4", "deepseek-r1", "thinking")
            )

            result.append(
                LocalModelMetadata(
                    id=model_id,
                    display_name=item.get("display_name") or model_id,
                    max_context_length=ctx_len,
                    supports_images=bool(caps.get("vision", False)),
                    supports_thinking=has_reasoning,
                    supports_tool_use=bool(caps.get("trained_for_tool_use", True)),
                    is_loaded=is_loaded,
                    quantization=quant_name,
                    params_string=item.get("params_string"),
                    model_type=mtype,
                )
            )
        return result

    def _fetch_api_v0_models(self, timeout: float) -> list[LocalModelMetadata]:
        url = f"{self.lm_studio_url}/api/v0/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []

        items = data.get("data")
        if not isinstance(items, list):
            return []

        result: list[LocalModelMetadata] = []
        for item in items:
            model_id = item.get("id")
            if not model_id:
                continue
            mtype = str(item.get("type", "llm")).lower()
            caps = item.get("capabilities") or []
            is_loaded = item.get("state") == "loaded"
            ctx_len = int(item.get("max_context_length") or 131072)
            result.append(
                LocalModelMetadata(
                    id=model_id,
                    display_name=model_id,
                    max_context_length=ctx_len,
                    supports_images=(mtype == "vlm"),
                    supports_thinking=any(tok in model_id.lower() for tok in ("qwen3", "gemma-4", "r1")),
                    supports_tool_use=("tool_use" in caps),
                    is_loaded=is_loaded,
                    quantization=item.get("quantization"),
                    model_type=mtype,
                )
            )
        return result

    def _fetch_openai_v1_models(self, timeout: float) -> list[LocalModelMetadata]:
        url = f"{self.lm_studio_url}/v1/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []

        items = data.get("data")
        if not isinstance(items, list):
            return []

        return [
            LocalModelMetadata(
                id=item["id"],
                display_name=item["id"],
                model_type="embedding" if "embed" in item["id"].lower() else "llm",
            )
            for item in items
            if isinstance(item, dict) and item.get("id")
        ]
