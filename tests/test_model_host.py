"""Unit tests for Model Host metadata mapping, Dual-Model routing, and VRAM anti-thrashing."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agy_local.model_host import DualModelRouter, LocalModelMetadata


class ModelHostTest(unittest.TestCase):
    def _sample_models(self, primary_loaded: bool = True, aux_loaded: bool = False) -> list[LocalModelMetadata]:
        return [
            LocalModelMetadata(
                id="qwen3.6-35b-a3b",
                display_name="Qwen3.6 35B A3B",
                max_context_length=262144,
                supports_images=True,
                supports_thinking=True,
                is_loaded=primary_loaded,
                quantization="4bit",
                params_string="35B",
                model_type="llm",
            ),
            LocalModelMetadata(
                id="gemma-4-e4b-it-mlx@4bit",
                display_name="Gemma 4 E4B IT MLX",
                max_context_length=131072,
                supports_images=True,
                supports_thinking=False,
                is_loaded=aux_loaded,
                quantization="4bit",
                params_string="4B",
                model_type="llm",
            ),
        ]

    def test_dual_model_routing_primary_vs_auxiliary(self) -> None:
        router = DualModelRouter(
            lm_studio_url="http://localhost:1234",
            primary_model="qwen3.6-35b-a3b",
            auxiliary_model="gemma-4-e4b-it-mlx@4bit",
        )
        router._last_known_models = self._sample_models()

        # Primary turn (has tools, requests primary model)
        target, used_aux = router.select_target_model(
            "qwen3.6-35b-a3b",
            {"tools": [{"functionDeclarations": [{"name": "read_file"}]}]},
        )
        self.assertEqual(target, "qwen3.6-35b-a3b")
        self.assertFalse(used_aux)

        # Auxiliary turn: title generation (no tools)
        target_title, used_aux_title = router.select_target_model(
            "gemini-3.5-flash-lite",
            {"contents": [{"role": "user", "parts": [{"text": "Generate title"}]}]},
        )
        self.assertEqual(target_title, "gemma-4-e4b-it-mlx@4bit")
        self.assertTrue(used_aux_title)

        # Auxiliary turn: flash_lite subagent (has tools, but requests internal flash-lite ID)
        target_sub, used_aux_sub = router.select_target_model(
            "gemini-3.5-flash-lite",
            {"tools": [{"functionDeclarations": [{"name": "read_file"}]}]},
        )
        self.assertEqual(target_sub, "gemma-4-e4b-it-mlx@4bit")
        self.assertTrue(used_aux_sub)

    def test_anti_thrashing_falls_back_when_primary_evicted(self) -> None:
        router = DualModelRouter(
            lm_studio_url="http://localhost:1234",
            primary_model="qwen3.6-35b-a3b",
            auxiliary_model="gemma-4-e4b-it-mlx@4bit",
            anti_thrashing_enabled=True,
        )
        router._last_known_models = self._sample_models(primary_loaded=True, aux_loaded=False)

        # Simulate LM Studio evicting primary_model after auxiliary_model is loaded
        with patch.object(router, "get_loaded_model_ids", return_value={"gemma-4-e4b-it-mlx@4bit"}):
            warning = router.record_execution_and_check_thrashing(
                used_auxiliary=True,
                primary_was_loaded_before=True,
            )
            self.assertIsNotNone(warning)
            self.assertTrue(router.thrashing_detected)

        # Subsequent auxiliary calls must now route to primary_model to prevent thrashing
        target, used_aux = router.select_target_model("gemini-3.5-flash-lite", {"contents": []})
        self.assertEqual(target, "qwen3.6-35b-a3b")
        self.assertFalse(used_aux)

    def test_build_fetch_available_models_response_excludes_helpers_from_sorts(self) -> None:
        router = DualModelRouter(
            lm_studio_url="http://localhost:1234",
            primary_model="qwen3.6-35b-a3b",
            auxiliary_model="gemma-4-e4b-it-mlx@4bit",
        )
        with patch.object(router, "fetch_models", return_value=self._sample_models()):
            resp = router.build_fetch_available_models_response()

        sort_ids = resp["agentModelSorts"][0]["groups"][0]["modelIds"]
        self.assertEqual(sort_ids, ["qwen3.6-35b-a3b", "gemma-4-e4b-it-mlx@4bit"])
        self.assertNotIn("gemini-3.5-flash-lite", sort_ids)
        # But helper key exists in models dict so internal lookups succeed
        self.assertIn("gemini-3.5-flash-lite", resp["models"])
        self.assertEqual(resp["models"]["qwen3.6-35b-a3b"]["maxTokens"], 262144)
        self.assertEqual(resp["models"]["gemma-4-e4b-it-mlx@4bit"]["maxTokens"], 131072)


if __name__ == "__main__":
    unittest.main()
