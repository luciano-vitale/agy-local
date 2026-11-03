"""Unit and integration tests for Fail-Closed safety checks and Translation Proxy endpoints."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from agy_local.config import AgyLocalConfig
from agy_local.model_host import DualModelRouter, LocalModelMetadata
from agy_local.proxy import PROXY_SERVICE_SIGNATURE, TranslationProxyServer
from agy_local.safety import (
    FailClosedSafetyError,
    REQUIRED_BINARY_MARKERS,
    resolve_safe_agy_binary,
    verify_binary_supports_cloud_code_url,
)


class SafetyAndProxyTest(unittest.TestCase):
    def test_verify_and_resolve_binary_with_pinned_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            good_sys = tmp_dir / "agy-sys"
            pinned = tmp_dir / "agy-pinned"

            good_sys.write_bytes(b"header " + b" ".join(REQUIRED_BINARY_MARKERS) + b" footer")
            good_sys.chmod(0o755)

            self.assertTrue(verify_binary_supports_cloud_code_url(good_sys))
            res = resolve_safe_agy_binary(system_binary=good_sys, pinned_binary=pinned)
            self.assertEqual(res.path, good_sys)
            self.assertFalse(res.used_pinned_fallback)
            self.assertTrue(pinned.is_file())

            # Simulate an agy update that removes CLOUD_CODE_URL from system binary
            good_sys.write_bytes(b"updated binary without cloud code url hook")
            res_fallback = resolve_safe_agy_binary(system_binary=good_sys, pinned_binary=pinned)
            self.assertEqual(res_fallback.path, pinned)
            self.assertTrue(res_fallback.used_pinned_fallback)
            self.assertIsNotNone(res_fallback.warning)

            # Simulate both binaries lacking CLOUD_CODE_URL -> must fail closed
            pinned.unlink()
            with self.assertRaises(FailClosedSafetyError):
                resolve_safe_agy_binary(system_binary=good_sys, pinned_binary=pinned)

    def test_translation_proxy_health_and_v1internal_endpoints(self) -> None:
        cfg = AgyLocalConfig(
            lm_studio_url="http://localhost:1234",
            proxy_host="127.0.0.1",
            proxy_port=0,
            primary_model="qwen3.6-35b-a3b",
        )
        router = DualModelRouter(lm_studio_url=cfg.lm_studio_url, primary_model="qwen3.6-35b-a3b")
        sample = [
            LocalModelMetadata(
                id="qwen3.6-35b-a3b",
                display_name="Qwen3.6 35B A3B",
                max_context_length=262144,
                supports_images=True,
                supports_thinking=True,
                is_loaded=True,
            )
        ]

        with patch.object(router, "fetch_models", return_value=sample):
            srv = TranslationProxyServer(("127.0.0.1", 0), config=cfg, router=router)
            port = srv.server_address[1]
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/__agy_local/health", timeout=2) as r:
                    health = json.loads(r.read())
                self.assertEqual(health["service"], PROXY_SERVICE_SIGNATURE)
                self.assertEqual(health["primary_model"], "qwen3.6-35b-a3b")

                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1internal:fetchAvailableModels",
                    data=b"{}",
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2) as r:
                    catalog = json.loads(r.read())
                self.assertIn("qwen3.6-35b-a3b", catalog["models"])
            finally:
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
