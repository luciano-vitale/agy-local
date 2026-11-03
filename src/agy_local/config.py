"""Configuration and isolated session settings management for agy-local."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_LM_STUDIO_URL = "http://localhost:1234"
DEFAULT_PROXY_HOST = "127.0.0.1"
DEFAULT_PROXY_PORT = 47130


def get_agy_local_dir() -> Path:
    """Return the directory where agy-local stores its state and logs."""
    base = Path(os.environ.get("AGY_LOCAL_HOME", Path.home() / ".gemini" / "agy-local"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_cloud_agy_settings_path() -> Path:
    """Return the path to the standard cloud agy settings.json."""
    return Path.home() / ".gemini" / "antigravity-cli" / "settings.json"


def get_local_settings_path() -> Path:
    """Return the path to the isolated settings-local.json passed via --settings."""
    return get_agy_local_dir() / "settings-local.json"


def get_config_path() -> Path:
    """Return the path to agy-local's persistent config.json."""
    return get_agy_local_dir() / "config.json"


def get_proxy_pid_path() -> Path:
    return get_agy_local_dir() / "proxy.pid"


def get_proxy_log_path() -> Path:
    return get_agy_local_dir() / "proxy.log"


def get_pinned_agy_path() -> Path:
    bin_dir = get_agy_local_dir() / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    return bin_dir / "agy-pinned"


@dataclass
class AgyLocalConfig:
    lm_studio_url: str = DEFAULT_LM_STUDIO_URL
    proxy_host: str = DEFAULT_PROXY_HOST
    proxy_port: int = DEFAULT_PROXY_PORT
    primary_model: Optional[str] = None
    auxiliary_model: Optional[str] = None
    anti_thrashing: bool = True

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AgyLocalConfig":
        cfg_path = path or get_config_path()
        data: dict[str, Any] = {}
        if cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}

        lm_url = os.environ.get(
            "LM_STUDIO_URL",
            data.get("lm_studio_url", DEFAULT_LM_STUDIO_URL),
        ).rstrip("/")
        # Strip trailing /v1 if user passed http://localhost:1234/v1
        if lm_url.endswith("/v1"):
            lm_url = lm_url[:-3].rstrip("/")

        port_str = os.environ.get("AGY_LOCAL_PORT")
        port = int(port_str) if port_str else int(data.get("proxy_port", DEFAULT_PROXY_PORT))

        return cls(
            lm_studio_url=lm_url,
            proxy_host=data.get("proxy_host", DEFAULT_PROXY_HOST),
            proxy_port=port,
            primary_model=data.get("primary_model"),
            auxiliary_model=data.get("auxiliary_model"),
            anti_thrashing=bool(data.get("anti_thrashing", True)),
        )

    def save(self, path: Optional[Path] = None) -> None:
        cfg_path = path or get_config_path()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cfg_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(cfg_path)


def build_isolated_settings_file(
    primary_model_display_or_id: Optional[str] = None,
    cloud_settings_path: Optional[Path] = None,
    local_settings_path: Optional[Path] = None,
) -> Path:
    """Create or update ~/.gemini/agy-local/settings-local.json.

    Copies permissions, trustedWorkspaces, and colorScheme from the user's
    cloud settings.json so all workspace permissions and rules work identically,
    while isolating the 'model' field so Local Sessions and Cloud Sessions
    never overwrite each other's selected model.
    """
    src_path = cloud_settings_path or get_cloud_agy_settings_path()
    dst_path = local_settings_path or get_local_settings_path()

    base_settings: dict[str, Any] = {}
    if src_path.exists():
        try:
            base_settings = json.loads(src_path.read_text(encoding="utf-8"))
        except Exception:
            base_settings = {}

    if dst_path.exists():
        try:
            existing_local = json.loads(dst_path.read_text(encoding="utf-8"))
        except Exception:
            existing_local = {}
    else:
        existing_local = {}

    merged: dict[str, Any] = dict(existing_local)
    for shared_key in ("colorScheme", "permissions", "trustedWorkspaces"):
        if shared_key in base_settings:
            merged[shared_key] = base_settings[shared_key]

    if primary_model_display_or_id:
        merged["model"] = primary_model_display_or_id
    elif "model" in merged and merged.get("model") == base_settings.get("model"):
        # Remove cloud model name from local settings so agy uses defaultAgentModelId
        merged.pop("model", None)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return dst_path
