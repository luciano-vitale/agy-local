"""Fail-closed verification of CLOUD_CODE_URL support and binary pinning for agy."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import get_pinned_agy_path


REQUIRED_BINARY_MARKERS = (
    b"CLOUD_CODE_URL",
    b"Overriding CloudCodeServerURL via CLOUD_CODE_URL",
)


class FailClosedSafetyError(RuntimeError):
    """Raised when agy-local cannot guarantee that CLOUD_CODE_URL will intercept cloud traffic."""


@dataclass(frozen=True)
class ResolvedBinary:
    path: Path
    used_pinned_fallback: bool
    warning: Optional[str] = None


def find_system_agy_binary() -> Optional[Path]:
    """Locate the official agy binary on the system."""
    env_override = os.environ.get("AGY_BINARY_PATH")
    if env_override:
        p = Path(env_override).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()

    default_local = Path.home() / ".local" / "bin" / "agy"
    if default_local.is_file() and os.access(default_local, os.X_OK):
        return default_local.resolve()

    found = shutil.which("agy")
    if found:
        return Path(found).resolve()

    return None


def verify_binary_supports_cloud_code_url(binary_path: Path) -> bool:
    """Return True iff the binary contains the expected CLOUD_CODE_URL override hooks."""
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        return False

    try:
        content = binary_path.read_bytes()
    except OSError:
        return False

    return all(marker in content for marker in REQUIRED_BINARY_MARKERS)


def sync_pinned_binary(source_binary: Path, pinned_path: Optional[Path] = None) -> Path:
    """Save a backup copy of a known-good agy binary to ~/.gemini/agy-local/bin/agy-pinned."""
    target = pinned_path or get_pinned_agy_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    needs_copy = True
    if target.is_file():
        try:
            src_stat = source_binary.stat()
            dst_stat = target.stat()
            if src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) == int(dst_stat.st_mtime):
                needs_copy = False
        except OSError:
            needs_copy = True

    if needs_copy:
        tmp_target = target.with_suffix(".tmp")
        shutil.copy2(source_binary, tmp_target)
        tmp_target.chmod(0o755)
        tmp_target.replace(target)

    return target


def resolve_safe_agy_binary(
    system_binary: Optional[Path] = None,
    pinned_binary: Optional[Path] = None,
) -> ResolvedBinary:
    """Verify the system agy binary and fall back to the pinned binary or fail closed.

    Guarantees that agy-local never launches an agy binary that would ignore
    CLOUD_CODE_URL and silently consume Cloud Session quota.
    """
    sys_bin = system_binary or find_system_agy_binary()
    pin_bin = pinned_binary or get_pinned_agy_path()

    if sys_bin and verify_binary_supports_cloud_code_url(sys_bin):
        try:
            sync_pinned_binary(sys_bin, pin_bin)
        except OSError:
            pass
        return ResolvedBinary(path=sys_bin, used_pinned_fallback=False)

    # System binary is missing or updated in a way that removed CLOUD_CODE_URL
    if pin_bin.is_file() and verify_binary_supports_cloud_code_url(pin_bin):
        reason = (
            f"Il binario di sistema '{sys_bin}' non espone più il supporto verificato a CLOUD_CODE_URL "
            f"(probabile aggiornamento di agy). agy-local sta usando automaticamente il binario "
            f"compatibile salvato in '{pin_bin}' per proteggere i tuoi token cloud."
            if sys_bin
            else f"Binario 'agy' di sistema non trovato; uso il binario salvato in '{pin_bin}'."
        )
        return ResolvedBinary(path=pin_bin, used_pinned_fallback=True, warning=reason)

    raise FailClosedSafetyError(
        "FAIL-CLOSED: Impossibile verificare il supporto a CLOUD_CODE_URL nel binario 'agy' "
        f"({sys_bin or 'non trovato'}) e nessun binario di backup compatibile è presente in '{pin_bin}'. "
        "Avvio annullato per impedire il consumo accidentale di token del tuo abbonamento Cloud."
    )
