"""CLI launcher, daemon manager, and subcommands for agy-local."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from .config import (
    AgyLocalConfig,
    build_isolated_settings_file,
    get_pinned_agy_path,
    get_proxy_log_path,
    get_proxy_pid_path,
)
from .model_host import DualModelRouter
from .proxy import PROXY_SERVICE_SIGNATURE, TranslationProxyServer
from .safety import (
    FailClosedSafetyError,
    find_system_agy_binary,
    resolve_safe_agy_binary,
    verify_binary_supports_cloud_code_url,
)


AGY_SUBCOMMANDS = frozenset({
    "agent",
    "agents",
    "changelog",
    "help",
    "install",
    "mcp",
    "mic-serve",
    "models",
    "plugin",
    "plugins",
    "remote-control",
    "update",
})


def query_proxy_health(config: AgyLocalConfig, timeout: float = 1.5) -> Optional[dict[str, Any]]:
    """Return the health payload if the local Translation Proxy is running and verified."""
    url = f"http://{config.proxy_host}:{config.proxy_port}/__agy_local/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict) and data.get("service") == PROXY_SERVICE_SIGNATURE:
                return data
    except Exception:
        return None
    return None


def update_running_proxy_config(config: AgyLocalConfig, updates: dict[str, Any]) -> bool:
    """Push config updates to the running Translation Proxy if active."""
    url = f"http://{config.proxy_host}:{config.proxy_port}/__agy_local/config"
    try:
        body = json.dumps(updates).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def stop_proxy_daemon(config: AgyLocalConfig) -> bool:
    """Stop the background Translation Proxy daemon if running."""
    pid_path = get_proxy_pid_path()
    stopped = False
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(pid, signal.SIGTERM)
            for _ in range(20):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
            stopped = True
        except Exception:
            pass
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass
    return stopped


def ensure_proxy_running(config: AgyLocalConfig) -> dict[str, Any]:
    """Ensure the background Translation Proxy is running and passes the fail-closed handshake."""
    health = query_proxy_health(config, timeout=1.0)
    if health is not None:
        return health

    src_root = str(Path(__file__).resolve().parent.parent)
    env = os.environ.copy()
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_root}:{existing_py}" if existing_py else src_root

    log_path = get_proxy_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "ab") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agy_local.cli", "proxy-serve"],
            stdout=log_file,
            stderr=log_file,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )

    get_proxy_pid_path().write_text(f"{proc.pid}\n", encoding="utf-8")

    deadline = time.time() + 5.0
    while time.time() < deadline:
        health = query_proxy_health(config, timeout=0.5)
        if health is not None:
            return health
        if proc.poll() is not None:
            break
        time.sleep(0.1)

    raise FailClosedSafetyError(
        f"FAIL-CLOSED: Il Translation Proxy locale non ha risposto su "
        f"http://{config.proxy_host}:{config.proxy_port}/__agy_local/health. "
        f"Controlla i log con 'agy-local logs'."
    )


def run_proxy_server_foreground(config: AgyLocalConfig) -> int:
    """Run the Translation Proxy in the current process."""
    server = TranslationProxyServer((config.proxy_host, config.proxy_port), config=config)
    server.logger.info(
        "Starting agy-local Translation Proxy on http://%s:%d -> LM Studio %s (primary=%s, aux=%s)",
        config.proxy_host,
        config.proxy_port,
        config.lm_studio_url,
        config.primary_model,
        config.auxiliary_model,
    )

    def _handle_sigterm(signum: int, frame: Any) -> None:
        server.logger.info("Received signal %d, shutting down Translation Proxy", signum)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.server_close()
    return 0


def cmd_status(config: AgyLocalConfig) -> int:
    print("=== agy-local Status ===")
    sys_bin = find_system_agy_binary()
    pin_bin = get_pinned_agy_path()
    sys_ok = bool(sys_bin and verify_binary_supports_cloud_code_url(sys_bin))
    pin_ok = bool(pin_bin.is_file() and verify_binary_supports_cloud_code_url(pin_bin))

    print(f"• System agy binary : {sys_bin or 'not found'} ({'CLOUD_CODE_URL OK' if sys_ok else 'UNVERIFIED'})")
    print(f"• Pinned backup agy : {pin_bin} ({'READY' if pin_ok else 'not created yet'})")

    health = query_proxy_health(config, timeout=1.5)
    if health:
        print(f"• Translation Proxy : RUNNING on http://{config.proxy_host}:{config.proxy_port} (requests={health.get('requests_served')})")
    else:
        print(f"• Translation Proxy : STOPPED (will auto-start on next 'agy-local' run)")

    router = DualModelRouter(
        lm_studio_url=config.lm_studio_url,
        primary_model=config.primary_model,
        auxiliary_model=config.auxiliary_model,
        anti_thrashing_enabled=config.anti_thrashing,
    )
    models = router.fetch_models(timeout=2.0)
    if models:
        loaded = [m.id for m in models if m.is_loaded]
        print(f"• Model Host        : {config.lm_studio_url} (ONLINE, {len(models)} LLM models)")
        print(f"• Loaded in VRAM    : {', '.join(loaded) if loaded else 'none (JIT load on first prompt)'}")
    else:
        print(f"• Model Host        : {config.lm_studio_url} (OFFLINE / unreachable)")

    primary = (health.get("primary_model") if health else None) or router.primary_model or "auto"
    aux = (health.get("auxiliary_model") if health else None) or config.auxiliary_model or "off (using Primary)"
    print(f"• Primary Model     : {primary}")
    print(f"• Auxiliary Model   : {aux}")
    if health and health.get("thrashing_detected"):
        print(f"• VRAM Anti-Thrash  : ACTIVE FALLBACK ({health.get('thrashing_reason')})")
    return 0


def cmd_aux(config: AgyLocalConfig, args: list[str]) -> int:
    router = DualModelRouter(
        lm_studio_url=config.lm_studio_url,
        primary_model=config.primary_model,
        auxiliary_model=config.auxiliary_model,
    )
    models = router.fetch_models(timeout=2.0)
    known_ids = [m.id for m in models]

    if not args:
        print(f"Current Auxiliary Local Model: {config.auxiliary_model or 'off (uses Primary Local Model)'}")
        print("\nUsage:")
        print("  agy-local aux <model_id>   Set lightweight local model for background/auxiliary tasks")
        print("  agy-local aux off          Disable auxiliary model (use Primary Local Model for all tasks)")
        if models:
            print("\nAvailable LM Studio LLM models:")
            for m in models:
                marker = " [CURRENT AUX]" if m.id == config.auxiliary_model else ""
                print(f"  - {m.id:35} ({m.formatted_label()}{marker})")
        return 0

    target = args[0].strip()
    if target.lower() in ("off", "none", "disable", "reset"):
        config.auxiliary_model = None
        config.save()
        update_running_proxy_config(config, {"auxiliary_model": None})
        print("Auxiliary Local Model disattivato: tutte le chiamate useranno il Primary Local Model.")
        return 0

    if known_ids and target not in known_ids:
        # Try case-insensitive substring match if unambiguous
        matches = [mid for mid in known_ids if target.lower() in mid.lower()]
        if len(matches) == 1:
            target = matches[0]
        elif len(matches) > 1:
            print(f"Modello '{target}' ambiguo. Corrispondenze trovate: {', '.join(matches)}", file=sys.stderr)
            return 1
        else:
            print(
                f"Attenzione: '{target}' non risulta attualmente nella lista di LM Studio ({len(known_ids)} modelli), "
                "ma lo salvo comunque nella configurazione.",
                file=sys.stderr,
            )

    config.auxiliary_model = target
    config.save()
    update_running_proxy_config(config, {"auxiliary_model": target})
    print(f"Auxiliary Local Model impostato su: {target}")
    return 0


def cmd_logs(args: list[str]) -> int:
    lines_count = 50
    if len(args) >= 2 and args[0] == "-n":
        try:
            lines_count = int(args[1])
        except ValueError:
            pass
    log_path = get_proxy_log_path()
    if not log_path.exists():
        print(f"Nessun file di log presente in {log_path}")
        return 0
    all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in all_lines[-lines_count:]:
        print(line)
    return 0


def cmd_install(config: AgyLocalConfig, target_bin_dir: Optional[Path] = None) -> int:
    """Install the agy-local symlink into ~/.local/bin, verify PATH, and prepare agy-pinned."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    source_bin = repo_root / "bin" / "agy-local"
    if not source_bin.is_file():
        print(f"❌ Script eseguibile non trovato in '{source_bin}'.", file=sys.stderr)
        return 1

    source_bin.chmod(0o755)
    dest_dir = target_bin_dir or (Path.home() / ".local" / "bin")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_link = dest_dir / "agy-local"

    if dest_link.is_symlink() or dest_link.exists():
        dest_link.unlink()
    dest_link.symlink_to(source_bin)

    print(f"✅ Symlink creato: {dest_link} -> {source_bin}")

    # Verify and pin the official agy binary
    try:
        resolved = resolve_safe_agy_binary()
        print(f"✅ Binario ufficiale 'agy' verificato: {resolved.path}")
        print(f"✅ Copia di sicurezza 'agy-pinned' pronta in: {get_pinned_agy_path()}")
    except FailClosedSafetyError as err:
        print(f"⚠️  {err}", file=sys.stderr)

    # Create isolated settings file
    local_settings = build_isolated_settings_file()
    print(f"✅ Profilo impostazioni isolato pronto in: {local_settings}")

    # Check PATH
    path_entries = [Path(p).expanduser().resolve() for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if dest_dir.resolve() not in path_entries:
        print(
            f"\n⚠️  La cartella '{dest_dir}' non è nel tuo PATH attuale.\n"
            f"   Aggiungila eseguendo:\n"
            f"   echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.zshrc && source ~/.zshrc"
        )
    else:
        print(f"✅ '{dest_dir}' è già presente nel tuo PATH.")

    # Check LM Studio status
    router = DualModelRouter(lm_studio_url=config.lm_studio_url)
    models = router.fetch_models(timeout=1.5)
    if models:
        print(f"✅ LM Studio raggiungibile su {config.lm_studio_url} ({len(models)} modelli LLM rilevati).")
    else:
        print(
            f"ℹ️  LM Studio non risponde attualmente su {config.lm_studio_url}. "
            "Avvia il server locale di LM Studio prima di eseguire 'agy-local'."
        )

    print("\nInstallazione completata! Ora puoi aprire un terminale ed eseguire: agy-local")
    return 0


def _extract_agy_local_flags(argv: list[str], config: AgyLocalConfig) -> list[str]:
    """Extract --aux-model and --lm-url flags meant for agy-local and return remaining agy args."""
    remaining: list[str] = []
    i = 0
    changed = False
    while i < len(argv):
        arg = argv[i]
        if arg == "--aux-model" and i + 1 < len(argv):
            val = argv[i + 1]
            config.auxiliary_model = None if val.lower() in ("off", "none") else val
            changed = True
            i += 2
            continue
        if arg.startswith("--aux-model="):
            val = arg.split("=", 1)[1]
            config.auxiliary_model = None if val.lower() in ("off", "none") else val
            changed = True
            i += 1
            continue
        if arg == "--lm-url" and i + 1 < len(argv):
            config.lm_studio_url = argv[i + 1].rstrip("/")
            if config.lm_studio_url.endswith("/v1"):
                config.lm_studio_url = config.lm_studio_url[:-3].rstrip("/")
            changed = True
            i += 2
            continue
        if arg.startswith("--lm-url="):
            config.lm_studio_url = arg.split("=", 1)[1].rstrip("/")
            if config.lm_studio_url.endswith("/v1"):
                config.lm_studio_url = config.lm_studio_url[:-3].rstrip("/")
            changed = True
            i += 1
            continue
        remaining.append(arg)
        i += 1

    if changed:
        config.save()
        update_running_proxy_config(
            config,
            {
                "primary_model": config.primary_model,
                "auxiliary_model": config.auxiliary_model,
            },
        )
    return remaining


def main(argv: Optional[list[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    config = AgyLocalConfig.load()

    if raw_args:
        cmd = raw_args[0]
        if cmd == "proxy-serve":
            return run_proxy_server_foreground(config)
        if cmd == "status":
            return cmd_status(config)
        if cmd == "stop":
            if stop_proxy_daemon(config):
                print("Translation Proxy fermato.")
            else:
                print("Translation Proxy non era in esecuzione.")
            return 0
        if cmd == "restart":
            stop_proxy_daemon(config)
            health = ensure_proxy_running(config)
            print(
                f"Translation Proxy riavviato su http://{config.proxy_host}:{config.proxy_port} "
                f"(modelli LM Studio rilevati: {health.get('available_models_count')})."
            )
            return 0
        if cmd == "aux":
            return cmd_aux(config, raw_args[1:])
        if cmd == "logs":
            return cmd_logs(raw_args[1:])
        if cmd == "install":
            return cmd_install(config)

    agy_args = _extract_agy_local_flags(raw_args, config)

    # 1. Fail-Closed binary verification & automatic backup
    try:
        resolved_bin = resolve_safe_agy_binary()
    except FailClosedSafetyError as err:
        print(f"❌ {err}", file=sys.stderr)
        return 1

    if resolved_bin.warning:
        print(f"⚠️  {resolved_bin.warning}", file=sys.stderr)

    # 2. Ensure Translation Proxy is running and passes fail-closed health handshake
    try:
        health = ensure_proxy_running(config)
    except FailClosedSafetyError as err:
        print(f"❌ {err}", file=sys.stderr)
        return 1

    # 3. Build isolated settings-local.json so Local Session and Cloud Session never conflict
    local_settings_path = build_isolated_settings_file()

    # Check if user is invoking a subcommand or help flag that does not accept --settings / --model
    first_pos = next((a for a in agy_args if not a.startswith("-")), None)
    is_subcommand = first_pos in AGY_SUBCOMMANDS
    has_help_flag = ("-h" in agy_args) or ("--help" in agy_args)
    user_passed_settings = any(a == "--settings" or a.startswith("--settings=") for a in agy_args)
    user_passed_model = any(a == "--model" or a.startswith("--model=") for a in agy_args)

    # If user explicitly passed --model <m> on CLI, persist it as primary_model
    if user_passed_model:
        for idx, a in enumerate(agy_args):
            if a == "--model" and idx + 1 < len(agy_args):
                config.primary_model = agy_args[idx + 1]
                config.save()
                update_running_proxy_config(config, {"primary_model": config.primary_model})
                break
            if a.startswith("--model="):
                config.primary_model = a.split("=", 1)[1]
                config.save()
                update_running_proxy_config(config, {"primary_model": config.primary_model})
                break

    final_cmd: list[str] = [str(resolved_bin.path)]
    if not is_subcommand and not has_help_flag:
        if not user_passed_settings:
            final_cmd.extend(["--settings", str(local_settings_path)])
        active_primary = health.get("primary_model") or config.primary_model
        if not user_passed_model and active_primary:
            final_cmd.extend(["--model", str(active_primary)])

    final_cmd.extend(agy_args)

    env = os.environ.copy()
    env["CLOUD_CODE_URL"] = f"http://{config.proxy_host}:{config.proxy_port}"

    proc = subprocess.run(final_cmd, env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
