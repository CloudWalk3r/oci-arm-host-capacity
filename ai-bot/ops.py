"""
Server ops for the bot: health of every instance, Lanternlight and Autopilot status, Ollama, and alert rules.

The bot reaches each host over SSH with a dedicated key that the host binds to ops-agent/ai-bot-ops as a
forced command, so the only things it can ask for are the verbs that script allows. Nothing here imports
telegram: bot.py owns the command handlers, buttons and the alert job.
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone

import asyncssh
import httpx

log = logging.getLogger("ai-bot.ops")


def _parse_hosts(spec: str) -> dict[str, str]:
    hosts = {}
    for part in spec.split(","):
        name, _, addr = part.strip().partition("=")
        if name and addr:
            hosts[name.strip()] = addr.strip()
    return hosts


# name=address pairs, e.g. micro1=10.0.1.40,micro2=10.0.1.116,arm1=10.0.1.123,arm2=10.0.1.55
OPS_HOSTS = _parse_hosts(os.getenv("OPS_HOSTS", ""))
OPS_SSH_USER = os.getenv("OPS_SSH_USER", "ubuntu").strip()
OPS_SSH_KEY = os.getenv("OPS_SSH_KEY", "/app/ops-ssh/id_ed25519").strip()
OPS_KNOWN_HOSTS = os.getenv("OPS_KNOWN_HOSTS", "/app/ops-ssh/known_hosts").strip()
LANTERN_HOST = os.getenv("OPS_LANTERN_HOST", "arm1").strip()
AUTOPILOT_HOST = os.getenv("OPS_AUTOPILOT_HOST", "arm1").strip()
ALERT_MINUTES = float(os.getenv("OPS_ALERT_MINUTES", "10") or 10)
STATE_FILE = os.getenv("OPS_STATE_FILE", "data/ops_state.json").strip()
DISK_ALERT_PCT = float(os.getenv("OPS_DISK_ALERT_PCT", "85") or 85)
MEM_ALERT_PCT = float(os.getenv("OPS_MEM_ALERT_PCT", "8") or 8)
LANTERN_BLOCKED_HOURS = float(os.getenv("OPS_LANTERN_BLOCKED_HOURS", "3") or 3)
STREAMS = ("youtube", "site", "gumroad", "newsletter", "stock")

OPS_AVAILABLE = bool(OPS_HOSTS) and os.path.exists(OPS_SSH_KEY)


class OpsError(Exception):
    pass


async def call(host: str, verb: str, *args: str, timeout: float = 90, stdin: str | None = None) -> dict:
    """Run one ai-bot-ops verb on a host and return its JSON reply. `stdin` is text the verb reads as its input."""
    addr = OPS_HOSTS.get(host)
    if not addr:
        raise OpsError(f"unknown host {host!r}")
    try:
        async with asyncssh.connect(
            addr, username=OPS_SSH_USER, client_keys=[OPS_SSH_KEY], known_hosts=OPS_KNOWN_HOSTS,
            agent_path=None, connect_timeout=15,
        ) as conn:
            result = await asyncio.wait_for(conn.run(" ".join((verb, *args)), check=False, input=stdin), timeout)
    except asyncio.TimeoutError:
        raise OpsError(f"{host}: timed out") from None
    except (OSError, ValueError, asyncssh.Error) as e:  # ValueError: bad key file or connection options
        raise OpsError(f"{host}: cannot connect ({type(e).__name__}: {e})") from None
    lines = (result.stdout or "").strip().splitlines()
    try:
        data = json.loads(lines[-1])
    except (IndexError, ValueError):
        raise OpsError(f"{host}: unexpected reply {((result.stderr or '') + (result.stdout or '')).strip()[:200]!r}") from None
    if isinstance(data, dict) and data.get("error"):
        raise OpsError(f"{host}: {data['error']}")
    return data


async def all_health() -> dict[str, dict | OpsError]:
    names = list(OPS_HOSTS)
    results = await asyncio.gather(*(call(n, "health") for n in names), return_exceptions=True)
    out: dict[str, dict | OpsError] = {}
    for name, r in zip(names, results):
        out[name] = r if isinstance(r, (dict, OpsError)) else OpsError(f"{name}: {type(r).__name__}: {r}")
    return out


async def ollama_status(base_url: str) -> dict:
    root = base_url.rstrip("/").removesuffix("/v1")
    async with httpx.AsyncClient(timeout=8) as c:
        version = (await c.get(f"{root}/api/version")).json().get("version")
        tags = (await c.get(f"{root}/api/tags")).json().get("models", [])
        loaded = (await c.get(f"{root}/api/ps")).json().get("models", [])
    return {"url": root, "version": version, "models": tags, "loaded": loaded}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt_age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def _iso_age(stamp: str) -> str:
    try:
        t = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return "?"
    return fmt_age(time.time() - t) + " ago"


def host_problems(name: str, h: dict) -> dict[str, str]:
    """Conditions worth an alert for one host, keyed so the same problem is only reported once."""
    p = {}
    if h["disk_used_pct"] >= DISK_ALERT_PCT:
        p[f"{name}:disk"] = f"{name}: disk {h['disk_used_pct']:.0f}% full ({h['disk_free_gb']} GB free)"
    if h["mem_total_mb"] and 100 * h["mem_avail_mb"] / h["mem_total_mb"] < MEM_ALERT_PCT:
        p[f"{name}:mem"] = f"{name}: low memory ({h['mem_avail_mb']} of {h['mem_total_mb']} MB available)"
    for c in h.get("containers") or []:
        if not c["managed"]:
            continue  # one-off `docker run` jobs (studio builds) come and go by design
        if c["state"] != "running":
            p[f"{name}:ctr:{c['name']}"] = f"{name}: container {c['name']} is {c['state']}" + (" (out of memory)" if c["oom"] else "")
        elif c.get("health") == "unhealthy":
            p[f"{name}:ctr:{c['name']}"] = f"{name}: container {c['name']} is unhealthy"
    for unit in h.get("failed_units") or []:
        p[f"{name}:unit:{unit}"] = f"{name}: systemd unit {unit} failed"
    if h.get("reboot_required"):
        p[f"{name}:reboot"] = f"{name}: reboot required to finish updates"
    return p


def format_servers(results: dict[str, dict | OpsError]) -> str:
    lines = [f"Servers ({len(results)})"]
    for name, h in results.items():
        lines.append("")
        if isinstance(h, OpsError):
            lines.append(f"❌ {name}: unreachable - {h}")
            continue
        problems = host_problems(name, h)
        lines.append(f"{'⚠️' if problems else '✅'} {name} · {h['arch']} · up {fmt_age(h['uptime_s'])}")
        swap = f", swap {h['swap_used_mb']} MB used" if h["swap_used_mb"] > 50 else ""
        lines.append(
            f"   load {h['load'][0]} ({h['cpus']} cpu) · mem {h['mem_avail_mb']}/{h['mem_total_mb']} MB free{swap}"
        )
        lines.append(f"   disk {h['disk_used_pct']:.0f}% used ({h['disk_free_gb']} GB free)")
        ctrs = h.get("containers")
        if ctrs is None:
            lines.append("   no docker")
        elif ctrs:
            parts = []
            for c in sorted(ctrs, key=lambda c: c["name"]):
                if c["state"] == "running" and c.get("health") != "unhealthy":
                    mark = "✓" if c["managed"] else "(job)"
                else:
                    mark = "✗" if c["managed"] else f"({c['state']})"
                extra = f" {c['restarts']}↻" if c["restarts"] else ""
                parts.append(f"{c['name']} {mark}{extra}")
            lines.append("   " + ", ".join(parts))
        else:
            lines.append("   no containers")
        upkeep = []
        if h.get("updates"):
            sec = f" ({h['security_updates']} security)" if h.get("security_updates") else ""
            upkeep.append(f"{h['updates']} updates{sec}")
        if h.get("reboot_required"):
            upkeep.append("reboot needed")
        if upkeep:
            lines.append("   " + " · ".join(upkeep))
        for msg in problems.values():
            if not msg.endswith("reboot required to finish updates"):
                lines.append(f"   ⚠️ {msg.split(': ', 1)[1]}")
    return "\n".join(lines)


def format_lantern(s: dict) -> str:
    if not s.get("present"):
        return f"Lanternlight is not installed on {LANTERN_HOST}."
    lines = [f"Lanternlight ({LANTERN_HOST}): {s.get('state') or '?'}"]
    if s.get("waiting_for"):
        lines.append(f"Waiting for: {s['waiting_for']}")
    if s.get("last_error"):
        lines.append(f"Last error: {s['last_error'][:300]}")
    parked = [k for k, v in (s.get("fails") or {}).items() if v >= 2]
    if parked:
        lines.append(f"\nParked after 2 failed builds ({len(parked)}):")
        by_error: dict[str, list[str]] = {}
        for part in parked:
            by_error.setdefault((s.get("errors") or {}).get(part, ""), []).append(part)
        for err, parts in by_error.items():
            lines += [f"• {part}" for part in parts]
            if err:
                lines.append(f"   ↳ {err[:200]}")
    posted = s.get("posted_24h") or []
    lines.append(f"\nPosted in the last 24h: {len(posted)}" + (f" ({', '.join(posted[-4:])})" if posted else ""))
    if s.get("next"):
        lines.append(f"Next up ({s.get('next_total', len(s['next']))}):")
        lines += [f"• {n}" for n in s["next"][:4]]
    if s.get("last_log"):
        stamp, _, rest = s["last_log"].partition(" ")
        lines.append(f"\nLast queue log ({_iso_age(stamp)}): {rest[:200]}")
    return "\n".join(lines)


def format_nightly(n: dict) -> str:
    if n.get("running"):
        line = f"Nightly YouTube build: running ({n.get('topic', '?')}, started {_iso_age(n.get('started', ''))}, step: {n.get('last', '?')})"
        if n.get("progress"):
            line += f"\n   render progress: {n['progress']}"
        return line
    if not n.get("last"):
        return "Nightly YouTube build: no runs logged yet"
    ok = n["last"].startswith("queued ")
    return f"Nightly YouTube build: {'✅' if ok else '⚠️'} last run {_iso_age(n.get('last_at', ''))}: {n['last']}"


def format_autopilot(s: dict) -> str:
    if not s.get("present"):
        return f"Autopilot is not installed on {AUTOPILOT_HOST}."
    lines = [f"{s['status']}", "", format_nightly(s.get("nightly") or {})]
    jobs = s.get("studio_jobs") or []
    if jobs:
        lines.append("Studio jobs running: " + "; ".join(f"{j['cmd'].strip(chr(34))[:80]} (started {j['running_for'].lower()})" for j in jobs))
    bad = [e for e in s.get("events", []) if e.get("level") in ("error", "warning", "warn")]
    if bad:
        lines.append(f"\nAlerts in the last 24h ({len(bad)}):")
        for e in bad[:5]:
            when = datetime.fromtimestamp(float(e["ts"]), timezone.utc).strftime("%d %b %H:%M")
            # notify.alert() already prefixes its messages with "ALERT [stream]".
            msg = e["msg"].removeprefix("ALERT ") if e["msg"].startswith("ALERT [") else f"[{e['stream']}] {e['msg']}"
            lines.append(f"• {when} {msg[:200]}")
        if len(bad) > 5:
            lines.append(f"…and {len(bad) - 5} more")
    else:
        lines.append("\nNo alerts in the last 24h.")
    return "\n".join(lines)


def format_ollama(o: dict, used_by: list[tuple[str, str]]) -> str:
    lines = [f"Ollama {o.get('version') or ''} at {o['url']}: up"]
    if o["models"]:
        lines.append("Installed: " + ", ".join(f"{m['name']} ({m.get('size', 0) / 1e9:.1f} GB)" for m in o["models"]))
    else:
        lines.append("Installed: none")
    if o["loaded"]:
        parts = []
        for m in o["loaded"]:
            until = (m.get("expires_at") or "")[11:16]
            parts.append(f"{m['name']} ({m.get('size', 0) / 1e9:.1f} GB" + (f", until {until} UTC)" if until else ")"))
        lines.append("Loaded now: " + ", ".join(parts))
    else:
        lines.append("Loaded now: none (models load on first use)")
    if used_by:
        installed = {m["name"] for m in o["models"]}
        lines.append("Bot models: " + ", ".join(f"{k} {'✓' if mid in installed else '✗ not installed'}" for k, mid in used_by))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def load_state(path: str = STATE_FILE) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning("Could not read %s: %s", path, e)
        return {}


def save_state(state: dict, path: str = STATE_FILE) -> None:
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, path)
    except Exception as e:
        log.warning("Could not write %s: %s", path, e)


async def collect(state: dict, ollama_url: str) -> dict:
    since = state.get("ap_since") or time.time()
    health, lantern, autopilot, ollama = await asyncio.gather(
        all_health(),
        call(LANTERN_HOST, "lantern-status") if LANTERN_HOST in OPS_HOSTS else asyncio.sleep(0, None),
        call(AUTOPILOT_HOST, "autopilot-status", f"{since:.3f}") if AUTOPILOT_HOST in OPS_HOSTS else asyncio.sleep(0, None),
        ollama_status(ollama_url) if ollama_url else asyncio.sleep(0, None),
        return_exceptions=True,
    )
    return {"health": health, "lantern": lantern, "autopilot": autopilot, "ollama": ollama, "at": time.time()}


def evaluate(state: dict, snap: dict) -> list[str]:
    """Update state in place from one snapshot; return alert lines to send (new problems, one-off events, resolutions)."""
    now = snap["at"]
    first_run = "active" not in state
    prev: dict[str, str] = state.get("active", {})
    fails: dict[str, int] = state.setdefault("fail_counts", {})
    # Docker's RestartCount only grows on restart-policy restarts (crashes) and resets when a container is recreated.
    restarts: dict[str, int] = state.setdefault("restarts", {})
    current: dict[str, str] = {}
    unknown: list[str] = []  # key prefixes we could not check this round: keep their previous state
    events: list[str] = []

    for name, h in snap["health"].items():
        if isinstance(h, Exception):
            fails[name] = fails.get(name, 0) + 1
            unknown.append(f"{name}:")
            if fails[name] >= 2:
                current[f"{name}:down"] = f"{name}: unreachable ({h})"
            continue
        fails[name] = 0
        current.update(host_problems(name, h))
        for c in h.get("containers") or []:
            if not c["managed"]:
                continue
            ref = f"{name}/{c['name']}"
            seen = restarts.get(ref)
            if seen is not None and c["restarts"] > seen:
                events.append(f"🔁 {name}: container {c['name']} crashed and restarted ({c['restarts'] - seen}x since last check)")
            restarts[ref] = c["restarts"]

    lantern = snap["lantern"]
    if isinstance(lantern, Exception):
        unknown.append("lantern:")
    elif lantern and lantern.get("present"):
        parked = sorted(k for k, v in (lantern.get("fails") or {}).items() if v >= 2)
        for part in parked:
            if part not in state.get("lantern_parked", []) and not first_run:
                err = (lantern.get("errors") or {}).get(part, "")
                events.append(f"🧩 Lanternlight parked {part} after 2 failed builds" + (f": {err[:200]}" if err else ""))
        state["lantern_parked"] = parked
        if lantern.get("state") == "blocked":
            since = state.setdefault("lantern_blocked_since", now)
            # On the first run we cannot tell how long it has been blocked, so report it straight away.
            if first_run or "lantern:blocked" in prev or now - since >= LANTERN_BLOCKED_HOURS * 3600:
                current["lantern:blocked"] = (
                    f"Lanternlight is blocked ({len(parked)} parked parts): {lantern.get('waiting_for') or 'see /lantern'}"
                )
        else:
            state.pop("lantern_blocked_since", None)
        if lantern.get("last_error"):
            current["lantern:error"] = f"Lanternlight error: {lantern['last_error'][:300]}"

    ap = snap["autopilot"]
    if isinstance(ap, Exception):
        unknown += ["autopilot:", "nightly:"]
        if not isinstance(snap["health"].get(AUTOPILOT_HOST), Exception):
            current["autopilot:down"] = f"Autopilot not answering: {ap}"
    elif ap and ap.get("present"):
        ev = ap.get("events") or []
        if ev:
            state["ap_since"] = max(float(e["ts"]) for e in ev)
        else:
            state.setdefault("ap_since", now)
        for e in sorted(ev, key=lambda e: float(e["ts"])):
            if first_run:
                break
            if e.get("level") in ("error", "warning", "warn"):
                events.append(f"🎬 Autopilot: {e['msg'].removeprefix('ALERT ')[:500]}")
            elif e.get("stream") == "digest":
                events.append(f"🗓 Autopilot digest\n{_digest_text(e['msg'])}")
        n = ap.get("nightly") or {}
        last = n.get("last") or ""
        recent = n.get("last_at") and (now - _stamp(n["last_at"]) < 26 * 3600)
        if not n.get("running") and recent and not last.startswith(("queued ", "would ")):
            current[f"nightly:{n['last_at']}"] = f"Nightly YouTube build stopped ({n.get('topic', '?')}): last step was '{last}'"

    o = snap["ollama"]
    if isinstance(o, Exception):
        current["ollama:down"] = f"Ollama unreachable ({type(o).__name__})"

    for key, msg in prev.items():
        if key not in current and any(key.startswith(u) for u in unknown):
            current[key] = msg

    alerts = [f"⚠️ {msg}" for key, msg in current.items() if key not in prev]
    alerts += [f"✅ Resolved: {msg}" for key, msg in prev.items() if key not in current]
    state["active"] = current
    return alerts + events


def _digest_text(msg: str) -> str:
    # The digest's Block links point at the guardian's public URL, which is not set up; drop them.
    lines = [l for l in msg.splitlines() if not l.strip().startswith("Block:")]
    return "\n".join(lines).replace(" Tap Block to stop one; do nothing to let them go out.", "")[:3500]


def _stamp(s: str) -> float:
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0
