"""
Coursework for the bot: what is pending, the brief of one assignment, writing one as a Word document,
and the daily update.

The Assignments project is installed on one server (ASSIGN_HOST) together with the snapshot of the LMS
that the PC pushes there. This module reaches it through the ops agent (ops.py), which can only list,
show a brief, hand out the writing task and build the document. The LMS is contacted from the PC alone,
never from a server, and nothing is ever submitted.

The text is written here, by the models the bot already has, between the agent's `assign-prompt` and
`assign-build`. Nothing here imports telegram: bot.py owns the commands, buttons and the daily job.
"""
import asyncio
import base64
import json
import logging
import os
import re
import time

import httpx

import ops
from tgformat import escape_md

log = logging.getLogger("ai-bot.assign")

HOST = os.getenv("ASSIGN_HOST", "").strip()
STATE_FILE = os.getenv("ASSIGN_STATE_FILE", "data/assignments.json").strip()
CHECK_MINUTES = float(os.getenv("ASSIGN_CHECK_MINUTES", "15") or 15)
# Models tried after the chat's own model, in this order (keys from /model).
FALLBACKS = [k.strip() for k in os.getenv(
    "ASSIGN_MODELS", "groq-gptoss120b,gemini-pro,gpt-4o,gemini-flash,groq-gptoss20b,qwen-coder").split(",") if k.strip()]
# Size of the task given to a model, in characters: the instructions plus as much course material as fits.
PROMPT_CHARS = {"groq": 16000, "github": 18000, "gemini": 40000, "ollama": 18000}
MIN_PROMPT_CHARS = 6000
MIN_WORDS = 150  # a shorter reply is a refusal or an error message, not an assignment
HOSTED_TIMEOUT = 600  # seconds for a hosted model to finish
OLLAMA_IDLE_TIMEOUT = 900  # seconds a local model may go without producing anything
MAX_NOTE_BYTES = 5900  # what fits the agent's limit once encoded

AVAILABLE = ops.OPS_AVAILABLE and HOST in ops.OPS_HOSTS
LOCK = asyncio.Lock()  # one assignment is written at a time


def _clock(name: str, default: str) -> str:
    """A time of day from the environment as HH:MM, or "" for off."""
    value = os.getenv(name, default).strip().lower()
    if value in ("", "off", "no", "false", "0"):
        return ""
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value):
        log.warning("%s must look like 07:00, got %r; using %s", name, value, default)
        return default
    return value


# The daily update goes out after the first sync of the day, once it is DIGEST_TIME (in the Assignments
# project's time zone). If the PC has not synced by DIGEST_LATEST, it goes out with the data there is.
DIGEST_TIME = _clock("ASSIGN_DIGEST_TIME", "07:00")
DIGEST_LATEST = max(DIGEST_TIME, _clock("ASSIGN_DIGEST_LATEST", "10:00"))


class AssignError(Exception):
    pass


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
async def listing(everything: bool = False) -> dict:
    """Upcoming assignments and other deadlines; overdue ones too with `everything`."""
    return await ops.call(HOST, "assign-list", *(["all"] if everything else []), timeout=60)


async def brief(assign_id: int) -> dict:
    return await ops.call(HOST, "assign-brief", str(assign_id), timeout=90)


def explain(error: Exception) -> str:
    """An agent error as a sentence for the chat: without the host name and the error code."""
    text = str(error).removeprefix(f"{HOST}: ")
    code, _, rest = text.partition(": ")
    if code == "no_snapshot":
        return "No LMS data has arrived yet. On the PC run: python assignments.py sync --push"
    if re.fullmatch(r"[a-z_]+", code) and rest:
        text = rest
    # The project words its messages for its own command line.
    return text.replace("with `list`", "with /assignments").replace("run: sync ", "run: python assignments.py sync ")


# ---------------------------------------------------------------------------
# Formatting (Markdown for bot.send_formatted)
# ---------------------------------------------------------------------------
def _age(hours: float) -> str:
    if hours < 1:
        return "under an hour ago"
    return f"{hours:.0f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"


def _entry(a: dict, new: bool = False) -> list[str]:
    urgent = a.get("days_left") is not None and 0 <= a["days_left"] < 2
    due = escape_md(a["due_text"]) + (f" — **{escape_md(a['left'])}**{' ⏰' if urgent else ''}" if a.get("left") else "")
    lines = [
        f"**{escape_md(a['name'])}**" + (" 🆕" if new else ""),
        f"- {escape_md(a['course_code'])} {escape_md(a['course_name'])}",
        f"- Due {due}",
        f"- ID `{a['id']}`" + (f" · written: {escape_md(a['document'])}" if a.get("document") else ""),
    ]
    return lines + [""]


def format_list(view: dict, new_ids=(), title: str = "## 📋 Assignments") -> str:
    lines = [title, "", f"LMS data from {escape_md(view['synced_text'])} ({_age(view['age_hours'])})."]
    if view["stale"]:
        lines += ["", f"> ⚠️ The PC has not synced for over {view['stale_hours']:g} h, so this may be out of date."]
    upcoming = view["upcoming"]
    lines += ["", f"### 🔴 To do ({len(upcoming)})", ""]
    if not upcoming:
        lines += ["Nothing due. All caught up.", ""]
    for a in upcoming:
        lines += _entry(a, a["id"] in new_ids)

    others = view["other_deadlines"]
    if others:
        lines += [f"### 🗓 Other deadlines ({len(others)})", ""]
        for e in others:
            course = f"{escape_md(e['course'])}, " if e.get("course") else ""
            lines.append(f"- **{escape_md(e['name'])}** — {course}{escape_md(e['due_text'])} ({escape_md(e['left'])})")
        lines.append("")

    if "overdue" in view:
        if view["overdue"]:
            lines += [f"### ⚠️ Overdue ({len(view['overdue'])})", ""]
            for a in view["overdue"]:
                lines += _entry(a)
    elif view["overdue_count"]:
        lines += [f"⚠️ Overdue and not submitted: {view['overdue_count']} (see `/assignments all`)", ""]

    lines.append(f"✅ Submitted: {view['submitted_count']} of {view['total']}")
    return "\n".join(lines)


def format_brief(b: dict) -> str:
    # Telegram joins the lines of a paragraph into one, so the brief's "**Due:** ..." lines become a list
    # and every line of the lecturer's instructions becomes its own paragraph.
    lines, section = [], ""
    for line in (b["brief"].strip() or f"# {escape_md(b['name'])}\n\nThis assignment has no brief.").splitlines():
        bare = line.strip()
        if bare.startswith("#"):
            section = bare.lstrip("# ").lower()
        elif re.match(r"\*\*[^*]+:\*\*", bare):
            line = f"- {bare}"
        elif section == "instructions" and bare and not re.match(r"([-*+>]|\d+[.)])\s", bare):
            line = f"{bare}\n"
        lines.append(line)
    text = "\n".join(lines).strip()
    notes = list(b.get("warnings") or [])
    if b.get("submitted"):
        notes.append("This assignment is already marked as submitted on the LMS.")
    if notes:
        text += "\n\n" + "\n".join(f"> ⚠️ {escape_md(n)}" for n in notes)
    return text


# ---------------------------------------------------------------------------
# Daily update
# ---------------------------------------------------------------------------
def load_state() -> dict:
    return ops.load_state(STATE_FILE)


def save_state(state: dict) -> None:
    ops.save_state(state, STATE_FILE)


def pending_update(view: dict, state: dict) -> str | None:
    """The message that is due now, if any: the daily update, or word of an assignment that just appeared.

    Updates `state`; save it once the message has been sent.
    """
    first_run = "seen" not in state
    seen = set(state.get("seen", []))
    upcoming = [a["id"] for a in view["upcoming"]]
    new = [i for i in upcoming if i not in seen]

    if DIGEST_TIME and state.get("last_digest") != view["today"]:
        fresh = view["synced_day"] == view["today"]
        if view["now"] < DIGEST_TIME or (not fresh and view["now"] < DIGEST_LATEST):
            return None  # the new ones wait for the update
        state.update(last_digest=view["today"], seen=sorted(seen | set(upcoming)))
        text = format_list(view, () if first_run else new, title="## ☀️ Daily update")
        return text + ("\n\nTap an assignment for its brief, or have it written." if upcoming else "")

    if not new:
        return None
    state["seen"] = sorted(seen | set(upcoming))
    if first_run:
        return None
    lines = [f"## 🆕 New assignment{'s' if len(new) > 1 else ''}", ""]
    for a in view["upcoming"]:
        if a["id"] in new:
            lines += _entry(a)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _encode_notes(notes: str) -> str:
    raw = notes.strip().encode("utf-8")[:MAX_NOTE_BYTES].decode("utf-8", "ignore").encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _words(text: str) -> int:
    return len(re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).split())


def _too_large(error: Exception) -> bool:
    text = str(error).lower()
    return getattr(error, "status_code", None) == 413 or any(
        s in text for s in ("too large", "context length", "context window", "maximum context", "tokens per minute"))


def _why(error: Exception) -> str:
    if isinstance(error, (asyncio.TimeoutError, httpx.TimeoutException)):
        return "took too long"
    message = getattr(error, "message", None) or str(error) or type(error).__name__
    return re.sub(r"\s+", " ", str(message))[:200]


async def _hosted(model: dict, task: dict, on_text) -> tuple[str, list[str]]:
    parts, finish = [], None
    async with asyncio.timeout(HOSTED_TIMEOUT):
        stream = await model["client"].chat.completions.create(
            model=model["model"], stream=True, temperature=0.7,
            messages=[{"role": "system", "content": task["system"]}, {"role": "user", "content": task["prompt"]}],
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                parts.append(choice.delta.content)
                await on_text(parts)
            finish = choice.finish_reason or finish
    return "".join(parts), ["The model stopped at its output limit, so the text may be cut off."] if finish == "length" else []


async def _ollama(model: dict, task: dict, on_text) -> tuple[str, list[str]]:
    # The native API, because only it can set the context window: at the default size a long task is cut short.
    root = model["url"].rstrip("/").removesuffix("/v1")
    payload = {
        "model": model["model"], "stream": True,
        "messages": [{"role": "system", "content": task["system"]}, {"role": "user", "content": task["prompt"]}],
        "options": {"num_ctx": task["num_ctx"], "num_predict": task["max_tokens"], "temperature": 0.7},
    }
    parts, final = [], {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(OLLAMA_IDLE_TIMEOUT, connect=10)) as http:
        async with http.stream("POST", f"{root}/api/chat", json=payload) as r:
            if r.status_code != 200:
                raise AssignError(f"HTTP {r.status_code}: {(await r.aread()).decode('utf-8', 'replace')[:200]}")
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise AssignError(chunk["error"])
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    parts.append(piece)
                    await on_text(parts)
                if chunk.get("done"):
                    final = chunk
    warnings = []
    if final.get("done_reason") == "length":
        warnings.append("The model stopped at its output limit, so the text may be cut off.")
    if (final.get("prompt_eval_count") or 0) + (final.get("eval_count") or 0) >= task["num_ctx"]:
        warnings.append("The model's context window was full, so part of the material was probably dropped.")
    return "".join(parts), warnings


async def write(assign_id: int, notes: str, models: list[dict], progress) -> dict:
    """Write one assignment and build its document on the server.

    `models` are tried in order: dicts with label, provider, model and either client (an AsyncOpenAI
    client) or url (Ollama). `progress(text)` is awaited with a line saying what is happening.
    Returns the agent's reply (name, document, words, warnings...) plus docx (bytes), model (the label
    of the one that wrote it) and failures (what the models before it said).
    """
    if not models:
        raise AssignError("no model is available to write with")
    extra = [_encode_notes(notes)] if notes.strip() else []
    tasks: dict[int, dict] = {}

    async def task_for(chars: int) -> dict:
        if chars not in tasks:
            tasks[chars] = await ops.call(HOST, "assign-prompt", str(assign_id), str(chars), *extra, timeout=240)
        return tasks[chars]

    failures = []
    for model in models:
        chars = PROMPT_CHARS.get(model["provider"], 18000)
        while True:
            task = await task_for(chars)  # an agent error here is not the model's fault, so it ends the job
            started = time.monotonic()
            last = [started]

            async def on_text(parts: list[str], task=task, model=model, started=started, last=last) -> None:
                if time.monotonic() - last[0] >= 20:
                    last[0] = time.monotonic()
                    minutes = (time.monotonic() - started) / 60
                    await progress(f"✍️ Writing “{task['name']}” with {model['label']}\n"
                                   f"{_words(''.join(parts))} words so far, {minutes:.0f} min in.")

            await progress(f"✍️ Writing “{task['name']}” with {model['label']}…"
                           + ("\nA local model takes several minutes." if model["provider"] == "ollama" else ""))
            try:
                text, warnings = await (_ollama if model["provider"] == "ollama" else _hosted)(model, task, on_text)
                if _words(text) < MIN_WORDS:
                    raise AssignError(f"the reply was only {_words(text)} words")
            except Exception as e:  # noqa: BLE001 - any failure of a model means: try the next one
                if _too_large(e) and chars // 2 >= MIN_PROMPT_CHARS:
                    chars //= 2  # the same model once more, with less course material
                    continue
                log.warning("assignment %s: %s failed: %s", assign_id, model["label"], e)
                failures.append(f"{model['label']}: {_why(e)}")
                break
            await progress(f"📄 Building the document for “{task['name']}”…")
            try:
                built = await ops.call(HOST, "assign-build", str(assign_id), timeout=240, stdin=text)
            except ops.OpsError as e:
                if "too_short" not in str(e):
                    raise
                failures.append(f"{model['label']}: {explain(e)}")
                break
            docx = base64.b64decode(built.pop("docx"))
            return {**built, "docx": docx, "model": model["label"], "failures": failures,
                    "warnings": list(task["warnings"]) + warnings}
    raise AssignError("no model produced a usable text:\n" + "\n".join(f"• {f}" for f in failures))
