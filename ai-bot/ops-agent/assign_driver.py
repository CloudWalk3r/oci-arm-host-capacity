"""Assignments for ai-bot: reads the Assignments project on this host and prints one JSON object.

ai-bot-ops runs this with the project's own Python (its .venv), so the project's modules can be imported.
Everything comes from the snapshot the PC pushed. The LMS is never contacted: the project refuses to on
a server, and nothing here asks it to. Nothing is ever submitted.

    list [all]             upcoming assignments and other deadlines (overdue ones too with "all")
    brief ID               set up the assignment folder and return its brief
    prompt ID CHARS [NOTES]  what a model needs to write the assignment: the system prompt and the task,
                           with the task kept under CHARS characters. NOTES is base64url, because the
                           request reaches this script as words split on spaces.
    build ID               read the written text on stdin, save it as DRAFT.md and build the .docx

The bot's own models do the writing, between `prompt` and `build`.
"""
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(os.environ.get("ASSIGNMENTS_DIR") or Path.home() / "Assignments")
MIN_WORDS = 150  # a shorter reply is a refusal or an error message, not an assignment
MAX_DOC_BYTES = 8_000_000


class Refused(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")


def main(argv: list[str]) -> dict:
    if not (PROJECT / "assignments.py").exists():
        raise Refused("not_installed", f"the Assignments project is not in {PROJECT}")
    os.chdir(PROJECT)
    sys.path.insert(0, str(PROJECT))
    import ai
    import config
    import snapshot
    import workspace
    import write_content
    from lms_api import format_ts, is_submitted

    def about(a: dict) -> dict:
        ws = workspace.find_workspace(a["id"])
        documents = sorted(ws.glob("*.docx"), key=lambda p: p.stat().st_mtime) if ws else []
        return {
            "id": a["id"], "name": a["name"], "course_code": a["course_code"], "course_name": a["course_name"],
            "due_text": format_ts(a.get("due_ts")), "left": snapshot.time_left(a["due_ts"]) if a.get("due_ts") else "",
            "document": documents[-1].name if documents else None,
        }

    def stale_note(snap) -> list[str]:
        if not snap.stale():
            return []
        return [f"The LMS data is {snap.age_hours():.0f} h old; the PC has not synced since."]

    command, args = argv[0], argv[1:]
    snap = snapshot.load()

    if command == "list":
        view = snapshot.view(snap, include_overdue=args == ["all"])
        by_id = {a["id"]: a for a in snap.assignments}
        for group in ("upcoming", "overdue"):
            for entry in view.get(group, []):
                entry.update(about(by_id[entry["id"]]))
        for entry in view["other_deadlines"]:
            entry.update(due_text=format_ts(entry["due_ts"]), left=snapshot.time_left(entry["due_ts"]))
        now = datetime.now(config.TZ)
        synced = snap.synced_at.astimezone(config.TZ)
        view.update(
            today=f"{now:%Y-%m-%d}", now=f"{now:%H:%M}", synced_day=f"{synced:%Y-%m-%d}",
            synced_text=f"{synced:%a %d %b, %H:%M}", stale_hours=config.SNAPSHOT_STALE_HOURS,
        )
        return view

    assign_id = int(args[0])
    prepared = workspace.prepare(snap, assign_id)  # raises not_synced
    assignment = snap.assignment(assign_id)
    info = {**about(assignment), "submitted": is_submitted(assignment)}
    warnings = list(prepared["warnings"]) + stale_note(snap)

    if command == "brief":
        return {**info, "brief": prepared["brief"], "warnings": warnings}

    if command == "prompt":
        config.PROMPT_CHAR_BUDGET = max(4000, min(int(args[1]), 60000))
        notes = base64.urlsafe_b64decode(args[2] + "=" * (-len(args[2]) % 4)).decode("utf-8") if len(args) > 2 else ""
        context = workspace.load_context(prepared["path"])
        if not context["data"].get("intro_text") and not context["instructions"] and not notes.strip():
            raise Refused("no_instructions", "this assignment has no readable instructions. Send "
                                             f"/write {assign_id} followed by a description of the task.")
        if context["unreadable"]:
            warnings.append("Not readable, so not given to the model: " + ", ".join(context["unreadable"]))
        return {
            **info, "system": ai.SYSTEM_PROMPT, "prompt": ai.build_prompt(context, notes),
            "max_tokens": config.AI_MAX_TOKENS, "num_ctx": config.OLLAMA_NUM_CTX, "warnings": warnings,
        }

    if command == "build":
        text = ai.clean_output(sys.stdin.buffer.read().decode("utf-8", "replace"))
        words = len(text.split())
        if words < MIN_WORDS:
            raise Refused("too_short", f"the text is only {words} words")
        (prepared["path"] / write_content.DRAFT_FILE).write_text(text, encoding="utf-8")
        path = write_content.build(prepared["path"], text)  # never overwrites: a new copy gets a number
        if path.stat().st_size > MAX_DOC_BYTES:
            raise Refused("too_large", f"{path.name} is {path.stat().st_size / 1e6:.1f} MB")
        return {
            **info, "document": path.name, "words": words, "warnings": warnings,
            "docx": base64.b64encode(path.read_bytes()).decode(),
        }

    raise Refused("bad_request", f"unknown command {command}")


if __name__ == "__main__":
    try:
        result = main(sys.argv[1:])
    except Exception as e:  # noqa: BLE001 - every failure goes back to the bot as JSON
        code = getattr(e, "code", None)
        result = {"error": (f"{code}: {e}" if code else str(e) if isinstance(e, Refused) else f"{type(e).__name__}: {e}")[:500]}
    print(json.dumps(result, default=str))
    sys.exit(1 if "error" in result else 0)
