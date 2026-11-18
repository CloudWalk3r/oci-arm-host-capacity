"""
Feedback on what the servers publish: Autopilot's YouTube videos (each long video with its Short) and
Lanternlight reels, a template audit across every built video, and the weekly channel numbers.

The material comes from the ai-bot-ops agent on the Autopilot host (review-list, review-item, review-lantern,
yt-stats). A text model reads the script, facts, title and description; a vision model looks at the thumbnail
and sampled frames. Reviews advise only: nothing here changes a video, and blocking one is the owner's call.
Nothing here imports telegram: bot.py owns the commands, buttons and jobs.
"""
import asyncio
import json
import logging
import os
import re
import statistics
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from openai import APIStatusError, AsyncOpenAI

import ops

log = logging.getLogger("ai-bot.review")


def _models(spec: str) -> list[tuple[str, str]]:
    pairs = (m.strip().partition(":") for m in spec.split(","))
    return [(provider, model) for provider, _, model in pairs if provider and model]


STATE_FILE = os.getenv("REVIEW_STATE_FILE", "data/reviews.json").strip()
ENABLED = os.getenv("REVIEW_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
WEEKLY_CHECK_MINUTES = 30  # how often bot.py looks whether the weekly channel numbers are due
# How many of the next unposted Lanternlight parts /review lists.
LANTERN_AHEAD = int(os.getenv("REVIEW_LANTERN_AHEAD", "6") or 0)
# Times in messages are shown in this offset from UTC (330 = Sri Lanka).
UTC_OFFSET = timedelta(minutes=int(os.getenv("REVIEW_UTC_OFFSET_MINUTES", "330") or 0))
# provider:model, tried in order. The first that answers wins.
TEXT_MODELS = _models(os.getenv(
    "REVIEW_TEXT_MODELS", "groq:openai/gpt-oss-120b,gemini:gemini-3.5-flash,gemini:gemini-flash-lite-latest"
))
VISION_MODELS = _models(os.getenv(
    "REVIEW_VISION_MODELS", "gemini:gemini-3.5-flash,gemini:gemini-flash-latest,gemini:gemini-flash-lite-latest"
))
ENDPOINTS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}
LONG_FRAMES, SHORT_FRAMES = 8, 5  # stills sampled evenly through the video (the agent allows up to 8)
MODEL_REST_S = 900  # how long a model that failed is tried last
KEEP_REVIEWS = 300

LOCK = asyncio.Lock()  # one review at a time, so two requests don't fight over rate limits
_clients: dict[str, AsyncOpenAI] = {}
_resting: dict[str, float] = {}  # model -> time until which it is tried last


class ReviewError(Exception):
    pass


def _client(provider: str) -> AsyncOpenAI | None:
    base_url, key_name = ENDPOINTS.get(provider, ("", ""))
    key = os.getenv(key_name, "").strip() if key_name else ""
    if not key:
        return None
    if provider not in _clients:
        _clients[provider] = AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0, timeout=150)
    return _clients[provider]


AVAILABLE = ENABLED and ops.OPS_AVAILABLE and any(_client(p) for p, _ in TEXT_MODELS)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ANSWER = (
    'Answer with JSON only: {"summary": "one plain sentence on the overall state", "findings": [{"severity": '
    '"serious" or "minor", "where": "WHERE", "problem": "what is wrong, quoting the words or numbers involved", '
    '"fix": "the concrete change to make"}]}. "serious" is for a wrong number or claim, a promise the content '
    "breaks, or a visible defect; anything else worth changing is \"minor\". Report only what you can point to in "
    "the material: no general advice, no suggestions to add things that are merely absent, nothing about what you "
    "cannot see or hear, and nothing a viewer wouldn't notice. A few sure findings are worth more than many "
    "doubtful ones, so leave out whatever you are not sure of. Most reviews should end with between zero and three "
    "findings, never more than five, most important first; an empty list when nothing is wrong."
)

YT_TEXT = (
    "You review one video of a faceless YouTube channel that explains personal-finance ideas with whiteboard "
    "drawings and a narrator, before it is published. Look for what would hurt it with viewers or with YouTube: "
    "numbers in the script, captions, board or description that disagree with each other or with the brief's "
    "facts, claims the facts don't support, a title that promises something the script doesn't deliver, an "
    "opening that takes too long "
    "to say why to keep watching, a passage a newcomer couldn't follow, description text, chapter labels, tags or "
    "hashtags that don't match what the script covers, and wording that reads as advice to one person rather than "
    "an explanation.\n\n"
    "How these videos are made, so you don't report the template as a mistake:\n"
    "- Each beat has `says` (the narration, with numbers spelled out for the voice), sometimes `caption` (on-screen "
    "text, in digits) and `board` (what is written on the whiteboard). \"seventy two\" and \"72\" are the same "
    "value; report a difference only when the values themselves differ.\n"
    "- A Short has four beats: hook, insight, answer, end. The end line is cut off on purpose so it runs back into "
    "the hook when the Short loops, and it sends viewers to the long video. A Short covers one slice of the long "
    "video on purpose: never report what it leaves out, and judge its title against the Short alone.\n"
    "- People, amounts and rates in the script are worked examples; `brief.assumptions` says what they rest on "
    "(the video does not have to repeat it). `brief.facts` was computed by a calculator and is the reference: "
    "check everything else against it. You may verify a sum, a difference or a simple division yourself, but "
    "results of compounding (growth over years, how long money lasts) can't be done in your head, so never call "
    "such a figure wrong unless it disagrees with the facts. For a Short the facts cover the whole long video, so "
    "most of them won't appear.\n"
    "- Chapter timestamps in the description are generated; judge whether each label fits the part of the script "
    "it points at, not the timestamps.\n\n"
    + ANSWER.replace("WHERE", "a beat id, or Title, Description, Chapters, Tags or Hashtags")
)

YT_STILLS = (
    "You check stills from one video of a personal-finance YouTube channel before it is published. The videos are "
    "marker drawings on a white whiteboard with a narrator; Shorts are vertical. You get the title, then the "
    "thumbnail when there is one, then frames labelled with their timestamps.\n\n"
    "Report only what is visibly wrong: text cut off at an edge, text overlapping other text or a drawing, a "
    "misspelling, text too small to read on a phone, a frame that is blank or nearly empty, a drawing that plainly "
    "contradicts the number written next to it, a thumbnail whose words can't be read at small size or that "
    "promises something different from the title.\n\n"
    "The stills are scaled down, so don't report softness or low resolution. Every scene starts on a clean board "
    "and the drawing builds up, so a frame with little or nothing drawn yet is normal, as is a half-finished "
    "drawing; report emptiness only when most of the frames are empty, as one finding. The marker pen travels over "
    "the board while it draws and can cover a word for a moment. The board often keeps the previous example's "
    "drawing or chart in view while the next one is written beside it, so a drawing showing the earlier numbers "
    "is not a contradiction. Captions appear a few words at a time, so a caption showing a fragment of a sentence "
    "is normal too. The thumbnail is an illustration in its own style. Don't comment on the style or propose a "
    "redesign.\n\n"
    + ANSWER.replace("WHERE", "Thumbnail, or Frame at m:ss")
)

LL_TEXT = (
    "You review one part of a serialized fantasy story before it is posted as a vertical reel of about 45 seconds "
    "on Facebook and Instagram (pixel art, one narrator). The story is told in the second person: the viewer is the "
    "main character. You get the story bible, what happened in earlier parts, the names of the parts still to "
    "come, this part's narration lines and the captions that go with the post.\n\n"
    "Look for: a first line that wouldn't stop someone scrolling, anything that contradicts the bible or earlier "
    "parts (names, objects, the rule of the world, what characters know), a line that is confusing when heard once "
    "at speaking speed, an ending that doesn't land the cliffhanger (or, in the final part, doesn't resolve the "
    "story), captions that describe a different moment than this part, a wrong part number or next-part line, a "
    "comment question that can't be answered from this part alone, hashtags unrelated to the story, spelling "
    "mistakes, and anything unsuitable for a general audience.\n\n"
    "The bible is right by definition: its rule and world are the story's own logic, however unlike the real "
    "world, so report a line only when it contradicts the bible or an earlier part, not when it is merely strange. "
    "When you report a contradiction, quote both sides in `problem`: the line or caption, and the exact sentence "
    "of the bible or earlier part it conflicts with. If you can't quote the second one, there is no contradiction "
    "to report. "
    "The Facebook and Instagram captions are meant to be nearly identical, the next-part line is a fixed short "
    "phrase, and every part repeats the series hashtag; none of these is a problem.\n\n"
    + ANSWER.replace("WHERE", "Line N (counting from 1), Facebook caption, Instagram caption or Hashtags")
)

LL_STILLS = (
    "You check the contact sheet of one reel of a pixel-art fantasy story before it is posted: a grid of stills in "
    "playing order, left to right, top to bottom. You also get how the story bible describes the characters.\n\n"
    "Report only what is visibly wrong: a still that is black, blank or mostly empty, a scene that runs off the "
    "edge of the art so bare background shows, a glitch or misplaced sprite, on-screen text that is cut off or "
    "misspelled, a character drawn differently from what the bible states. Where the bible says nothing about a "
    "character's clothes or colours, any look is fine. Each still carries a small number and shot name in its "
    "corner, which are not part of the reel. The caption on a still is whatever was being said at that instant, "
    "often half a sentence, so it need not describe what the still shows. The stills are small, so don't report "
    "softness or pixelation (pixel art is the style), and don't comment on the style.\n\n"
    + ANSWER.replace("WHERE", "Still N (counting from 1 in playing order)")
)

AUDIT_TEXT = (
    "You audit the template behind a faceless personal-finance YouTube channel (whiteboard drawings and a narrator; "
    "every long video has a companion Short). Titles, descriptions, chapter labels, tags and hashtags are generated "
    "by code, so one flaw in the template repeats in every video. You get counts computed across all videos "
    "(`stats`), one compact line per video (`videos`), and a few videos in full (`samples`).\n\n"
    "Find problems that repeat because of the template: wording in a fixed slot that is only right for some videos "
    "(for example a chapter label that names something specific but appears everywhere), boilerplate that is "
    "wrong, stale or duplicated, titles built to one pattern so they blur together, Shorts whose title or "
    "description doesn't stand on its own, hashtags or tags that are the same everywhere or miss the topic, "
    "descriptions that bury what the video is about. Ignore one-off slips in a single video.\n\n"
    "Conventions that are intended: a Short has four beats (hook, insight, answer, end) and its end line runs back "
    "into the hook so it loops; narration spells numbers out and captions use digits; chapter timestamps are "
    "generated.\n\n"
    + ANSWER.replace("WHERE", "the part of the template: Long title, Short title, Description, Chapters, Tags or "
                              "Hashtags")
    .replace('"fix": "the concrete change to make"', '"fix": "the change to make in the template", "videos": '
             "how many videos show it, as a number")
)

AUDIT_STILLS = (
    "You audit how the videos of a faceless personal-finance YouTube channel look. They are marker drawings on a "
    "white whiteboard; every video comes from the same template. You get thumbnails from several different videos, "
    "then frames from two long videos and two vertical Shorts.\n\n"
    "Find what repeats across them: thumbnail text that can't be read at small size, thumbnails so alike that "
    "viewers couldn't tell the videos apart, text that is cut off or overlaps in the same place in several images, "
    "large empty areas in every frame, captions covering the drawing. Say how many of the images show each problem. "
    "The stills are scaled down, so don't report softness. A frame caught mid-drawing is normal.\n\n"
    + ANSWER.replace("WHERE", "Thumbnails, Long video frames or Short frames")
)

STATS_TEXT = (
    "You read one week's view counts for a small YouTube channel (personal-finance whiteboard explainers; long "
    "videos and Shorts). Each row has the kind, the title, total views, views gained this week (when known) and the "
    "age in days. Name at most three patterns that the numbers support, each resting on at least three videos: for "
    "example a topic or a title wording that the better performers share and the weaker ones lack. Shorts and long "
    "videos are found by viewers in different ways, so beyond saying which kind does better, compare Shorts with "
    "Shorts and long videos with long videos. Older videos have had more time, so compare like with like. Each "
    "pattern is one sentence of at most 35 words that names its videos by a few words of their titles, with their "
    "views. If the numbers are too small or too even to support a pattern, say so and stop. Answer with JSON only: "
    '{"patterns": ["one sentence each"], "note": "one sentence when there is no pattern worth acting on, else '
    'empty"}.'
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start : end + 1]) if start >= 0 and end > start else None
    if not isinstance(data, dict):
        raise ValueError("no JSON object in the answer")
    return data


def _image(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


async def _ask(models: list[tuple[str, str]], system: str, content, what: str, timeout: float = 120) -> dict:
    """Send one request to the first model that answers with JSON. `content` is a string or a list of parts."""
    last = "no model with an API key is configured"
    # A model that just failed (overloaded, timed out) goes to the back of the line for a while.
    for provider, model in sorted(models, key=lambda m: _resting.get(m[1], 0) > time.time()):
        client = _client(provider)
        if client is None:
            continue
        for attempt in range(2):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0.2, response_format={"type": "json_object"}, timeout=timeout,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                )
                return _parse_json(r.choices[0].message.content or "")
            except APIStatusError as e:
                last = f"{model} answered HTTP {e.status_code}"
                try:
                    wait = float(e.response.headers.get("retry-after") or 0)
                except ValueError:
                    wait = 0
                if attempt == 0 and e.status_code == 429 and 0 < wait <= 70:  # a per-minute limit: wait it out
                    await asyncio.sleep(wait)
                    continue
                _resting[model] = time.time() + MODEL_REST_S
                break
            except ValueError:
                last = f"{model} gave an answer that could not be read"
                break
            except Exception as e:  # noqa: BLE001 - connection errors and timeouts: move on to the next model
                last = f"{model}: {type(e).__name__}"
                _resting[model] = time.time() + MODEL_REST_S
                break
        log.info("%s: %s; trying the next model", what, last)
    raise ReviewError(last)


# A model arguing with itself inside a finding ("wait, actually…") is not sure of it, so the finding is dropped.
SECOND_THOUGHTS = re.compile(r"\b(wait,|actually,|let'?s check|let me|on closer look|looking closely|however, looking)", re.I)


def _findings(answer: dict, source: str) -> list[dict]:
    out = []
    for f in answer.get("findings") or []:
        if not isinstance(f, dict) or not str(f.get("problem") or "").strip():
            continue
        if SECOND_THOUGHTS.search(str(f["problem"])):
            log.info("dropped an unsure finding: %s", str(f["problem"])[:120])
            continue
        finding = {
            "severity": "serious" if str(f.get("severity", "")).lower().startswith("s") else "minor",
            "where": str(f.get("where") or "").strip()[:80],
            "problem": str(f["problem"]).strip()[:600],
            "fix": str(f.get("fix") or "").strip()[:400],
            "source": source,
        }
        if str(f.get("videos") or "").strip().isdigit():
            finding["videos"] = int(f["videos"])
        out.append(finding)
    return sorted(out[:5], key=lambda f: f["severity"] != "serious")


async def _look(system: str, intro: str, images: list[tuple[str, str]]) -> tuple[list[dict], str]:
    """Vision pass over (label, base64 JPEG) pairs. Returns findings and a note on what was checked."""
    if not images:
        return [], "no stills to check"
    content = [{"type": "text", "text": intro}]
    for label, b64 in images:
        content += [{"type": "text", "text": label}, _image(b64)]
    try:
        answer = await _ask(VISION_MODELS, system, content, "stills", timeout=45)
    except ReviewError as e:
        return [], f"not checked ({e})"
    found = _findings(answer, "stills")
    for f in found:  # a still shows where to look; it can't prove a defect in the moving video
        f["severity"] = "minor"
    return found, "checked"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def load() -> dict:
    state = ops.load_state(STATE_FILE)
    state.setdefault("reviews", {})
    state.setdefault("stats", [])
    return state


def save(state: dict) -> None:
    reviews = state["reviews"]
    for key in sorted(reviews, key=lambda k: reviews[k].get("at", 0))[:-KEEP_REVIEWS]:
        del reviews[key]
    ops.save_state(state, STATE_FILE)


def stored(key: str) -> dict | None:
    return load()["reviews"].get(key)


def update_stored(key: str, **fields) -> None:
    state = load()
    if key in state["reviews"]:
        state["reviews"][key].update(fields)
        save(state)


def _store(rec: dict) -> dict:
    state = load()
    state["reviews"][rec["key"]] = rec
    save(state)
    return rec


# ---------------------------------------------------------------------------
# YouTube: a long video and its Short are reviewed together
# ---------------------------------------------------------------------------
def _groups(youtube: list[dict]) -> list[list[dict]]:
    """Whiteboard items as [long, short...] groups; a Short whose long video is gone stands alone."""
    items = [i for i in youtube if i.get("episode")]
    groups = {i["id"]: [i] for i in items if i["kind"] == "long"}
    alone = []
    for i in items:
        if i["kind"] == "long":
            continue
        if i.get("parent") in groups:
            groups[i["parent"]].append(i)
        else:
            alone.append([i])
    return list(groups.values()) + alone


async def _review_video(item: dict, long_title: str | None) -> dict:
    kind = item["kind"]
    d = await ops.call(
        ops.AUTOPILOT_HOST, "review-item", item["id"], str(LONG_FRAMES if kind == "long" else SHORT_FRAMES), timeout=300
    )
    beats = []
    for b in d.get("beats") or []:
        beat = {"id": b.get("id"), "says": b.get("text")}
        if kind != "long" and b.get("cap"):
            beat["caption"] = b["cap"]
        if b.get("board"):
            beat["board"] = b["board"]
        beats.append(beat)
    meta = d.get("meta") or {}
    brief = d.get("brief") or {}
    if kind != "long":  # the promise and audience describe the long video, not this slice of it
        brief = {k: brief.get(k) for k in ("assumptions", "facts")}
    material = {
        "kind": "Short (vertical, loops)" if kind != "long" else "long video",
        "title": meta.get("title") or d["title"],
        "description": meta.get("description"),
        "tags": meta.get("tags"),
        "length_seconds": d.get("duration"),
        "brief": brief,
        "beats": beats,
    }
    if meta.get("hashtags"):
        material["hashtags"] = meta["hashtags"]
    if long_title:
        material["long_video_it_points_to"] = long_title
    answer = await _ask(TEXT_MODELS, YT_TEXT, json.dumps(material, ensure_ascii=False), f"script of {item['id']}")
    images = [("Thumbnail:", d["thumbnail"])] if d.get("thumbnail") else []
    images += [(f"Frame at {_clock(f['t'])}:", f["jpg"]) for f in d.get("frames") or []]
    seen, stills = await _look(YT_STILLS, f"Title: {material['title']} ({material['kind']})", images)
    return {
        "id": item["id"], "kind": kind, "title": material["title"], "duration": d.get("duration"),
        "summary": str(answer.get("summary") or "").strip()[:300],
        "findings": _findings(answer, "script") + seen,
        "stills": stills, "thumbnail": bool(d.get("thumbnail")), "frames": len(d.get("frames") or []),
    }


async def review_youtube(item_id: str) -> dict:
    """Review the long video `item_id` belongs to together with its Short. Stores and returns the record."""
    async with LOCK:
        listing = await ops.call(ops.AUTOPILOT_HOST, "review-list")
        group = next((g for g in _groups(listing["youtube"]) if any(i["id"] == item_id for i in g)), None)
        if group is None:
            raise ReviewError(f"no whiteboard video with the id {item_id}")
        lead = group[0]
        sections = []
        for item in group:
            sections.append(await _review_video(item, lead["title"] if item is not lead else None))
        return _store({
            "key": f"yt:{lead['id']}", "kind": "youtube", "id": lead["id"], "title": lead["title"], "at": time.time(),
            "status": lead["status"], "publish_at": lead.get("publish_at"), "url": lead.get("url"),
            "sections": sections,
        })


# ---------------------------------------------------------------------------
# Lanternlight
# ---------------------------------------------------------------------------
def lantern_key(slug: str, part: int) -> str:
    return f"ll:{slug}:{part}"


async def review_lantern(slug: str, part: int, built: float | None = None) -> dict:
    async with LOCK:
        d = await ops.call(ops.LANTERN_HOST, "review-lantern", slug, str(part), timeout=120)
        sheet = d.pop("sheet", None)
        final = d["part"] == d["of"]
        material = {
            "story": d["title"], "premise": d.get("premise"), "bible": d.get("bible"),
            "earlier_parts": d.get("earlier"),
            "parts_still_to_come": d.get("later"),
            "this_part": {"number": d["part"], "of": d["of"], "final_part": final, **(d.get("this") or {})},
            "narration_lines": d.get("lines"),
            "captions": {"facebook": d["built"].get("caption_fb"), "instagram": d["built"].get("caption_ig")},
            "length_seconds": d["built"].get("duration"),
        }
        answer = await _ask(TEXT_MODELS, LL_TEXT, json.dumps(material, ensure_ascii=False), f"story {slug}:{part}")
        characters = json.dumps((d.get("bible") or {}).get("characters"), ensure_ascii=False)
        seen, stills = await _look(
            LL_STILLS, f"{d['title']}, part {d['part']} of {d['of']}. Characters: {characters}",
            [("Contact sheet:", sheet)] if sheet else [],
        )
        name = (d.get("this") or {}).get("name")
        return _store({
            "key": lantern_key(slug, part), "kind": "lantern", "slug": slug, "part": part, "of": d["of"],
            "title": f"{d['title']}, part {d['part']} of {d['of']}" + (f": {name}" if name else ""),
            "at": time.time(), "built": built, "posted": d.get("posted"),
            "sections": [{
                "kind": "reel", "title": name, "duration": d["built"].get("duration"),
                "summary": str(answer.get("summary") or "").strip()[:300],
                "findings": _findings(answer, "story") + seen, "stills": stills, "frames": 1 if sheet else 0,
            }],
        })


# ---------------------------------------------------------------------------
# What is waiting. Nothing is reviewed until the owner picks it.
# ---------------------------------------------------------------------------
async def targets() -> dict:
    """Videos that have not gone out yet: queued YouTube groups, and the next Lanternlight parts to be posted.

    `lantern_more` counts the built parts further down the queue that are not listed.
    """
    listing = await ops.call(ops.AUTOPILOT_HOST, "review-list")
    reviews = load()["reviews"]
    youtube = []
    for group in _groups(listing["youtube"]):
        lead = group[0]
        if lead["status"] in ("queued", "held", "blocked"):
            youtube.append({**lead, "shorts": len(group) - 1, "review": reviews.get(f"yt:{lead['id']}")})
    youtube.sort(key=lambda i: i.get("publish_at") or 0)
    lantern = []
    unposted = [p for p in listing["lantern"] if not p["posted"]]
    for p in unposted[:LANTERN_AHEAD]:
        rec = reviews.get(lantern_key(p["slug"], p["part"]))
        lantern.append({**p, "review": rec if rec and rec.get("built") == p["built"] else None})
    return {"youtube": youtube, "lantern": lantern, "lantern_more": len(unposted) - len(lantern)}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _clock(seconds: float | None) -> str:
    m, s = divmod(int(seconds or 0), 60)
    return f"{m}:{s:02d}"


def _when(ts: float | None) -> str:
    if not ts:
        return "no publish time"
    local = datetime.fromtimestamp(ts, timezone.utc) + UTC_OFFSET
    left = ts - time.time()
    return local.strftime("%a %d %b %H:%M") + (f" (in {ops.fmt_age(left)})" if left > 0 else "")


def _count(findings: list[dict]) -> str:
    serious = sum(f["severity"] == "serious" for f in findings)
    minor = len(findings) - serious
    parts = [f"{serious} serious"] * bool(serious) + [f"{minor} minor"] * bool(minor)
    return ", ".join(parts) or "nothing to fix"


def _finding_lines(findings: list[dict]) -> list[str]:
    lines = []
    for n, f in enumerate(findings, 1):
        mark = "❗" if f["severity"] == "serious" else "👁" if f["source"] == "stills" else "▫️"
        where = f"{f['where']}: " if f["where"] else ""
        seen = f" (about {f['videos']} videos)" if f.get("videos") else ""
        lines.append(f"{n}. {mark} {where}{f['problem']}{seen}")
        if f["fix"]:
            lines.append(f"    Fix: {f['fix']}")
    return lines


def verdict(rec: dict) -> str:
    return _count([f for s in rec["sections"] for f in s["findings"]])


def format_review(rec: dict) -> str:
    youtube = rec["kind"] == "youtube"
    head = [("🎬 " if youtube else "🏮 ") + rec["title"]]
    if youtube:
        state = {"queued": "Publishes", "held": "Held, planned for", "blocked": "Blocked, was due"}.get(rec.get("status"))
        head.append(f"{state} {_when(rec.get('publish_at'))}" if state else f"Published: {rec.get('url') or 'yes'}")
    else:
        head.append("Already posted" if rec.get("posted") else "Not posted yet")
    out = ["\n".join(head)]
    unchecked = []
    for s in rec["sections"]:
        label = {"long": "Long video", "short": "Short", "reel": "Reel"}.get(s["kind"], s["kind"])
        named = f' "{s["title"]}"' if s["kind"] == "short" and s.get("title") else ""
        block = [f"{label}{named} ({_clock(s.get('duration'))}): {_count(s['findings'])}"]
        block += _finding_lines(s["findings"])
        if not s["findings"] and s.get("summary"):
            block.append(s["summary"])
        out.append("\n".join(block))
        if s["stills"] != "checked":
            unchecked.append(f"{label}: stills {s['stills']}")
    if unchecked:
        out.append("\n".join(unchecked) + "\nOnly the text was reviewed there. \"Review again\" retries the stills.")
    else:
        checked = "Script, title, description, thumbnail and frames" if youtube else "Story, captions and contact sheet"
        out.append(f"{checked} were checked. 👁 marks something seen in a still: a place to look, not a verdict.")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Template audit across every built video
# ---------------------------------------------------------------------------
CHAPTER = re.compile(r"^\s*(\d{1,2}:\d\d(?::\d\d)?)\s+(.+?)\s*$")


def _spread(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def _template_stats(episodes: list[dict]) -> tuple[dict, list[dict]]:
    """Counts across all videos for the model, and the findings that need no judgement."""
    n = len(episodes)
    longs = [e["long"] for e in episodes if e.get("long")]
    shorts = [e["short"] for e in episodes if e.get("short")]

    def meta(d: dict, key: str):
        return (d.get("meta") or {}).get(key) or ("" if key in ("title", "description") else [])

    def shared_lines(videos: list[dict]) -> dict:
        lines = Counter()
        for d in videos:
            lines.update({l.strip() for l in meta(d, "description").splitlines() if l.strip() and not CHAPTER.match(l)})
        return {l: c for l, c in lines.most_common(12) if c >= max(2, 0.8 * len(videos))}

    labels = Counter()
    for d in longs:
        labels.update({m.group(2) for m in map(CHAPTER.match, meta(d, "description").splitlines()) if m})
    titles = Counter(meta(d, "title").strip().lower() for d in longs + shorts if meta(d, "title"))
    hashtags = Counter(h.lower() for d in shorts for h in meta(d, "hashtags"))
    tags = Counter(t.lower() for d in longs for t in meta(d, "tags"))

    def seconds(videos: list[dict]) -> dict:
        ds = sorted(d["duration"] for d in videos if d.get("duration"))
        return {"shortest": ds[0], "median": statistics.median(ds), "longest": ds[-1]} if ds else {}

    stats = {
        "episodes": n, "long_videos": len(longs), "shorts": len(shorts),
        "chapter_labels_and_how_many_videos_use_each": dict(labels.most_common(20)),
        "description_lines_shared_by_most_long_videos": shared_lines(longs),
        "description_lines_shared_by_most_shorts": shared_lines(shorts),
        "long_title_lengths": sorted(len(meta(d, "title")) for d in longs),
        "short_hashtags_and_how_many_shorts_use_each": dict(hashtags.most_common(15)),
        "tags_used_by_most_long_videos": {t: c for t, c in tags.most_common(15) if c >= 0.5 * len(longs)},
        "long_video_seconds": seconds(longs), "short_seconds": seconds(shorts),
    }

    def certain(where: str, problem: str, fix: str, videos: list, serious: bool = False) -> dict:
        names = ", ".join(f'"{v}"' for v in videos[:3]) + (f" and {len(videos) - 3} more" if len(videos) > 3 else "")
        return {"severity": "serious" if serious else "minor", "where": where, "problem": f"{problem}: {names}",
                "fix": fix, "videos": len(videos), "source": "counted"}

    found = []
    checks = [
        ("Long title", "The same title is used by more than one video", "Make each title unique.",
         [t for t, c in titles.items() if c > 1], True),
        ("Long title", "Title over 100 characters, which YouTube rejects", "Shorten the title.",
         [meta(d, "title") for d in longs + shorts if len(meta(d, "title")) > 100], True),
        ("Long title", "Title over 70 characters, so search results cut it off", "Keep titles to 70 characters.",
         [meta(d, "title") for d in longs if 70 < len(meta(d, "title")) <= 100], False),
        ("Tags", "Tags add up to more than YouTube's 500 characters", "Trim the tag list.",
         [d["title"] for d in longs if sum(len(t) + 1 for t in meta(d, "tags")) > 500], False),
        ("Thumbnail", "Long video without a thumbnail file", "Render the thumbnail before queueing.",
         [d["title"] for d in longs if not d.get("thumbnail")], True),
        ("Description", "Video without a description", "Check the description step.",
         [d["title"] for d in longs + shorts if not meta(d, "description").strip()], True),
        ("Short title", "Short with the same title as its long video", "Give the Short its own title.",
         [e["short"]["title"] for e in episodes if e.get("long") and e.get("short")
          and meta(e["short"], "title").strip().lower() == meta(e["long"], "title").strip().lower()], False),
        ("Short", "Short longer than 3 minutes, so YouTube won't treat it as a Short", "Cut it under 3 minutes.",
         [d["title"] for d in shorts if (d.get("duration") or 0) > 180], True),
    ]
    for where, problem, fix, videos, serious in checks:
        if videos:
            found.append(certain(where, problem, fix, videos, serious))
    return stats, found


async def audit(progress=None) -> dict:
    """Look across every built video for problems the template repeats. Stores and returns the record."""
    async def say(text: str) -> None:
        if progress:
            await progress(text)

    async with LOCK:
        listing = await ops.call(ops.AUTOPILOT_HOST, "review-list")
        groups = [g for g in _groups(listing["youtube"]) if g[0]["status"] in ("queued", "held", "published", "blocked")]
        gate = asyncio.Semaphore(3)

        async def fetch(item_id: str, frames: int = 0) -> dict:
            async with gate:
                return await ops.call(ops.AUTOPILOT_HOST, "review-item", item_id, str(frames), timeout=300)

        await say(f"Reading {sum(len(g) for g in groups)} videos from {len(groups)} episodes…")
        wanted = [i for g in groups for i in g[:2]]
        details = dict(zip((i["id"] for i in wanted), await asyncio.gather(*(fetch(i["id"]) for i in wanted))))
        episodes = []
        for g in sorted(groups, key=lambda g: g[0].get("publish_at") or 0):
            lead = details[g[0]["id"]]
            other = details[g[1]["id"]] if len(g) > 1 else None
            episodes.append({"long": lead if lead["kind"] == "long" else None,
                             "short": other if lead["kind"] == "long" else lead})
        stats, findings = _template_stats(episodes)

        def compact(e: dict) -> dict:
            lg, sh = e.get("long") or {}, e.get("short") or {}
            desc = ((lg.get("meta") or {}).get("description") or "").splitlines()
            return {
                "long_title": (lg.get("meta") or {}).get("title"),
                "short_title": (sh.get("meta") or {}).get("title"),
                "description_first_line": next((l for l in desc if l.strip()), "")[:200],
                "chapters": [m.group(2) for m in map(CHAPTER.match, desc) if m],
                "short_hashtags": (sh.get("meta") or {}).get("hashtags"),
            }

        def full(e: dict) -> dict:
            lg, sh = e.get("long") or {}, e.get("short") or {}
            return {
                "long": {**{k: (lg.get("meta") or {}).get(k) for k in ("title", "description", "tags")},
                         "beats": [{"id": b.get("id"), "chapter": b.get("chapter"), "says": (b.get("text") or "")[:160]}
                                   for b in lg.get("beats") or []]},
                "short": {**{k: (sh.get("meta") or {}).get(k) for k in ("title", "description", "hashtags")},
                          "beats": [{"id": b.get("id"), "says": b.get("text")} for b in sh.get("beats") or []]},
            }

        await say("Comparing titles, descriptions, chapters and tags across all of them…")
        material = {"stats": stats, "videos": [compact(e) for e in episodes],
                    "samples": [full(e) for e in _spread([e for e in episodes if e.get("long")], 3)]}
        text_note = "checked"
        try:
            answer = await _ask(TEXT_MODELS, AUDIT_TEXT, json.dumps(material, ensure_ascii=False), "template audit")
            findings += _findings(answer, "template")
        except ReviewError as e:
            answer, text_note = {}, f"not checked ({e})"

        await say("Looking at thumbnails and frames…")
        with_long = [e for e in episodes if e.get("long")]
        images = [(f"Thumbnail of \"{e['long']['title']}\":", e["long"]["thumbnail"])
                  for e in _spread([e for e in with_long if e["long"].get("thumbnail")], 6)]
        picks = [(e["long"], 4) for e in _spread(with_long, 2)]
        picks += [(e["short"], 3) for e in _spread([e for e in episodes if e.get("short")], 2)]
        for d, frames in zip((p[0] for p in picks), await asyncio.gather(*(fetch(d["id"], n) for d, n in picks))):
            what = "long video" if d["kind"] == "long" else "Short"
            images += [(f"Frame at {_clock(f['t'])} of the {what} \"{d['title']}\":", f["jpg"]) for f in frames.get("frames") or []]
        seen, stills = await _look(AUDIT_STILLS, f"{len(images)} stills from {len(episodes)} videos made by one template.", images)

        rec = {
            "at": time.time(), "episodes": len(episodes), "videos": len(wanted),
            "summary": str(answer.get("summary") or "").strip()[:400],
            "findings": findings + seen, "text": text_note, "stills": stills, "images": len(images),
        }
        state = load()
        state["audit"] = rec
        save(state)
        return rec


def format_audit(rec: dict) -> str:
    when = (datetime.fromtimestamp(rec["at"], timezone.utc) + UTC_OFFSET).strftime("%a %d %b %H:%M")
    out = [f"📊 Template audit: {rec['videos']} videos from {rec['episodes']} episodes ({when})\n{_count(rec['findings'])}"]
    if rec.get("summary"):
        out.append(rec["summary"])
    by_source = {"counted": "Counted across every video", "template": "Titles, descriptions, chapters and tags",
                 "stills": "Thumbnails and frames"}
    for source, label in by_source.items():
        found = [f for f in rec["findings"] if f["source"] == source]
        if found:
            out.append(label + "\n" + "\n".join(_finding_lines(found)))
    notes = []
    if rec["text"] != "checked":
        notes.append(f"Titles and descriptions: {rec['text']}.")
    if rec["stills"] != "checked":
        notes.append(f"Thumbnails and frames: {rec['stills']}.")
    else:
        notes.append(f"{rec['images']} stills were compared; scripts were read in full for 3 videos, the rest by "
                     "title, description, chapters and tags. 👁 marks something seen in stills: a place to look, "
                     "not a verdict.")
    out.append("\n".join(notes))
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Weekly channel numbers
# ---------------------------------------------------------------------------
def weekly_due(now: float | None = None) -> bool:
    """Monday morning local time, or right away the first time, or when a Monday was missed."""
    now = now or time.time()
    last = load().get("weekly_at")
    if not last:
        return True
    local = datetime.fromtimestamp(now, timezone.utc) + UTC_OFFSET
    return (local.weekday() == 0 and local.hour >= 9 and now - last > 3 * 86400) or now - last > 8 * 86400


async def stats_report(weekly: bool = False) -> tuple[str, dict]:
    """Views per video against the last weekly snapshot. Returns the report and the snapshot it was made from."""
    data = await ops.call(ops.AUTOPILOT_HOST, "yt-stats", timeout=150)
    videos = data["videos"]
    state = load()
    previous = state["stats"][-1] if state["stats"] else None
    before = (previous or {}).get("views", {})
    now = time.time()
    for v in videos:
        v["gain"] = v["views"] - before[v["vid"]] if v["vid"] in before else None
        try:
            v["age_days"] = (now - datetime.strptime(v["published"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()) / 86400
        except (KeyError, ValueError):
            v["age_days"] = None

    def total(kind: str | None, key: str = "views") -> int:
        return sum(v[key] or 0 for v in videos if kind in (None, v["kind"]))

    def line(kind: str, label: str) -> str:
        count = sum(v["kind"] == kind for v in videos)
        gained = f" (+{total(kind, 'gain'):,})" if previous else ""
        return f"{label}: {count}, {total(kind):,} views{gained}"

    def name(v: dict) -> str:
        title = v["title"] if len(v["title"]) <= 72 else v["title"][:70].rstrip() + "…"
        return f"{title} ({'Short' if v['kind'] == 'short' else 'long'})"

    since = f"since {ops.fmt_age(now - previous['at'])} ago" if previous else "so far"
    gained = f", +{total(None, 'gain'):,} {since}" if previous else ""
    out = ["📈 YouTube numbers\n"
           f"{len(videos)} videos, {total(None):,} views{gained}, {total(None, 'likes'):,} likes, "
           f"{total(None, 'comments'):,} comments\n{line('long', 'Long videos')}\n{line('short', 'Shorts')}"]
    if previous:
        movers = sorted((v for v in videos if v["gain"]), key=lambda v: -v["gain"])[:5]
        if movers:
            out.append(f"Most views {since}\n" + "\n".join(f"{n}. +{v['gain']:,} {name(v)}, {v['views']:,} in total"
                                                           for n, v in enumerate(movers, 1)))
        else:
            out.append(f"No video gained views {since}.")
        new = [v for v in videos if v["vid"] not in before]
        if new:
            out.append("New since the last report\n" + "\n".join(f"- {name(v)}: {v['views']:,} views" for v in new[:8]))
    else:
        top = sorted(videos, key=lambda v: -v["views"])[:5]
        out.append("Most viewed so far\n" + "\n".join(f"{n}. {v['views']:,} {name(v)}" for n, v in enumerate(top, 1)))
    quiet = [v for v in videos if v["views"] < 10 and (v["age_days"] or 0) > 3]
    if quiet:
        out.append(f"Under 10 views after 3 days or more: {len(quiet)} videos\n" + "\n".join(f"- {name(v)}" for v in quiet[:5]))

    if data.get("analytics"):
        a = data["analytics"]
        kept = sorted((v for v in videos if v.get("last28")), key=lambda v: -(v["last28"].get("averageViewPercentage") or 0))
        block = [f"Last 28 days: {a.get('estimatedMinutesWatched', 0):,.0f} minutes watched, "
                 f"{a.get('subscribersGained', 0) - a.get('subscribersLost', 0):+,.0f} subscribers"]
        block += [f"- {v['last28'].get('averageViewPercentage', 0):.0f}% watched on average: {name(v)}" for v in kept[:3]]
        if len(kept) > 5:
            block += [f"- {v['last28'].get('averageViewPercentage', 0):.0f}% watched on average: {name(v)}" for v in kept[-2:]]
        out.append("\n".join(block))
    elif data.get("analytics_error") == "disabled":
        out.append("Watch time and retention are missing because the YouTube Analytics API is switched off in the "
                   "Google Cloud project. Enable it here and the next report includes them:\n"
                   + (data.get("analytics_enable_url") or "Google Cloud console > APIs > YouTube Analytics API"))
    elif data.get("analytics_error"):
        out.append(f"Watch time and retention could not be read: {data['analytics_error']}")

    rows = [{"kind": v["kind"], "title": v["title"], "views": v["views"], "gained": v["gain"],
             "age_days": round(v["age_days"]) if v["age_days"] is not None else None} for v in videos]
    try:
        answer = await _ask(TEXT_MODELS, STATS_TEXT, json.dumps(rows, ensure_ascii=False), "stats patterns")
        patterns = [str(p).strip() for p in answer.get("patterns") or [] if str(p).strip()][:3]
        if patterns:
            out.append("What the numbers suggest\n" + "\n".join(f"- {p}" for p in patterns))
        elif str(answer.get("note") or "").strip():
            out.append(str(answer["note"]).strip())
    except ReviewError as e:
        log.info("no patterns in the stats report: %s", e)

    if not previous:
        out.append("This is the starting point; from the next weekly report on you'll see what each video gained."
                   if weekly else "The weekly report hasn't run yet, so there is nothing to compare against.")
    return "\n\n".join(out), {"at": now, "views": {v["vid"]: v["views"] for v in videos}}


def record_weekly(snapshot: dict) -> None:
    """Keep the snapshot of a weekly report that was delivered; the next report compares against it."""
    state = load()
    state["stats"] = (state["stats"] + [snapshot])[-26:]
    state["weekly_at"] = snapshot["at"]
    save(state)
