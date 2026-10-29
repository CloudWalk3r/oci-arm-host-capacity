"""
Telegram AI helper bot.

Routes chats to local Ollama models (private, on the ARM box over the VCN)
or free hosted APIs (Groq, Google Gemini) via one OpenAI-compatible client.

Commands: /start /help /ask /code /summarize /translate /search /image /deals
/model /reset /tools /status /about  (plus hidden /local /whoami).
"""
import asyncio
import base64
import json
import logging
import math
import os
import re
import shutil
import time
from difflib import SequenceMatcher
from io import BytesIO
from urllib.parse import quote

import httpx
from openai import AsyncOpenAI
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from downloader import (
    JDClient,
    Job,
    MYJD_AVAILABLE,
    SUPPORTED_HOSTS,
    detect_host,
    extract_urls,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
log = logging.getLogger("ai-bot")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://10.0.1.21:11434/v1").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# GitHub Models: a GitHub PAT (classic or fine-grained) with the "models" permission.
GITHUB_MODELS_TOKEN = os.getenv("GITHUB_MODELS_TOKEN", "").strip()
# GitHub Models' endpoint currently returns a placeholder "OK" (API change/outage), so it's
# gated off by default. Set GITHUB_MODELS_ENABLED=1 to re-enable once it serves completions.
GITHUB_MODELS_ENABLED = os.getenv("GITHUB_MODELS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")

# Gemini native endpoint (used for image generation; chat uses the OpenAI-compatible one)
GEMINI_NATIVE = "https://generativelanguage.googleapis.com/v1beta"
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-2.5-flash-image").strip()
# Gemini's free tier grants no image-generation quota (image models are effectively
# paid-only and return HTTP 429), so /image is gated off by default. Set IMAGE_ENABLED=1
# once you have a key with image quota (e.g. a billed Gemini project).
IMAGE_ENABLED = os.getenv("IMAGE_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")

# /deals: Steam regional price + multi-store deals (CheapShark, no key; IsThereAnyDeal, optional key)
STEAM_CC = os.getenv("STEAM_CC", "IN").strip().upper()
ITAD_API_KEY = os.getenv("ITAD_API_KEY", "").strip()
ITAD_COUNTRY = os.getenv("ITAD_COUNTRY", STEAM_CC).strip().upper()
# CheapShark rejects requests without a descriptive User-Agent.
DEALS_UA = "ai-bot-deals/1.0 (personal Telegram bot)"
DEALS_MAX = 10

# /watch: persistent price-drop alerts, checked on a schedule by the job queue.
WATCHES_FILE = os.getenv("WATCHES_FILE", "data/watches.json").strip()
WATCH_INTERVAL_HOURS = float(os.getenv("WATCH_INTERVAL_HOURS", "12") or 12)
MAX_WATCHES_PER_CHAT = 25

ALLOWED_USER_IDS = {
    int(x.strip())
    for x in os.getenv("ALLOWED_USER_IDS", "").split(",")
    if x.strip().isdigit()
}

# /dl: download big files (FuckingFast/DataNodes/FileKeeper) via JDownloader, upload to Telegram.
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID", "").strip()
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "").strip()
# Local Bot API server base (e.g. http://telegram-bot-api:8081). Blank => public API, 50 MB cap.
TELEGRAM_BOT_API_BASE = os.getenv("TELEGRAM_BOT_API_BASE", "").strip().rstrip("/")
LOCAL_BOT_API = bool(TELEGRAM_BOT_API_BASE)
MYJD_EMAIL = os.getenv("MYJD_EMAIL", "").strip()
MYJD_PASSWORD = os.getenv("MYJD_PASSWORD", "").strip()
MYJD_DEVICE = os.getenv("MYJD_DEVICE", "").strip()
# Public api.telegram.org caps bot uploads at 50 MB; a local server allows up to ~2000 MB.
DL_MAX_FILE_MB = int(os.getenv("DL_MAX_FILE_MB", "2000" if LOCAL_BOT_API else "49") or 2000)
DL_AVAILABLE = MYJD_AVAILABLE and bool(MYJD_EMAIL and MYJD_PASSWORD and MYJD_DEVICE)
DL_POLL_SECONDS = 5
# Long transfers: sending a 2 GB file through the local server takes a while.
DL_SEND_TIMEOUT = float(os.getenv("DL_SEND_TIMEOUT", "3600") or 3600)

SYSTEM_PROMPT = (
    "You are a helpful assistant for academic and software development work. "
    "Be concise, accurate, and show code or steps when useful."
)
CODE_PROMPT = (
    "You are an expert programming assistant. Provide correct, idiomatic code with brief "
    "explanations. Use fenced code blocks. When debugging, identify the root cause first."
)
MAX_HISTORY_MESSAGES = 20
TG_MAX = 4096

# Optional web search
try:
    from ddgs import DDGS  # noqa: F401
    SEARCH_AVAILABLE = True
except Exception:  # pragma: no cover
    SEARCH_AVAILABLE = False

try:
    from bs4 import BeautifulSoup  # noqa: F401
    BS4_AVAILABLE = True
except Exception:  # pragma: no cover
    BS4_AVAILABLE = False

# provider -> OpenAI-compatible connection
PROVIDERS = {
    "ollama": {"base_url": OLLAMA_BASE_URL, "api_key": "ollama"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key": GROQ_API_KEY},
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_API_KEY,
    },
    "github": {
        "base_url": "https://models.github.ai/inference",
        "api_key": GITHUB_MODELS_TOKEN,
        "enabled": GITHUB_MODELS_ENABLED,
        "headers": {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    },
}

# friendly key -> (provider, real model id, human label)
MODELS = {
    "groq-gptoss120b": ("groq", "openai/gpt-oss-120b", "Groq GPT-OSS 120B (fast, strong)"),
    "groq-gptoss20b": ("groq", "openai/gpt-oss-20b", "Groq GPT-OSS 20B (fast)"),
    "groq-qwen27b": ("groq", "qwen/qwen3.8-27b", "Groq Qwen 3.8 27B"),
    "gemini-flash": ("gemini", "gemini-flash-latest", "Gemini Flash latest (hosted)"),
    "gemini-pro": ("gemini", "gemini-pro-latest", "Gemini Pro latest (hosted)"),
    "gpt-4o": ("github", "openai/gpt-4o", "GPT-4o (GitHub Models)"),
    "gpt-4o-mini": ("github", "openai/gpt-4o-mini", "GPT-4o mini (GitHub Models, fast)"),
    "gh-llama70b": ("github", "meta/Llama-3.3-70B-Instruct", "Llama 3.3 70B (GitHub Models)"),
    "gh-deepseek": ("github", "deepseek/DeepSeek-V3-0324", "DeepSeek V3 (GitHub Models)"),
    "qwen-coder": ("ollama", "qwen2.5-coder:7b", "Qwen2.5 Coder 7B (local, private)"),
    "llama3.2": ("ollama", "llama3.2:3b", "Llama 3.2 3B (local, private, fast)"),
}
DEFAULT_LOCAL = "qwen-coder"


def provider_available(provider: str) -> bool:
    if provider == "ollama":
        return True
    conf = PROVIDERS[provider]
    if not conf["api_key"]:
        return False
    return conf.get("enabled", True)


def available_models() -> dict:
    return {k: v for k, v in MODELS.items() if provider_available(v[0])}


def pick_default() -> str:
    env_default = os.getenv("DEFAULT_MODEL", "").strip()
    avail = available_models()
    if env_default in avail:
        return env_default
    for key in ("groq-gptoss120b", "gemini-flash", "gpt-4o", "groq-gptoss20b", "qwen-coder", "llama3.2"):
        if key in avail:
            return key
    return next(iter(avail))


DEFAULT_MODEL = pick_default()
IMAGE_AVAILABLE = bool(GEMINI_API_KEY) and IMAGE_ENABLED

STATE: dict[int, dict] = {}


def chat_state(chat_id: int) -> dict:
    return STATE.setdefault(chat_id, {"model": DEFAULT_MODEL, "history": []})


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def allowed(update: Update) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    user = update.effective_user
    return bool(user and user.id in ALLOWED_USER_IDS)


async def deny(update: Update) -> None:
    uid = update.effective_user.id if update.effective_user else "?"
    await update.effective_message.reply_text(
        f"Not authorized. Your Telegram user ID is {uid}.\n"
        "Ask the owner to add it to ALLOWED_USER_IDS."
    )


# ---------------------------------------------------------------------------
# LLM core
# ---------------------------------------------------------------------------
async def stream_completion(model_key: str, messages: list):
    provider, model_id, _ = MODELS[model_key]
    conf = PROVIDERS[provider]
    client = AsyncOpenAI(
        base_url=conf["base_url"],
        api_key=conf["api_key"] or "none",
        default_headers=conf.get("headers"),
    )
    stream = await client.chat.completions.create(
        model=model_id, messages=messages, stream=True, temperature=0.4
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def generate_and_reply(update, ctx, messages, model_key, *, remember=False, user_text=None):
    """Stream a completion into a Telegram message (editing as tokens arrive)."""
    chat_id = update.effective_chat.id
    await ctx.bot.send_chat_action(chat_id, ChatAction.TYPING)
    placeholder = await update.effective_message.reply_text("...")

    full = ""
    last_edit = 0.0
    try:
        async for delta in stream_completion(model_key, messages):
            full += delta
            now = time.monotonic()
            if now - last_edit >= 1.2 and full.strip():
                last_edit = now
                try:
                    await placeholder.edit_text(full[:TG_MAX])
                except Exception:
                    pass
    except Exception as e:
        log.exception("completion failed")
        await placeholder.edit_text(f"Error from {MODELS[model_key][2]}:\n{e}")
        return None

    if not full.strip():
        full = "(empty response)"
    chunks = [full[i : i + TG_MAX] for i in range(0, len(full), TG_MAX)]
    try:
        await placeholder.edit_text(chunks[0])
    except Exception:
        pass
    for extra in chunks[1:]:
        await update.effective_message.reply_text(extra)

    if remember and user_text is not None:
        st = chat_state(chat_id)
        st["history"].append({"role": "user", "content": user_text})
        st["history"].append({"role": "assistant", "content": full})
        st["history"] = st["history"][-MAX_HISTORY_MESSAGES:]
    return full


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    st = chat_state(update.effective_chat.id)
    await update.effective_message.reply_text(
        "Hi! I'm your AI helper for academic work and software development.\n\n"
        f"Current model: {MODELS[st['model']][2]}\n\n"
        "Just send a message to chat, or use a command:\n"
        "/ask - ask anything\n"
        "/code - coding help & debugging\n"
        "/summarize - summarize text or a link\n"
        "/translate - translate text\n"
        "/search - search the web\n"
        "/image - generate an image\n"
        "/deals - best prices for a game\n"
        "/watch - alert me when a game gets cheap\n"
        "/dl - download a file link to Telegram\n"
        "/model - switch AI model\n"
        "/reset - clear memory\n"
        "/tools - capabilities\n"
        "/status - system status\n"
        "/about - about this bot\n"
        "/help - full help"
    )


async def cmd_help(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    await update.effective_message.reply_text(
        "How to use me:\n\n"
        "- Send any message to chat (keeps context).\n"
        "/ask <question> - one-off question\n"
        "/code <request> - coding help, debugging, generation\n"
        "/summarize <text or URL> - summarize; or reply to a message with /summarize\n"
        "/translate <language> <text> - e.g. /translate spanish Good morning\n"
        "/search <query> - web search + AI synthesis of current info\n"
        "/image <prompt> - generate an image\n"
        "/deals <game name> - Steam price + best deals across stores\n"
        "/watch <game name> <price> - alert when it drops to/below that price\n"
        "/watches - list your active price watches\n"
        "/unwatch <number> - remove a watch (see /watches)\n"
        "/dl <link> [more links] - download from FuckingFast/DataNodes/FileKeeper and send here;\n"
        "    paste several at once, reply to a message of links, or send /dl alone to pick a host.\n"
        "    Then /dlstatus and /dlcancel\n"
        "/model [key] - list or switch models (see /model)\n"
        "/reset - start a fresh conversation\n"
        "/tools - list capabilities\n"
        "/status - check model/server status\n"
        "/about - about this bot"
    )


async def cmd_ask(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    text = " ".join(ctx.args).strip()
    if not text:
        return await update.effective_message.reply_text("Usage: /ask <your question>")
    st = chat_state(update.effective_chat.id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += st["history"][-MAX_HISTORY_MESSAGES:]
    messages.append({"role": "user", "content": text})
    await generate_and_reply(update, ctx, messages, st["model"], remember=True, user_text=text)


async def cmd_code(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    text = " ".join(ctx.args).strip()
    if not text:
        return await update.effective_message.reply_text("Usage: /code <what you need help with>")
    st = chat_state(update.effective_chat.id)
    messages = [
        {"role": "system", "content": CODE_PROMPT},
        {"role": "user", "content": text},
    ]
    await generate_and_reply(update, ctx, messages, st["model"])


async def cmd_summarize(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    arg = " ".join(ctx.args).strip()
    reply = update.effective_message.reply_to_message
    source = ""
    if arg.startswith("http://") or arg.startswith("https://"):
        try:
            source = await asyncio.to_thread(fetch_url_text, arg)
        except Exception as e:
            return await update.effective_message.reply_text(f"Couldn't fetch that URL: {e}")
    elif arg:
        source = arg
    elif reply and reply.text:
        source = reply.text
    if not source.strip():
        return await update.effective_message.reply_text(
            "Usage: /summarize <text or URL>, or reply to a message with /summarize"
        )
    source = source[:12000]
    st = chat_state(update.effective_chat.id)
    messages = [
        {"role": "system", "content": "Summarize clearly with key points and takeaways."},
        {"role": "user", "content": f"Summarize the following:\n\n{source}"},
    ]
    await generate_and_reply(update, ctx, messages, st["model"])


async def cmd_translate(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    if not ctx.args:
        return await update.effective_message.reply_text(
            "Usage: /translate <language> <text>  (e.g. /translate french Hello)"
        )
    if len(ctx.args) >= 2:
        target, text = ctx.args[0], " ".join(ctx.args[1:])
    else:
        target, text = "English", ctx.args[0]
    st = chat_state(update.effective_chat.id)
    messages = [
        {"role": "system", "content": "You are a professional translator. Output only the translation."},
        {"role": "user", "content": f"Translate the following into {target}:\n\n{text}"},
    ]
    await generate_and_reply(update, ctx, messages, st["model"])


async def cmd_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    if not SEARCH_AVAILABLE:
        return await update.effective_message.reply_text("Web search isn't available (ddgs not installed).")
    query = " ".join(ctx.args).strip()
    if not query:
        return await update.effective_message.reply_text("Usage: /search <query>")
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        results = await asyncio.to_thread(web_search, query, 6)
    except Exception as e:
        return await update.effective_message.reply_text(f"Search failed: {e}")
    if not results:
        return await update.effective_message.reply_text("No results found.")
    context_lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("href") or r.get("url", "")
        body = r.get("body") or r.get("snippet", "")
        context_lines.append(f"[{i}] {title}\n{url}\n{body}")
    ctx_text = "\n\n".join(context_lines)
    st = chat_state(update.effective_chat.id)
    messages = [
        {"role": "system", "content": "Answer using the search results. Cite sources as [n] and list the URLs at the end."},
        {"role": "user", "content": f"Question: {query}\n\nSearch results:\n{ctx_text}"},
    ]
    await generate_and_reply(update, ctx, messages, st["model"])


async def cmd_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    if not IMAGE_AVAILABLE:
        if GEMINI_API_KEY and not IMAGE_ENABLED:
            return await update.effective_message.reply_text(
                "Image generation is off: Gemini's free tier grants no image quota "
                "(the image model returns quota-exceeded). Add a billed Gemini key and "
                "set IMAGE_ENABLED=1 to turn it on."
            )
        return await update.effective_message.reply_text("Image generation needs a Gemini API key.")
    prompt = " ".join(ctx.args).strip()
    if not prompt:
        return await update.effective_message.reply_text("Usage: /image <description>")
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.UPLOAD_PHOTO)
    try:
        img = await gemini_image(prompt)
    except Exception as e:
        log.exception("image gen failed")
        return await update.effective_message.reply_text(f"Image generation failed: {e}")
    if not img:
        return await update.effective_message.reply_text("No image was returned for that prompt.")
    await update.effective_message.reply_photo(photo=BytesIO(img), caption=prompt[:1000])


async def cmd_deals(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    name = " ".join(ctx.args).strip()
    if not name:
        return await update.effective_message.reply_text("Usage: /deals <game name>  (e.g. /deals witcher 3)")
    await ctx.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        text = await find_deals(name)
    except Exception as e:
        log.exception("deals lookup failed")
        return await update.effective_message.reply_text(f"Deal lookup failed: {e}")
    for i in range(0, len(text), TG_MAX):
        await update.effective_message.reply_text(
            text[i : i + TG_MAX], link_preview_options=LinkPreviewOptions(is_disabled=True)
        )


async def cmd_watch(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    args = ctx.args
    if len(args) < 2 or not _is_price(args[-1]):
        return await update.effective_message.reply_text(
            f"Usage: /watch <game name> <target price>\n"
            f"e.g. /watch witcher 3 500   (price in {watch_currency()} region currency)"
        )
    target = float(args[-1])
    name = " ".join(args[:-1]).strip()
    chat_id = update.effective_chat.id
    mine = [w for w in WATCHES if w["chat_id"] == chat_id]
    if len(mine) >= MAX_WATCHES_PER_CHAT:
        return await update.effective_message.reply_text(
            f"You already have {MAX_WATCHES_PER_CHAT} watches. Remove one with /unwatch first."
        )
    await ctx.bot.send_chat_action(chat_id, ChatAction.TYPING)
    async with deals_client() as c:
        game, others = await resolve_game(c, name)
        if not game:
            return await update.effective_message.reply_text(not_found_text(name, others))
        current = await best_price(c, game)
    if not current:
        return await update.effective_message.reply_text(
            f"Found {game['title']}, but couldn't get a current price for it."
        )
    if current["price"] <= target:
        return await update.effective_message.reply_text(
            f"{current['title']} is already {fmt_price(current['price'], current['currency'])} "
            f"at {current['shop']} (at or below {fmt_price(target, current['currency'])}).\n{current['url']}\n\n"
            "Not adding a watch since it already meets your target."
        )
    WATCHES.append(
        {
            "chat_id": chat_id,
            "name": name,
            "title": game["title"],
            **{k: game[k] for k in GAME_ID_KEYS if game.get(k)},
            "target": target,
            "currency": current["currency"],
            "added": int(time.time()),
        }
    )
    save_watches()
    await update.effective_message.reply_text(
        f"Watching {current['title']}.\n"
        f"Now {fmt_price(current['price'], current['currency'])} at {current['shop']}; "
        f"I'll alert you when it drops to {fmt_price(target, current['currency'])} or below.\n"
        f"Checking every {WATCH_INTERVAL_HOURS:g}h. See your list with /watches."
    )


async def cmd_watches(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    chat_id = update.effective_chat.id
    mine = [w for w in WATCHES if w["chat_id"] == chat_id]
    if not mine:
        return await update.effective_message.reply_text("No active watches. Add one with /watch <game> <price>.")
    lines = ["Your price watches (remove with /unwatch <number>):"]
    for i, w in enumerate(mine, 1):
        lines.append(f"{i}. {w['title']} -> {fmt_price(w['target'], w.get('currency', STEAM_CC))}")
    await update.effective_message.reply_text("\n".join(lines))


async def cmd_unwatch(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    chat_id = update.effective_chat.id
    mine = [w for w in WATCHES if w["chat_id"] == chat_id]
    if not mine:
        return await update.effective_message.reply_text("No active watches to remove.")
    if not ctx.args or not ctx.args[0].isdigit():
        return await update.effective_message.reply_text("Usage: /unwatch <number>  (see /watches)")
    idx = int(ctx.args[0])
    if not 1 <= idx <= len(mine):
        return await update.effective_message.reply_text(f"Pick a number between 1 and {len(mine)} (see /watches).")
    target = mine[idx - 1]
    WATCHES.remove(target)
    save_watches()
    await update.effective_message.reply_text(f"Stopped watching {target['title']}.")


async def check_watches(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not WATCHES:
        return
    fired, backfilled = [], False
    async with deals_client() as c:
        for w in list(WATCHES):
            try:
                if not any(w.get(k) for k in GAME_ID_KEYS):
                    game, _ = await resolve_game(c, w["name"])
                    if not game:
                        continue
                    w.update({k: game[k] for k in GAME_ID_KEYS if game.get(k)}, title=game["title"])
                    backfilled = True
                current = await best_price(c, w)
            except Exception as e:
                log.warning("watch check failed for %r: %s", w["name"], e)
                continue
            if (
                current
                and current["currency"] == w.get("currency", current["currency"])
                and current["price"] <= w["target"]
            ):
                fired.append((w, current))
            await asyncio.sleep(1.5)  # be gentle on the APIs
    for w, current in fired:
        try:
            await ctx.bot.send_message(
                w["chat_id"],
                f"Price drop! {current['title']} is now "
                f"{fmt_price(current['price'], current['currency'])} at {current['shop']} "
                f"(target {fmt_price(w['target'], w.get('currency', current['currency']))}).\n{current['url']}",
                link_preview_options=LinkPreviewOptions(is_disabled=True),
            )
        except Exception as e:
            log.warning("could not notify chat %s: %s", w["chat_id"], e)
        if w in WATCHES:
            WATCHES.remove(w)
    if fired or backfilled:
        save_watches()


async def cmd_models(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    st = chat_state(update.effective_chat.id)
    lines = ["Available models (switch with /model <key>):"]
    for key, (_, _, label) in available_models().items():
        marker = " <- current" if key == st["model"] else ""
        lines.append(f"- {key}: {label}{marker}")
    await update.effective_message.reply_text("\n".join(lines))


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    if not ctx.args:
        return await cmd_models(update, ctx)
    st = chat_state(update.effective_chat.id)
    avail = available_models()
    key = ctx.args[0].strip()
    if key not in avail:
        return await update.effective_message.reply_text(f"Unknown/unavailable model '{key}'. Use /model to list.")
    st["model"] = key
    await update.effective_message.reply_text(f"Switched to {MODELS[key][2]}.")


async def cmd_local(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    st = chat_state(update.effective_chat.id)
    avail = available_models()
    key = DEFAULT_LOCAL if DEFAULT_LOCAL in avail else next(
        (k for k, v in avail.items() if v[0] == "ollama"), None
    )
    if not key:
        return await update.effective_message.reply_text("No local model available.")
    st["model"] = key
    await update.effective_message.reply_text(f"Switched to {MODELS[key][2]}.")


async def cmd_reset(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    chat_state(update.effective_chat.id)["history"] = []
    await update.effective_message.reply_text("Fresh conversation - memory cleared.")


async def cmd_tools(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    lines = [
        "Capabilities:",
        "- Chat / Q&A (/ask or just message me) - with memory",
        "- Coding help & debugging (/code)",
        "- Summarize text or links (/summarize)",
        "- Translate (/translate)",
        f"- Web search (/search) - {'on' if SEARCH_AVAILABLE else 'off'}",
        f"- Image generation (/image) - {'on' if IMAGE_AVAILABLE else 'off'}",
        f"- Game deals (/deals) - Steam + {'IsThereAnyDeal' if ITAD_API_KEY else 'CheapShark'}",
        f"- Price-drop alerts (/watch) - checks every {WATCH_INTERVAL_HOURS:g}h",
        f"- File downloads to Telegram (/dl) - {'on' if DL_AVAILABLE else 'off'}"
        + (f", up to {DL_MAX_FILE_MB} MB" if DL_AVAILABLE else "")
        + (" via local Bot API" if LOCAL_BOT_API else ""),
        "- Multiple models, local + hosted (/model)",
    ]
    await update.effective_message.reply_text("\n".join(lines))


async def cmd_status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    st = chat_state(update.effective_chat.id)
    # local Ollama reachability
    ollama_ok = "unknown"
    try:
        async with httpx.AsyncClient(timeout=6) as c:
            r = await c.get(OLLAMA_BASE_URL.replace("/v1", "") + "/api/tags")
            ollama_ok = f"up ({len(r.json().get('models', []))} models)" if r.status_code == 200 else f"http {r.status_code}"
    except Exception:
        ollama_ok = "unreachable"
    providers = []
    providers.append(f"Groq: {'set' if GROQ_API_KEY else '-'}")
    providers.append(f"Gemini: {'set' if GEMINI_API_KEY else '-'}")
    providers.append(f"GitHub Models: {'on' if provider_available('github') else 'off'}")
    await update.effective_message.reply_text(
        "Status:\n"
        f"- Current model: {MODELS[st['model']][2]}\n"
        f"- Local Ollama (ARM box): {ollama_ok}\n"
        f"- Providers: {', '.join(providers)}\n"
        f"- Models available: {len(available_models())}\n"
        f"- Search: {'on' if SEARCH_AVAILABLE else 'off'} | Image: {'on' if IMAGE_AVAILABLE else 'off'}"
    )


async def cmd_about(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    await update.effective_message.reply_text(
        "About this bot\n\n"
        "A personal AI assistant running on your own Oracle Cloud Always-Free servers.\n"
        "- Bot runs on an AMD micro instance\n"
        "- Private local models (Ollama) run on an ARM instance over the internal network\n"
        "- Also uses free hosted APIs (Groq, Gemini) for heavier tasks\n"
        "Your data stays on your infrastructure; no third-party bot platform."
    )


async def cmd_whoami(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    await update.effective_message.reply_text(
        f"Your Telegram user ID: {u.id}\nUsername: @{u.username}" if u else "?"
    )


# ---------------------------------------------------------------------------
# /dl - download big files via JDownloader, upload to Telegram
# ---------------------------------------------------------------------------
_JD_CLIENT: JDClient | None = None
# One active download per chat keeps progress messages and cancellation unambiguous.
DL_ACTIVE: dict[int, dict] = {}
# Chats that picked a host via the inline keyboard and owe us link(s) next: chat_id -> {host, expires}.
DL_PENDING_HOST: dict[int, dict] = {}
# Chats mid-crawl: a page was pasted, links found and grouped by host, awaiting a host pick.
# chat_id -> {"crawl_name": str, "groups": {host_label: [urls]}, "status_msg": Message}
DL_CRAWL: dict[int, dict] = {}


def jd_client() -> JDClient:
    global _JD_CLIENT
    if _JD_CLIENT is None:
        _JD_CLIENT = JDClient(MYJD_EMAIL, MYJD_PASSWORD, MYJD_DEVICE)
    return _JD_CLIENT


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _human_eta(seconds: int) -> str:
    if seconds is None or seconds < 0:
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _progress_bar(percent: float, width: int = 16) -> str:
    filled = int(round(percent / 100 * width))
    return "█" * filled + "░" * (width - filled)


def _dl_unconfigured_text() -> str:
    missing = []
    if not MYJD_AVAILABLE:
        missing.append("myjdapi not installed")
    if not (MYJD_EMAIL and MYJD_PASSWORD and MYJD_DEVICE):
        missing.append("MYJD_EMAIL/MYJD_PASSWORD/MYJD_DEVICE not set")
    return (
        "/dl isn't configured yet.\n"
        f"Missing: {', '.join(missing)}.\n\n"
        "One-time setup:\n"
        "1. `docker compose up -d` (starts JDownloader + the local Bot API server)\n"
        "2. Open http://<server>:5800 and sign in under Settings -> My.JDownloader\n"
        "3. Put that account + the device name in .env (MYJD_*), then restart the bot."
    )


def _host_picker_markup() -> InlineKeyboardMarkup:
    # callback_data = "dlhost:<label>"; labels are short so we stay well under 64 bytes.
    rows = [
        [InlineKeyboardButton(label, callback_data=f"dlhost:{label}")]
        for label in sorted(set(SUPPORTED_HOSTS.values()))
    ]
    rows.append([InlineKeyboardButton("Cancel", callback_data="dlhost:__cancel__")])
    return InlineKeyboardMarkup(rows)


async def cmd_dl(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    msg = update.effective_message

    if not DL_AVAILABLE:
        return await msg.reply_text(_dl_unconfigured_text())

    # Links can come from the command args or from a message you reply to with /dl.
    text = " ".join(ctx.args)
    if msg.reply_to_message:
        text += " " + (msg.reply_to_message.text or "")
    urls = extract_urls(text)

    if not urls:
        # No links given: offer the host picker. After a tap, we wait for the link(s).
        return await msg.reply_text(
            "Pick a host, then send me the link(s).\n"
            "Or paste link(s) with /dl <link> — direct host links download right away, and a game "
            "page (e.g. a FitGirl page) gets scanned so you can pick a host.",
            reply_markup=_host_picker_markup(),
        )

    chat_id = update.effective_chat.id
    pages = [u for u in urls if not detect_host(u)]
    if pages:
        # Anything that isn't a direct host link is treated as a page to crawl (JD finds the
        # real mirror links and we group them by host). Direct links on the page are included.
        return await _start_crawl(ctx, chat_id, urls, msg)

    await _begin_download(ctx, chat_id, msg, urls)


async def on_dl_pick(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if not allowed(update):
        return
    chat_id = update.effective_chat.id
    choice = (query.data or "").split(":", 1)[1] if ":" in (query.data or "") else ""

    if choice == "__cancel__":
        DL_PENDING_HOST.pop(chat_id, None)
        return await query.edit_message_text("Cancelled.")

    if choice not in set(SUPPORTED_HOSTS.values()):
        return await query.edit_message_text("Unknown host.")

    if chat_id in DL_ACTIVE:
        return await query.edit_message_text(
            "A download is already running here. Use /dlstatus or /dlcancel first."
        )

    DL_PENDING_HOST[chat_id] = {"host": choice, "expires": time.monotonic() + 300}
    await query.edit_message_text(
        f"{choice} selected. Now send the {choice} link(s) as your next message "
        f"(you have 5 minutes). Send /dlcancel to abort."
    )


async def _begin_download(ctx, chat_id: int, msg, urls: list[str], only_host: str | None = None) -> None:
    """Shared entry point for /dl, the host-picker flow, and replied links."""
    supported = [
        u for u in urls
        if detect_host(u) and (only_host is None or detect_host(u) == only_host)
    ]
    skipped = [u for u in urls if u not in supported]

    if not supported:
        hint = f" from {only_host}" if only_host else ""
        return await msg.reply_text(
            f"None of those links are supported{hint}.\n"
            "Supported: " + ", ".join(sorted(set(SUPPORTED_HOSTS.values())))
        )

    if chat_id in DL_ACTIVE:
        return await msg.reply_text("A download is already running here. Use /dlstatus or /dlcancel first.")

    if not LOCAL_BOT_API:
        await msg.reply_text(
            "Note: no local Bot API server is set (TELEGRAM_BOT_API_BASE), so uploads are "
            f"capped at {DL_MAX_FILE_MB} MB. Larger files will download but can't be sent."
        )

    hosts = ", ".join(sorted({detect_host(u) for u in supported}))
    lines = [f"Queuing {len(supported)} file(s) from {hosts}..."]
    if skipped:
        lines.append(f"(skipped {len(skipped)} link(s) not matching the selection/host)")
    status = await msg.reply_text("\n".join(lines))
    _launch_job(ctx, chat_id, status, supported)


def _launch_job(ctx, chat_id: int, status_msg, urls: list[str]) -> None:
    task = asyncio.create_task(_run_download_job(ctx, chat_id, urls, status_msg))
    DL_ACTIVE[chat_id] = {"task": task, "job": None, "status_msg": status_msg}


# ---- page crawl -> group by host -> pick ------------------------------------
async def _start_crawl(ctx, chat_id: int, urls: list[str], msg) -> None:
    if chat_id in DL_ACTIVE:
        return await msg.reply_text("A download is already running here. Use /dlstatus or /dlcancel first.")
    if chat_id in DL_CRAWL:
        return await msg.reply_text("Already scanning a page here. Finish that pick or /dlcancel first.")
    status = await msg.reply_text(f"Scanning {len(urls)} page(s) for FuckingFast/DataNodes/FileKeeper links...")
    DL_CRAWL[chat_id] = {"crawl_name": None, "groups": {}, "status_msg": status}
    asyncio.create_task(_run_crawl(ctx, chat_id, urls, status))


async def _run_crawl(ctx, chat_id: int, urls: list[str], status) -> None:
    client = jd_client()
    crawl_name = None
    try:
        crawl_name = await asyncio.to_thread(client.add_crawl, urls)
        if chat_id in DL_CRAWL:
            DL_CRAWL[chat_id]["crawl_name"] = crawl_name

        # Poll until the found-link count holds steady (crawl settled) or we time out.
        stable, last_count, found = 0, -1, []
        for _ in range(40):  # ~40 * 3s = 2 min ceiling
            await asyncio.sleep(3)
            if chat_id not in DL_CRAWL:  # cancelled
                return
            found = await asyncio.to_thread(client.crawl_links, crawl_name)
            supported = [l for l in found if detect_host(l.get("url", ""))]
            if len(found) == last_count and (supported or len(found) > 0):
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            last_count = len(found)

        supported = [l for l in found if detect_host(l.get("url", ""))]
        if not supported:
            await asyncio.to_thread(client.remove_crawl, crawl_name)
            DL_CRAWL.pop(chat_id, None)
            return await _safe_edit(
                status,
                "No FuckingFast/DataNodes/FileKeeper links found on that page.\n"
                "Try pasting the direct file link(s) instead with /dl <link>.",
            )

        # Group the found links by host.
        groups: dict[str, list[str]] = {}
        sizes: dict[str, int] = {}
        for l in supported:
            host = detect_host(l["url"])
            groups.setdefault(host, []).append(l["url"])
            sizes[host] = sizes.get(host, 0) + int(l.get("bytesTotal") or 0)
        DL_CRAWL[chat_id]["groups"] = groups

        lines = ["Found links — pick a host to download its whole set:"]
        buttons = []
        for host in sorted(groups):
            n = len(groups[host])
            sz = f", {_human_bytes(sizes[host])}" if sizes.get(host) else ""
            lines.append(f"- {host}: {n} file(s){sz}")
            buttons.append([InlineKeyboardButton(f"{host} ({n})", callback_data=f"dlset:{host}")])
        buttons.append([InlineKeyboardButton("Cancel", callback_data="dlset:__cancel__")])
        await _safe_edit(status, "\n".join(lines))
        try:
            await status.edit_reply_markup(reply_markup=InlineKeyboardMarkup(buttons))
        except Exception:
            pass

    except Exception as e:
        log.exception("crawl failed")
        if crawl_name:
            await asyncio.to_thread(client.remove_crawl, crawl_name)
        DL_CRAWL.pop(chat_id, None)
        await _safe_edit(status, f"Scan failed: {e}")


async def on_dl_set(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if not allowed(update):
        return
    chat_id = update.effective_chat.id
    choice = (query.data or "").split(":", 1)[1] if ":" in (query.data or "") else ""

    state = DL_CRAWL.pop(chat_id, None)
    client = jd_client()
    if state and state.get("crawl_name"):
        await asyncio.to_thread(client.remove_crawl, state["crawl_name"])

    if choice == "__cancel__":
        return await query.edit_message_text("Cancelled.")
    if not state:
        return await query.edit_message_text("That scan expired. Send the page again with /dl.")

    urls = state["groups"].get(choice)
    if not urls:
        return await query.edit_message_text("No links for that host anymore. Try /dl again.")
    if chat_id in DL_ACTIVE:
        return await query.edit_message_text("A download is already running here. Use /dlstatus or /dlcancel first.")

    await query.edit_message_text(f"Queuing {len(urls)} file(s) from {choice}...")
    _launch_job(ctx, chat_id, query.message, urls)


async def _run_download_job(ctx, chat_id: int, urls: list[str], status) -> None:
    client = jd_client()
    job: Job | None = None
    try:
        job = await asyncio.to_thread(client.add_job, urls)
        DL_ACTIVE[chat_id]["job"] = job

        last_edit = ""
        stalled_since = None
        while True:
            await asyncio.sleep(DL_POLL_SECONDS)
            if DL_ACTIVE.get(chat_id, {}).get("cancel"):
                await asyncio.to_thread(client.cancel, job)
                await _safe_edit(status, "Download cancelled.")
                return

            snap = await asyncio.to_thread(client.poll, job)
            if snap.error:
                await _safe_edit(status, f"JDownloader reported a problem:\n{snap.error}")
                return

            if not snap.present:
                # Still being parsed / online-checked (or waiting on Cloudflare).
                if stalled_since is None:
                    stalled_since = time.monotonic()
                elif time.monotonic() - stalled_since > 180:
                    await _safe_edit(status, "Still waiting for JDownloader to start (check its UI for a captcha).")
                    stalled_since = time.monotonic()
                continue
            stalled_since = None

            if snap.finished:
                break

            bar = _progress_bar(snap.percent)
            speed = f"{_human_bytes(snap.speed)}/s" if snap.speed else "--"
            text = (
                f"Downloading {len(urls)} file(s)\n"
                f"{bar} {snap.percent:.1f}%\n"
                f"{_human_bytes(snap.bytes_loaded)} / {_human_bytes(snap.bytes_total)} "
                f"@ {speed}  ETA {_human_eta(snap.eta)}"
            )
            if snap.status and "download" not in snap.status.lower():
                text += f"\n{snap.status}"
            if text != last_edit:
                await _safe_edit(status, text)
                last_edit = text

        # Finished: upload every file in the job folder, then clean up.
        files = await asyncio.to_thread(client.finished_files, job)
        if not files:
            await _safe_edit(status, "Download finished but no files were found on disk.")
            return

        await _safe_edit(status, f"Download complete. Uploading {len(files)} file(s) to Telegram...")
        limit_bytes = DL_MAX_FILE_MB * 1024 * 1024
        sent, failed = 0, []
        for fp in files:
            size = fp.stat().st_size
            if size > limit_bytes:
                failed.append(f"{fp.name} ({_human_bytes(size)} > {DL_MAX_FILE_MB} MB limit)")
                continue
            try:
                await ctx.bot.send_chat_action(chat_id, ChatAction.UPLOAD_DOCUMENT)
                await ctx.bot.send_document(
                    chat_id,
                    document=fp,
                    filename=fp.name,
                    caption=f"{fp.name} ({_human_bytes(size)})",
                    read_timeout=DL_SEND_TIMEOUT,
                    write_timeout=DL_SEND_TIMEOUT,
                    disable_content_type_detection=True,
                )
                sent += 1
            except Exception as e:
                log.exception("send_document failed for %s", fp.name)
                failed.append(f"{fp.name}: {e}")

        summary = [f"Uploaded {sent}/{len(files)} file(s)."]
        if failed:
            summary.append("Failed:")
            summary += [f"- {x}" for x in failed]
        await _safe_edit(status, "\n".join(summary))

    except asyncio.CancelledError:
        if job:
            await asyncio.to_thread(client.cancel, job)
        raise
    except Exception as e:
        log.exception("download job failed")
        await _safe_edit(status, f"Download job failed: {e}")
    finally:
        # Remove the finished files from disk to reclaim space on the micro box.
        if job is not None:
            try:
                if job.read_dir.exists():
                    shutil.rmtree(job.read_dir, ignore_errors=True)
            except Exception:
                pass
        DL_ACTIVE.pop(chat_id, None)


async def _safe_edit(message, text: str) -> None:
    try:
        await message.edit_text(text[:TG_MAX])
    except Exception:
        pass


async def cmd_dlstatus(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    entry = DL_ACTIVE.get(update.effective_chat.id)
    if not entry:
        return await update.effective_message.reply_text("No active download in this chat.")
    job: Job | None = entry.get("job")
    if job is None:
        return await update.effective_message.reply_text("Download is being queued...")
    snap = await asyncio.to_thread(jd_client().poll, job)
    if not snap.present:
        return await update.effective_message.reply_text("Waiting for JDownloader to pick it up...")
    await update.effective_message.reply_text(
        f"{_progress_bar(snap.percent)} {snap.percent:.1f}%\n"
        f"{_human_bytes(snap.bytes_loaded)} / {_human_bytes(snap.bytes_total)} "
        f"@ {_human_bytes(snap.speed)}/s  ETA {_human_eta(snap.eta)}"
        + (f"\n{snap.status}" if snap.status else "")
    )


async def cmd_dlcancel(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    chat_id = update.effective_chat.id
    had_pending = DL_PENDING_HOST.pop(chat_id, None) is not None
    crawl = DL_CRAWL.pop(chat_id, None)
    if crawl and crawl.get("crawl_name"):
        await asyncio.to_thread(jd_client().remove_crawl, crawl["crawl_name"])
        had_pending = True
    entry = DL_ACTIVE.get(chat_id)
    if not entry:
        msg = "Cleared the pending selection." if had_pending else "No active download to cancel."
        return await update.effective_message.reply_text(msg)
    entry["cancel"] = True
    await update.effective_message.reply_text("Cancelling the current download...")


# ---------------------------------------------------------------------------
# Plain chat
# ---------------------------------------------------------------------------
async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    chat_id = update.effective_chat.id
    user_text = update.effective_message.text

    # If this chat just picked a host via the inline keyboard, treat this message as the link(s).
    pending = DL_PENDING_HOST.get(chat_id)
    if pending:
        if pending["expires"] < time.monotonic():
            DL_PENDING_HOST.pop(chat_id, None)
            return await update.effective_message.reply_text(
                "Host selection timed out. Send /dl to start again."
            )
        urls = extract_urls(user_text)
        if not urls:
            return await update.effective_message.reply_text(
                f"No links found. Send the {pending['host']} link(s), or /dlcancel to stop."
            )
        DL_PENDING_HOST.pop(chat_id, None)
        return await _begin_download(ctx, chat_id, update.effective_message, urls, only_host=pending["host"])

    st = chat_state(chat_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += st["history"][-MAX_HISTORY_MESSAGES:]
    messages.append({"role": "user", "content": user_text})
    await generate_and_reply(update, ctx, messages, st["model"], remember=True, user_text=user_text)


# ---------------------------------------------------------------------------
# Helpers (blocking - run via asyncio.to_thread)
# ---------------------------------------------------------------------------
def fetch_url_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ai-bot/1.0)"}
    r = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
    r.raise_for_status()
    if BS4_AVAILABLE:
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text
    return r.text


def web_search(query: str, n: int = 6) -> list:
    from ddgs import DDGS
    with DDGS() as d:
        return list(d.text(query, max_results=n))


def fmt_price(amount: float, currency: str) -> str:
    symbol = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(currency)
    return f"{symbol}{amount:,.2f}" if symbol else f"{amount:,.2f} {currency}"


# ---------------------------------------------------------------------------
# Game deals. A query is resolved ONCE to a concrete game identity (Steam appid,
# ITAD id, CheapShark id); every price source is then queried by that id, so all
# results in a reply belong to the same game.
# ---------------------------------------------------------------------------
STEAM_API = "https://store.steampowered.com/api"
ITAD_API = "https://api.isthereanydeal.com"
CHEAPSHARK_API = "https://www.cheapshark.com/api/1.0"
MATCH_MIN = 0.6
SUGGEST_MIN = 0.4
# Add-ons that often rank alongside the base game in store searches.
_EXTRA_WORDS = {
    "soundtrack", "ost", "dlc", "points", "coins", "credits", "currency", "pack", "bundle",
    "season", "pass", "expansion", "artbook", "demo", "skin", "skins", "upgrade", "playtest",
}


_ROMAN = {"ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}


def _words(title: str) -> list[str]:
    return [_ROMAN.get(w, w) for w in re.findall(r"[a-z0-9]+", title.lower())]


def _norm_title(title: str) -> str:
    return "".join(_words(title))


def _word_in(word: str, title_words: list[str]) -> bool:
    """Exact word, or an acronym of consecutive title words ("gta" -> grand theft auto)."""
    if word in title_words:
        return True
    if len(word) < 2 or word.isdigit():
        return False
    initials = "".join(w[0] for w in title_words)
    return word in initials and any(
        "".join(w[0] for w in title_words[i : i + len(word)]) == word for i in range(len(title_words))
    )


def _match_score(query: str, title: str) -> float:
    q, t = _norm_title(query), _norm_title(title)
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0
    score = SequenceMatcher(None, q, t).ratio()
    qw, tw = _words(query), _words(title)
    if qw and all(_word_in(w, tw) for w in qw):
        score = max(score, 0.85)
    if any(w.isdigit() and w not in tw for w in qw):
        score -= 0.4
    extras = [w for w in tw if w in _EXTRA_WORDS and w not in qw]
    score -= 0.3 * bool(extras)
    score -= 0.01 * max(0, len(tw) - len(qw))
    return score


def _pick(query: str, items: list, title_of) -> tuple[object | None, list]:
    """Best-matching item (or None if nothing is close enough) plus other close matches."""
    ranked = sorted(
        ((_match_score(query, title_of(it)), -i, it) for i, it in enumerate(items)),
        key=lambda x: (x[0], x[1]),
        reverse=True,
    )
    good = [it for s, _, it in ranked if s >= MATCH_MIN]
    return (good[0] if good else None), good[1:]


async def resolve_game(c: httpx.AsyncClient, name: str) -> tuple[dict | None, list[str]]:
    """Resolve a free-text query to one game. Returns (game, other_titles).

    When nothing matches confidently, game is None and other_titles holds suggestions.
    """
    game, others, suggestions = None, [], []

    try:
        r = await c.get(f"{STEAM_API}/storesearch/", params={"term": name, "cc": STEAM_CC, "l": "en"})
        r.raise_for_status()
        items = [i for i in r.json().get("items", []) if i.get("type") == "app"]
        best, rest = _pick(name, items, lambda i: i["name"])
        if best and rest and _match_score(name, best["name"]) < 1.0:
            ranked = await _rank_steam(c, name, [best, *rest][:5])
            best, rest = ranked[0], ranked[1:]
        if best:
            game = {"title": best["name"], "steam_appid": int(best["id"])}
            others = [i["name"] for i in rest]
        suggestions += [i["name"] for i in items]
    except Exception as e:
        log.warning("Steam search failed for %r: %s", name, e)

    if ITAD_API_KEY:
        try:
            if game:
                r = await c.get(
                    f"{ITAD_API}/games/lookup/v1", params={"key": ITAD_API_KEY, "appid": game["steam_appid"]}
                )
                r.raise_for_status()
                data = r.json()
                if data.get("found"):
                    game.update(itad_id=data["game"]["id"], slug=data["game"].get("slug"))
            else:
                r = await c.get(
                    f"{ITAD_API}/games/search/v1", params={"key": ITAD_API_KEY, "title": name, "results": 10}
                )
                r.raise_for_status()
                items = [g for g in r.json() if g.get("type") == "game"]
                best, rest = _pick(name, items, lambda g: g["title"])
                if best:
                    game = {"title": best["title"], "itad_id": best["id"], "slug": best.get("slug")}
                    others = [g["title"] for g in rest]
                suggestions += [g["title"] for g in items]
        except Exception as e:
            log.warning("ITAD resolve failed for %r: %s", name, e)

    if not game:
        try:
            r = await c.get(f"{CHEAPSHARK_API}/games", params={"title": name, "limit": 10})
            r.raise_for_status()
            items = r.json()
            best, rest = _pick(name, items, lambda g: g.get("external", ""))
            if best:
                game = {"title": best["external"], "cheapshark_id": best["gameID"]}
                if best.get("steamAppID"):
                    game["steam_appid"] = int(best["steamAppID"])
                others = [g["external"] for g in rest]
            suggestions += [g.get("external", "") for g in items]
        except Exception as e:
            log.warning("CheapShark resolve failed for %r: %s", name, e)

    if game:
        return game, list(dict.fromkeys(others))[:3]
    close = [s for s in dict.fromkeys(suggestions) if s and _match_score(name, s) >= SUGGEST_MIN]
    return None, close[:5]


async def _steam_details(c: httpx.AsyncClient, appid: int) -> dict | None:
    try:
        r = await c.get(f"{STEAM_API}/appdetails", params={"appids": appid, "cc": STEAM_CC, "l": "en"})
        r.raise_for_status()
        entry = r.json().get(str(appid)) or {}
        return entry.get("data") if entry.get("success") else None
    except Exception:
        return None


async def _rank_steam(c: httpx.AsyncClient, name: str, items: list) -> list:
    """Re-rank close Steam matches: drop non-games (DLC, tools, soundtracks) and favour
    the far more reviewed base game over mods/editors that share its name."""
    details = await asyncio.gather(*(_steam_details(c, i["id"]) for i in items))
    ranked = []
    for item, d in zip(items, details):
        if d and d.get("type") != "game":
            continue
        reviews = ((d or {}).get("recommendations") or {}).get("total", 0)
        ranked.append((_match_score(name, item["name"]) + 0.05 * math.log10(1 + reviews), item))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [i for _, i in ranked] or items


async def steam_price(c: httpx.AsyncClient, appid: int) -> dict | None:
    """Steam price for one app in STEAM_CC. None if not sold there; currency None if free."""
    r = await c.get(
        f"{STEAM_API}/appdetails", params={"appids": appid, "cc": STEAM_CC, "filters": "price_overview,basic"}
    )
    r.raise_for_status()
    entry = r.json().get(str(appid)) or {}
    if not entry.get("success"):
        return None
    data = entry.get("data") or {}
    price = data.get("price_overview")
    return {
        "url": f"https://store.steampowered.com/app/{appid}/",
        "currency": price["currency"] if price else None,
        "initial": price["initial"] / 100 if price else 0.0,
        "final": price["final"] / 100 if price else 0.0,
    }


_CHEAPSHARK_STORES: dict[str, str] = {}


async def cheapshark_prices(c: httpx.AsyncClient, game: dict) -> dict | None:
    base = CHEAPSHARK_API
    gid = game.get("cheapshark_id")
    if not gid and game.get("steam_appid"):
        r = await c.get(f"{base}/games", params={"steamAppID": game["steam_appid"]})
        r.raise_for_status()
        hits = r.json()
        gid = hits[0]["gameID"] if hits else None
    if not gid:
        return None

    if not _CHEAPSHARK_STORES:
        sr = await c.get(f"{base}/stores")
        sr.raise_for_status()
        _CHEAPSHARK_STORES.update({s["storeID"]: s["storeName"] for s in sr.json()})

    r = await c.get(f"{base}/games", params={"id": gid})
    r.raise_for_status()
    data = r.json()
    deals = [
        {
            "shop": _CHEAPSHARK_STORES.get(d["storeID"], f"store {d['storeID']}"),
            "price": float(d["price"]),
            "regular": float(d["retailPrice"]),
            "cut": round(float(d["savings"])),
            "url": f"https://www.cheapshark.com/redirect?dealID={d['dealID']}",
        }
        for d in data.get("deals", [])
    ]
    deals.sort(key=lambda d: d["price"])
    low = data.get("cheapestPriceEver") or {}
    return {
        "currency": "USD",
        "deals": deals,
        "low": float(low["price"]) if low.get("price") else None,
        "low_date": time.strftime("%Y-%m-%d", time.gmtime(low["date"])) if low.get("date") else None,
    }


async def itad_prices(c: httpx.AsyncClient, itad_id: str) -> dict | None:
    r = await c.post(
        f"{ITAD_API}/games/prices/v3",
        params={"key": ITAD_API_KEY, "country": ITAD_COUNTRY, "vouchers": "true"},
        json=[itad_id],
    )
    r.raise_for_status()
    entries = r.json()
    if not entries:
        return None
    entry = entries[0]
    deals = []
    currency = None
    for d in entry.get("deals", []):
        price = d.get("price") or {}
        regular = d.get("regular") or {}
        currency = currency or price.get("currency")
        deals.append(
            {
                "shop": (d.get("shop") or {}).get("name", "?"),
                "price": float(price.get("amount", 0)),
                "regular": float(regular.get("amount", 0)),
                "cut": int(d.get("cut") or 0),
                "voucher": d.get("voucher"),
                "drm": ", ".join(x.get("name", "") for x in d.get("drm") or []),
                "url": d.get("url", ""),
            }
        )
    deals.sort(key=lambda d: d["price"])
    low = ((entry.get("historyLow") or {}).get("all")) or {}
    return {
        "currency": currency or low.get("currency") or "USD",
        "deals": deals,
        "low": float(low["amount"]) if low.get("amount") is not None else None,
        "low_date": None,
    }


def format_deal_block(header: str, res: dict) -> list[str]:
    cur = res["currency"]
    lines = [header]
    for i, d in enumerate(res["deals"][:DEALS_MAX], 1):
        line = f"{i}. {d['shop']}: {fmt_price(d['price'], cur)}"
        if d["cut"] > 0:
            line += f" (-{d['cut']}%, was {fmt_price(d['regular'], cur)})"
        if d.get("voucher"):
            line += f" [code: {d['voucher']}]"
        if d.get("drm"):
            line += f" [{d['drm']}]"
        lines += [line, f"   {d['url']}"]
    if res["low"] is not None:
        when = f" on {res['low_date']}" if res.get("low_date") else ""
        lines.append(f"All-time low: {fmt_price(res['low'], cur)}{when}")
    return lines


def _is_price(token: str) -> bool:
    try:
        return float(token) > 0
    except ValueError:
        return False


def deals_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=20, headers={"User-Agent": DEALS_UA}, follow_redirects=True)


def not_found_text(name: str, suggestions: list[str]) -> str:
    text = f"No game found for '{name}'. Try a more exact title."
    if suggestions:
        text += "\n\nDid you mean:\n" + "\n".join(f"- {s}" for s in suggestions)
    return text


async def find_deals(name: str) -> str:
    async with deals_client() as c:
        game, others = await resolve_game(c, name)
        if not game:
            return not_found_text(name, others)

        steam, market, notes = None, None, []
        if game.get("steam_appid"):
            try:
                steam = await steam_price(c, game["steam_appid"])
            except Exception as e:
                notes.append(f"Steam price failed: {e}")
        if game.get("itad_id"):
            try:
                market = ("IsThereAnyDeal", ITAD_COUNTRY, await itad_prices(c, game["itad_id"]))
            except Exception as e:
                notes.append(f"IsThereAnyDeal failed: {e}")
        if not market or not market[2] or not market[2]["deals"]:
            try:
                market = ("CheapShark", "US", await cheapshark_prices(c, game))
            except Exception as e:
                notes.append(f"CheapShark failed: {e}")

    lines: list[str] = [game["title"]]
    if game.get("steam_appid"):
        if not steam:
            lines.append(f"Steam ({STEAM_CC}): not sold in this region")
        elif not steam["currency"]:
            lines.append(f"Steam ({STEAM_CC}): free or no price listed")
        else:
            cur = steam["currency"]
            line = f"Steam ({STEAM_CC}): {fmt_price(steam['final'], cur)}"
            if steam["final"] < steam["initial"]:
                cut = round(100 * (1 - steam["final"] / steam["initial"]))
                line += f" (-{cut}%, was {fmt_price(steam['initial'], cur)})"
            lines.append(line)
        if steam:
            lines.append(f"   {steam['url']}")
    lines.append("")

    if market and market[2] and market[2]["deals"]:
        source, region, res = market
        lines += format_deal_block(f"Best deals across stores ({region} prices, via {source}):", res)
    else:
        lines.append("No other store deals found.")

    if game.get("slug"):
        itad_url = f"https://isthereanydeal.com/game/{game['slug']}/info/"
    else:
        itad_url = "https://isthereanydeal.com/search/?q=" + quote(game["title"])
    lines += ["", f"All stores & price history: {itad_url}"]

    if others:
        lines += ["", "Not this one? Try: " + " | ".join(f"/deals {t}" for t in others)]
    if notes:
        lines += [""] + notes
    return "\n".join(lines).strip()


async def best_price(c: httpx.AsyncClient, game: dict) -> dict | None:
    """Cheapest current offer for an already-resolved game, for /watch comparisons.

    Prefers ITAD (ITAD_COUNTRY currency, widest store coverage), otherwise Steam
    (STEAM_CC currency). Callers must compare only against the same currency.
    """
    if ITAD_API_KEY and game.get("itad_id"):
        try:
            res = await itad_prices(c, game["itad_id"])
            if res and res["deals"]:
                d = res["deals"][0]
                return {
                    "title": game["title"],
                    "price": d["price"],
                    "currency": res["currency"],
                    "shop": d["shop"],
                    "url": d["url"],
                }
        except Exception as e:
            log.warning("ITAD best_price failed for %r: %s", game["title"], e)
    if game.get("steam_appid"):
        try:
            steam = await steam_price(c, game["steam_appid"])
        except Exception as e:
            log.warning("Steam best_price failed for %r: %s", game["title"], e)
            steam = None
        if steam and steam["currency"] and steam["final"] > 0:
            return {
                "title": game["title"],
                "price": steam["final"],
                "currency": steam["currency"],
                "shop": "Steam",
                "url": steam["url"],
            }
    return None


GAME_ID_KEYS = ("steam_appid", "itad_id", "slug", "cheapshark_id")


def watch_currency() -> str:
    return ITAD_COUNTRY if ITAD_API_KEY else STEAM_CC


# ---------------------------------------------------------------------------
# Watchlist storage (persisted to disk so alerts survive restarts)
# ---------------------------------------------------------------------------
WATCHES: list[dict] = []


def load_watches() -> None:
    WATCHES.clear()
    try:
        with open(WATCHES_FILE, encoding="utf-8") as f:
            WATCHES.extend(json.load(f))
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("Could not read %s: %s", WATCHES_FILE, e)


def save_watches() -> None:
    try:
        directory = os.path.dirname(WATCHES_FILE)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(WATCHES_FILE, "w", encoding="utf-8") as f:
            json.dump(WATCHES, f, indent=2)
    except Exception as e:
        log.warning("Could not write %s: %s", WATCHES_FILE, e)


async def gemini_image(prompt: str) -> bytes | None:
    url = f"{GEMINI_NATIVE}/models/{IMAGE_MODEL}:generateContent"
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(url, headers=headers, json=body)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])
    return None


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
async def post_init(app: Application) -> None:
    await app.bot.set_my_commands(
        [
            BotCommand("start", "Start the bot and show a welcome"),
            BotCommand("help", "Show all commands and how to use them"),
            BotCommand("ask", "Ask the AI anything"),
            BotCommand("code", "Coding help, debugging, and code generation"),
            BotCommand("summarize", "Summarize text, a link, or a file"),
            BotCommand("translate", "Translate text to another language"),
            BotCommand("search", "Search the web for current info"),
            BotCommand("image", "Generate an image from a prompt"),
            BotCommand("deals", "Find the best prices for a game"),
            BotCommand("watch", "Alert me when a game drops below a price"),
            BotCommand("watches", "List my active price watches"),
            BotCommand("unwatch", "Remove a price watch"),
            BotCommand("dl", "Download a file (FuckingFast/DataNodes/FileKeeper) to Telegram"),
            BotCommand("dlstatus", "Show current download progress"),
            BotCommand("dlcancel", "Cancel the current download"),
            BotCommand("model", "Switch the AI model"),
            BotCommand("reset", "Start a fresh conversation (clear memory)"),
            BotCommand("tools", "List available tools and capabilities"),
            BotCommand("status", "Check agent and server status"),
            BotCommand("about", "About this bot"),
        ]
    )
    load_watches()
    if app.job_queue is not None:
        app.job_queue.run_repeating(check_watches, interval=WATCH_INTERVAL_HOURS * 3600, first=60)
        log.info("Price watch checks every %gh (%d loaded)", WATCH_INTERVAL_HOURS, len(WATCHES))
    else:
        log.warning("JobQueue unavailable (install python-telegram-bot[job-queue]); /watch alerts won't auto-run")
    log.info("Available models: %s", list(available_models()))
    log.info("Default model: %s", DEFAULT_MODEL)
    log.info("Search: %s | Image: %s", SEARCH_AVAILABLE, IMAGE_AVAILABLE)
    log.info(
        "/dl: %s (local Bot API: %s, max %d MB, device %s)",
        "on" if DL_AVAILABLE else "off",
        "on" if LOCAL_BOT_API else "off (public API)",
        DL_MAX_FILE_MB,
        MYJD_DEVICE or "-",
    )
    log.info("Allowlist: %s", ALLOWED_USER_IDS or "(open to everyone)")


def main() -> None:
    builder = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init)
    if LOCAL_BOT_API:
        # Route through the self-hosted Bot API server so /dl can send files up to ~2 GB.
        # local_mode lets the server read finished downloads by path (no re-upload from the bot).
        builder = (
            builder.base_url(f"{TELEGRAM_BOT_API_BASE}/bot")
            .base_file_url(f"{TELEGRAM_BOT_API_BASE}/file/bot")
            .local_mode(True)
            .read_timeout(DL_SEND_TIMEOUT)
            .write_timeout(DL_SEND_TIMEOUT)
            .connect_timeout(60)
            .pool_timeout(DL_SEND_TIMEOUT)
        )
    app = builder.build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("code", cmd_code))
    app.add_handler(CommandHandler("summarize", cmd_summarize))
    app.add_handler(CommandHandler("translate", cmd_translate))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("deals", cmd_deals))
    app.add_handler(CommandHandler("watch", cmd_watch))
    app.add_handler(CommandHandler("watches", cmd_watches))
    app.add_handler(CommandHandler("unwatch", cmd_unwatch))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("local", cmd_local))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("tools", cmd_tools))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("whoami", cmd_whoami))
    app.add_handler(CommandHandler("dl", cmd_dl))
    app.add_handler(CommandHandler("dlstatus", cmd_dlstatus))
    app.add_handler(CommandHandler("dlcancel", cmd_dlcancel))
    app.add_handler(CallbackQueryHandler(on_dl_pick, pattern=r"^dlhost:"))
    app.add_handler(CallbackQueryHandler(on_dl_set, pattern=r"^dlset:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    log.info("Starting polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
