"""
Telegram AI assistant — a multi-feature personal bot built for Oracle Cloud
Always-Free ARM/AMD micro instances.

Features
────────
• AI chat, coding help, summarization, translation  (Groq / Gemini / Ollama)
• Web search with AI synthesis                      (/search via DuckDuckGo)
• Image generation                                  (/image via Gemini)
• Game price finder + price-drop alerts             (/deals /watch)
• Large-file downloader → Telegram uploader         (/dl via JDownloader 2)
  Supports FuckingFast · DataNodes · FileKeeper, with page-crawl + file picker.

Stack: python-telegram-bot 21 · local Bot API server (2 GB uploads) · JDownloader 2
       via MyJDownloader relay · Docker Compose on OCI.
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
from functools import partial
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
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.error import BadRequest, EndPointNotFound, InvalidToken, RetryAfter, TimedOut
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
from tgformat import Chunk, escape_md, render, split_markdown, styled, telegram_markdown
import ops
import review

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
# httpx logs every request URL at INFO, and Bot API URLs contain the bot token.
logging.getLogger("httpx").setLevel(logging.WARNING)
# asyncssh logs a dozen lines per connection at INFO, and the ops checks connect every few minutes.
logging.getLogger("asyncssh").setLevel(logging.WARNING)
log = logging.getLogger("ai-bot")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"].strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://10.0.1.55:11434/v1").strip()
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

# /servers /autopilot /lantern + alerts (ops.py). Alerts go to this chat; default: the first allowed user.
OPS_ALERT_CHAT_ID = int(os.getenv("OPS_ALERT_CHAT_ID", "").strip() or 0) or (min(ALLOWED_USER_IDS) if ALLOWED_USER_IDS else 0)

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
_TG_LOCAL_CEIL = 2000
_TG_PUBLIC_CEIL = 49
_dl_env = int(os.getenv("DL_MAX_FILE_MB", str(_TG_LOCAL_CEIL if LOCAL_BOT_API else _TG_PUBLIC_CEIL)) or _TG_LOCAL_CEIL)
_dl_ceil = _TG_LOCAL_CEIL if LOCAL_BOT_API else _TG_PUBLIC_CEIL
DL_MAX_FILE_MB = min(_dl_env, _dl_ceil)
DL_AVAILABLE = MYJD_AVAILABLE and bool(MYJD_EMAIL and MYJD_PASSWORD and MYJD_DEVICE)
DL_POLL_SECONDS = 5
# Long transfers: sending a 2 GB file through the local server takes a while.
DL_SEND_TIMEOUT = float(os.getenv("DL_SEND_TIMEOUT", "3600") or 3600)

# styled() appends the Telegram formatting rules, which live in tgformat.FORMAT_RULES.
SYSTEM_PROMPT = styled(
    "You are a helpful assistant for academic and software development work. "
    "Be concise, accurate, and show code or steps when useful."
)
CODE_PROMPT = styled(
    "You are an expert programming assistant. Provide correct, idiomatic code with brief "
    "explanations. Use fenced code blocks. When debugging, identify the root cause first."
)
SUMMARY_PROMPT = styled("Summarize clearly with key points and takeaways.")
SEARCH_PROMPT = styled(
    "Answer using the search results. Cite sources inline as [n]. End with a **Sources** "
    "section that lists each source you cited on its own line as: n. [page title](url)"
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
_OAI_CLIENTS: dict[str, AsyncOpenAI] = {}


def _oai_client(provider: str) -> AsyncOpenAI:
    if provider not in _OAI_CLIENTS:
        conf = PROVIDERS[provider]
        _OAI_CLIENTS[provider] = AsyncOpenAI(
            base_url=conf["base_url"],
            api_key=conf["api_key"] or "none",
            default_headers=conf.get("headers"),
        )
    return _OAI_CLIENTS[provider]


async def stream_completion(model_key: str, messages: list):
    provider, model_id, _ = MODELS[model_key]
    client = _oai_client(provider)
    stream = await client.chat.completions.create(
        model=model_id, messages=messages, stream=True, temperature=0.4
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------------------------------------------------------------------------
# Sending formatted text
#
# Text is written as Markdown and goes out as a Telegram rich message, which shows real
# headings, lists, tables and formulas. Where that is not possible (an older Bot API
# server, or content Telegram rejects) the same Markdown is converted to a classic
# HTML message instead. tgformat.py holds both conversions.
# ---------------------------------------------------------------------------
NO_PREVIEW = LinkPreviewOptions(is_disabled=True)
THINKING = {"html": "<tg-thinking>Thinking…</tg-thinking>"}
# Rich messages need Bot API 10.1. Cleared once the Bot API server turns out not to have them.
_rich_available = True


async def _rich_call(bot, method: str, **params):
    """Call a rich-message method, which this version of python-telegram-bot does not know yet.

    Returns the API result. None means the text has to go out as a classic message instead:
    the Bot API server has no rich messages, or Telegram rejected this particular content.
    """
    global _rich_available
    if not _rich_available:
        return None
    try:
        return await bot.do_api_request(method, api_kwargs=params, read_timeout=30)
    except (EndPointNotFound, InvalidToken):
        _rich_available = False
        log.warning("This Bot API server has no %s; using classic messages from now on", method)
    except BadRequest as e:
        log.warning("Telegram rejected %s (%s); using a classic message instead", method, e)
    return None


async def _unhurried(send):
    """Run `send()`. When Telegram asks the bot to slow down, wait as long as it says and try again."""
    for _ in range(2):
        try:
            return await send()
        except RetryAfter as e:
            await asyncio.sleep(e.retry_after + 0.5)
    return await send()


async def _html_or_plain(method, chunk: Chunk, **kwargs):
    """Call a send or edit method with the formatted text or, if Telegram rejects that, the plain text."""
    for text, mode in ((chunk.html, ParseMode.HTML), (chunk.plain, None)):
        try:
            return await method(text, parse_mode=mode, link_preview_options=NO_PREVIEW, **kwargs)
        except BadRequest as e:
            if "not modified" in str(e).lower():
                return None
            if mode is None:
                raise
            log.warning("Telegram rejected a formatted message (%s); sending it as plain text", e)


async def _send_piece(bot, chat_id: int, piece: str, reply_markup=None) -> None:
    """Send Markdown that fits one rich message: as a rich message, or else as classic messages."""
    params = {"chat_id": chat_id, "rich_message": {"markdown": telegram_markdown(piece)}}
    if reply_markup:
        params["reply_markup"] = reply_markup
    if await _unhurried(lambda: _rich_call(bot, "sendRichMessage", **params)) is not None:
        return
    chunks = render(piece) or [Chunk(piece, piece)]
    for chunk in chunks:
        markup = reply_markup if chunk is chunks[-1] else None
        await _unhurried(lambda: _html_or_plain(partial(bot.send_message, chat_id), chunk, reply_markup=markup))


async def send_formatted(bot, chat_id: int, markdown: str, *, reply_markup=None) -> None:
    """Send Markdown to a chat, split into several messages if it is too long for one."""
    pieces = split_markdown(markdown)
    for piece in pieces:
        await _send_piece(bot, chat_id, piece, reply_markup if piece is pieces[-1] else None)


async def reply_formatted(update: Update, markdown: str, *, reply_markup=None) -> None:
    """Answer in the chat an update came from. Use escape_md() on values that are not Markdown."""
    await send_formatted(update.get_bot(), update.effective_chat.id, markdown, reply_markup=reply_markup)


async def _sync_reply(sent: list, chunks: list[Chunk]) -> None:
    """Make the chat show `chunks`: edit the messages whose text changed and send any new ones.

    `sent` holds [message, HTML it currently shows] pairs, one per Telegram message so far.
    """
    for i, chunk in enumerate(chunks):
        if i >= len(sent):
            sent.append([await _html_or_plain(sent[0][0].reply_text, chunk), chunk.html])
        elif sent[i][1] != chunk.html:
            await _html_or_plain(sent[i][0].edit_text, chunk)
            sent[i][1] = chunk.html


class _DraftPreview:
    """A reply shown while it is generated, in a private chat: a draft that grows as tokens arrive.

    A draft is only a preview that Telegram drops after 30 seconds or when the bot sends a
    message, so the finished reply is sent as a real message at the end.
    """

    interval = 1.0  # seconds between updates

    def __init__(self, bot, chat_id: int, draft_id: int):
        self.bot, self.chat_id, self.draft_id = bot, chat_id, draft_id
        self.delivered = 0  # leading pieces of the reply that were already sent as real messages
        self._thinking = None

    async def _draft(self, rich_message: dict) -> bool:
        result = await _rich_call(
            self.bot, "sendRichMessageDraft", chat_id=self.chat_id, draft_id=self.draft_id, rich_message=rich_message
        )
        return result is not None

    async def open(self) -> bool:
        """Show "Thinking…" until the first words arrive. False if this chat cannot show drafts."""
        if not await self._draft(THINKING):
            return False
        self._thinking = asyncio.create_task(self._keep_thinking())
        return True

    async def _keep_thinking(self) -> None:
        # A local model can take longer to start answering than a draft stays on screen.
        while True:
            await asyncio.sleep(20)
            try:
                await self._draft(THINKING)
            except Exception as e:
                log.debug("could not refresh the thinking draft: %s", e)

    def _stop_thinking(self) -> None:
        if self._thinking:
            self._thinking.cancel()
            self._thinking = None

    async def _deliver(self, pieces: list[str]) -> None:
        for piece in pieces[self.delivered:]:
            await _send_piece(self.bot, self.chat_id, piece)
            self.delivered += 1

    async def update(self, markdown: str) -> None:
        self._stop_thinking()
        pieces = split_markdown(markdown)
        # When a reply outgrows one message, the full part is sent for real and the rest stays a draft.
        await self._deliver(pieces[:-1])
        if len(pieces) > self.delivered:
            await self._draft({"markdown": telegram_markdown(pieces[-1])})

    async def finish(self, markdown: str) -> None:
        self._stop_thinking()
        await self._deliver(split_markdown(markdown))

    async def fail(self, text: str) -> None:
        self._stop_thinking()
        await self.bot.send_message(self.chat_id, text)


class _MessagePreview:
    """A reply shown while it is generated, where drafts are unavailable: a message that is edited."""

    interval = 1.2  # seconds between edits, which Telegram rate-limits

    def __init__(self, placeholder):
        self.sent = [[placeholder, None]]

    async def update(self, markdown: str) -> None:
        await _sync_reply(self.sent, render(markdown))

    async def finish(self, markdown: str) -> None:
        placeholder = self.sent[0][0]
        pieces = split_markdown(markdown)
        # A reply that is still one message can be upgraded in place to a rich message.
        if len(self.sent) == 1 and len(pieces) == 1:
            # Lower case, because under its usual name the library insists on its own edit method,
            # which cannot send rich messages. Telegram ignores case in method names.
            edit = partial(
                _rich_call, placeholder.get_bot(), "editmessagetext", chat_id=placeholder.chat_id,
                message_id=placeholder.message_id, rich_message={"markdown": telegram_markdown(pieces[0])},
            )
            if await _unhurried(edit) is not None:
                return
        chunks = render(markdown) or [Chunk(markdown, markdown)]
        for _ in range(3):
            try:
                return await _sync_reply(self.sent, chunks)
            except RetryAfter as e:
                await asyncio.sleep(e.retry_after + 0.5)
            except TimedOut:
                await asyncio.sleep(1)
        await _sync_reply(self.sent, chunks)

    async def fail(self, text: str) -> None:
        await self.sent[0][0].edit_text(text)


async def _open_preview(update: Update, bot):
    chat, message = update.effective_chat, update.effective_message
    if chat.type == ChatType.PRIVATE:  # Telegram shows drafts in private chats only
        preview = _DraftPreview(bot, chat.id, message.message_id)
        try:
            if await preview.open():
                return preview
        except Exception as e:
            log.warning("could not open a draft (%s); editing a message instead", e)
    return _MessagePreview(await message.reply_text("…"))


async def generate_and_reply(update, ctx, messages, model_key, *, remember=False, user_text=None):
    """Stream a completion into Telegram, formatted, showing the reply as it is written."""
    chat_id = update.effective_chat.id
    await ctx.bot.send_chat_action(chat_id, ChatAction.TYPING)
    preview = await _open_preview(update, ctx.bot)

    full = ""
    last_update = 0.0
    interval = preview.interval
    try:
        async for delta in stream_completion(model_key, messages):
            full += delta
            now = time.monotonic()
            if now - last_update >= interval and full.strip():
                last_update = now
                try:
                    await preview.update(full)
                except RetryAfter as e:
                    interval = max(interval, e.retry_after + 0.5)
                    await asyncio.sleep(e.retry_after)
                except Exception as e:
                    # A missed preview update does no harm: the finished reply is sent below.
                    log.debug("preview update failed: %s", e)
    except Exception as e:
        log.exception("completion failed")
        await preview.fail(f"Error from {MODELS[model_key][2]}:\n{e}")
        return None

    if not full.strip():
        full = "(empty response)"
    try:
        await preview.finish(full)
    except Exception:
        log.exception("could not deliver the reply")

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
    dl_line = (
        "- /dl — download files to Telegram (FuckingFast / DataNodes / FileKeeper)\n"
        if DL_AVAILABLE else ""
    )
    await reply_formatted(
        update,
        "## 👋 Hi! I'm your AI assistant\n\n"
        f"Model: **{escape_md(MODELS[st['model']][2])}**\n\n"
        "Send a message to chat, or use a command:\n\n"
        "- /ask — ask anything\n"
        "- /code — coding help and debugging\n"
        "- /summarize — summarize text or a link\n"
        "- /translate — translate text\n"
        "- /search — search the web\n"
        "- /image — generate an image\n"
        "- /deals — best prices for a game\n"
        "- /watch — alert me when a game gets cheap\n"
        f"{dl_line}"
        "- /model — switch AI model\n"
        "- /reset — clear memory\n"
        "- /help — full help",
    )


async def cmd_help(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    dl_help = ""
    if DL_AVAILABLE:
        hosts = " / ".join(sorted(set(SUPPORTED_HOSTS.values())))
        dl_help = (
            "\n### 📥 File downloads\n"
            f"From {escape_md(hosts)}.\n\n"
            "- /dl <link> [link2 …] — download and upload to this chat\n"
            "- /dl <page URL> — scan a page, pick a host, pick files\n"
            "- /dl — pick a host first, then send links\n"
            "- /dlstatus — check progress\n"
            "- /dlcancel — stop\n"
        )
    # <placeholders> are safe to write as they are: only real formatting tags are read as HTML.
    await reply_formatted(
        update,
        "## How to use me\n"
        "\n### 💬 Chat\n"
        "Send any message to chat; I keep the context.\n\n"
        "- /ask <question> — one-off question\n"
        "- /code <request> — coding help, debugging, generation\n"
        "- /summarize <text or URL> — or reply to a message\n"
        "- /translate <language> <text>\n"
        "- /search <query> — web search with an AI answer\n"
        "- /image <prompt> — generate an image\n"
        "\n### 🎮 Game deals\n"
        "- /deals <game> — Steam and store prices\n"
        "- /watch <game> <price> — alert when it drops\n"
        "- /watches — your watches\n"
        "- /unwatch <number> — remove a watch\n"
        f"{dl_help}"
        "\n### ⚙️ Settings\n"
        "- /model — list or switch AI models\n"
        "- /reset — clear conversation memory\n"
        "- /tools — show capabilities\n"
        "- /status — check server health\n"
        "- /about — about this bot\n"
        "\n### 🖥 Servers\n"
        "- /servers — health of every instance\n"
        "- /autopilot — status, or /autopilot pause|resume <stream>\n"
        "- /lantern — Lanternlight queue, retry parked parts\n"
        "- /ollama — local model server\n"
        "\n### 🎬 Reviews\n"
        "- /review — videos and reels waiting to go out, with feedback on each\n"
        "- /audit — problems the video template repeats across every video\n"
        "- /ytstats — views per video and what changed since the weekly report",
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
        {"role": "system", "content": SUMMARY_PROMPT},
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
        {"role": "system", "content": SEARCH_PROMPT},
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
    await reply_formatted(update, text)


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
            return await reply_formatted(update, not_found_text(name, others))
        current = await best_price(c, game)
    if not current:
        return await update.effective_message.reply_text(
            f"Found {game['title']}, but couldn't get a current price for it."
        )
    now_price = _md_price(current["price"], current["currency"])
    target_price = _md_price(target, current["currency"])
    shop = _md_link(current["shop"], current["url"])
    if current["price"] <= target:
        return await reply_formatted(
            update,
            f"**{escape_md(current['title'])}** is already **{now_price}** at {shop}, "
            f"at or below your target of {target_price}.\n\n"
            "Not adding a watch since it already meets your target.",
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
    await reply_formatted(
        update,
        f"🔔 Watching **{escape_md(current['title'])}**.\n\n"
        f"Now **{now_price}** at {shop}. "
        f"I'll alert you when it drops to **{target_price}** or below.\n\n"
        f"Checking every {WATCH_INTERVAL_HOURS:g}h. See your list with /watches.",
    )


async def cmd_watches(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    chat_id = update.effective_chat.id
    mine = [w for w in WATCHES if w["chat_id"] == chat_id]
    if not mine:
        return await update.effective_message.reply_text("No active watches. Add one with /watch <game> <price>.")
    lines = ["## 🔔 Your price watches", ""]
    for i, w in enumerate(mine, 1):
        lines.append(f"{i}. **{escape_md(w['title'])}** — at or below {_md_price(w['target'], w.get('currency', STEAM_CC))}")
    lines += ["", "Remove one with /unwatch <number>."]
    await reply_formatted(update, "\n".join(lines))


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
            await send_formatted(
                ctx.bot,
                w["chat_id"],
                f"📉 **Price drop!** {escape_md(current['title'])} is now "
                f"**{_md_price(current['price'], current['currency'])}** at "
                f"{_md_link(current['shop'], current['url'])} "
                f"(your target: {_md_price(w['target'], w.get('currency', current['currency']))}).",
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
    lines = ["## 🤖 Models", "", "Switch with /model <key>. Tap a key to copy it.", ""]
    for key, (_, _, label) in available_models().items():
        marker = " ✅ **current**" if key == st["model"] else ""
        lines.append(f"- `{key}` — {escape_md(label)}{marker}")
    await reply_formatted(update, "\n".join(lines))


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
    def on(flag) -> str:
        return "✅ on" if flag else "⚪ off"

    lines = [
        "## 🧰 Capabilities",
        "",
        "- **Chat and Q&A** (/ask or just message me) — with memory",
        "- **Coding help and debugging** (/code)",
        "- **Summarize** text or links (/summarize)",
        "- **Translate** (/translate)",
        f"- **Web search** (/search) — {on(SEARCH_AVAILABLE)}",
        f"- **Image generation** (/image) — {on(IMAGE_AVAILABLE)}",
        f"- **Game deals** (/deals) — Steam + {'IsThereAnyDeal' if ITAD_API_KEY else 'CheapShark'}",
        f"- **Price-drop alerts** (/watch) — checks every {WATCH_INTERVAL_HOURS:g}h",
        f"- **File downloads to Telegram** (/dl) — {on(DL_AVAILABLE)}"
        + (f", up to {DL_MAX_FILE_MB} MB" if DL_AVAILABLE else "")
        + (" via local Bot API" if LOCAL_BOT_API else ""),
        "- **Multiple models**, local and hosted (/model)",
        f"- **Server ops** (/servers /autopilot /lantern /ollama) — {on(ops.OPS_AVAILABLE)}"
        + (f", alerts every {ops.ALERT_MINUTES:g}m" if ops.OPS_AVAILABLE else ""),
        f"- **Video reviews** (/review /audit /ytstats) — {on(review.AVAILABLE)}"
        + (", new videos reviewed automatically" if review.AVAILABLE else ""),
    ]
    await reply_formatted(update, "\n".join(lines))


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
    providers.append(f"Groq {'✅' if GROQ_API_KEY else '⚪'}")
    providers.append(f"Gemini {'✅' if GEMINI_API_KEY else '⚪'}")
    providers.append(f"GitHub Models {'✅' if provider_available('github') else '⚪'}")
    await reply_formatted(
        update,
        "## 📊 Status\n\n"
        f"- **Current model:** {escape_md(MODELS[st['model']][2])}\n"
        f"- **Local Ollama (ARM box):** {'🟢' if ollama_ok.startswith('up') else '🔴'} {ollama_ok}\n"
        f"- **Providers:** {' · '.join(providers)}\n"
        f"- **Models available:** {len(available_models())}\n"
        f"- **Search:** {'✅ on' if SEARCH_AVAILABLE else '⚪ off'}\n"
        f"- **Image:** {'✅ on' if IMAGE_AVAILABLE else '⚪ off'}",
    )


async def cmd_about(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    await reply_formatted(
        update,
        "## ℹ️ About this bot\n\n"
        "A personal AI assistant running on your own Oracle Cloud Always-Free servers.\n\n"
        "- The bot runs on an AMD micro instance\n"
        "- Private local models (Ollama) run on an ARM instance over the internal network\n"
        "- Free hosted APIs (Groq, Gemini) handle heavier tasks\n\n"
        "Your data stays on your infrastructure; no third-party bot platform.",
    )


async def cmd_whoami(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    if not u:
        return await update.effective_message.reply_text("?")
    username = f"@{escape_md(u.username)}" if u.username else "not set"
    # The ID is shown as code, which Telegram copies on tap.
    await reply_formatted(update, f"Your Telegram user ID: `{u.id}`\n\nUsername: {username}")


# ---------------------------------------------------------------------------
# /servers /autopilot /lantern /ollama - what the instances are doing, plus alerts (ops.py)
# ---------------------------------------------------------------------------
def _ops_button(label: str, action: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(label, callback_data=f"ops:{action}")


async def _servers_view():
    text = ops.format_servers(await ops.all_health())
    return text, InlineKeyboardMarkup([[_ops_button("🔄 Refresh", "servers")]])


async def _lantern_view():
    s = await ops.call(ops.LANTERN_HOST, "lantern-status")
    rows = [[_ops_button("🔄 Refresh", "lantern")]]
    if any(n >= 2 for n in (s.get("fails") or {}).values()):
        rows.insert(0, [_ops_button("♻️ Retry parked parts", "lretry")])
    return ops.format_lantern(s), InlineKeyboardMarkup(rows)


async def _autopilot_view():
    s = await ops.call(ops.AUTOPILOT_HOST, "autopilot-status", f"{time.time() - 86400:.0f}")
    paused = {p["stream"] for p in s.get("paused") or []}
    rows = [[_ops_button(f"▶️ Resume {stream}", f"ap:resume:{stream}")] for stream in sorted(paused - {"all"})]
    rows.append([_ops_button("▶️ Resume all", "ap:resume:all") if "all" in paused else _ops_button("⏸ Pause all", "ap:pause:all")])
    rows.append([_ops_button("🔄 Refresh", "autopilot")])
    return ops.format_autopilot(s), InlineKeyboardMarkup(rows)


async def _ollama_view():
    used_by = [(key, model_id) for key, (provider, model_id, _) in MODELS.items() if provider == "ollama"]
    markup = InlineKeyboardMarkup([[_ops_button("🔄 Refresh", "ollama")]])
    try:
        return ops.format_ollama(await ops.ollama_status(OLLAMA_BASE_URL), used_by), markup
    except Exception as e:
        return f"Ollama at {OLLAMA_BASE_URL} is unreachable ({type(e).__name__}).", markup


async def _ops_show(update: Update, view, needs_ssh: bool = True) -> None:
    """Run an ops view and show it, editing the message in place when it came from a button."""
    query = update.callback_query
    if needs_ssh and not ops.OPS_AVAILABLE:
        text, markup = "Server ops are not set up: set OPS_HOSTS and mount the ops SSH key (see .env.example).", None
        if query:
            return await query.edit_message_text(text)
        return await update.effective_message.reply_text(text)
    message = query.message if query else await update.effective_message.reply_text("Checking…")
    try:
        text, markup = await view()
    except ops.OpsError as e:
        text, markup = f"Could not check: {e}", None
    try:
        await message.edit_text(text[:TG_MAX], reply_markup=markup)
    except BadRequest as e:
        if "not modified" not in str(e).lower():  # Refresh with nothing new
            raise


async def cmd_servers(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    await _ops_show(update, _servers_view)


async def cmd_lantern(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    await _ops_show(update, _lantern_view)


async def cmd_autopilot(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    args = [a.lower() for a in ctx.args]
    if not args:
        return await _ops_show(update, _autopilot_view)
    if len(args) != 2 or args[0] not in ("pause", "resume") or args[1] not in (*ops.STREAMS, "all"):
        return await update.effective_message.reply_text(
            f"Usage: /autopilot — status\n/autopilot pause|resume <{'|'.join((*ops.STREAMS, 'all'))}>"
        )
    await _ops_show(update, lambda: _autopilot_set(*args))


async def cmd_ollama(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    await _ops_show(update, _ollama_view, needs_ssh=False)


async def _autopilot_set(action: str, stream: str):
    r = await ops.call(ops.AUTOPILOT_HOST, f"autopilot-{action}", stream)
    text, markup = await _autopilot_view()
    return f"{'⏸' if action == 'pause' else '▶️'} {r['message']}\n\n{text}", markup


async def _lantern_retry():
    r = await ops.call(ops.LANTERN_HOST, "lantern-retry")
    text, markup = await _lantern_view()
    return f"{'♻️' if r['ok'] else '⏳'} {r['message']}\n\n{text}", markup


async def on_ops(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if not allowed(update):
        return
    action = (query.data or "").split(":")[1:]
    views = {"servers": _servers_view, "lantern": _lantern_view, "autopilot": _autopilot_view, "lretry": _lantern_retry}
    if action == ["ollama"]:
        await _ops_show(update, _ollama_view, needs_ssh=False)
    elif len(action) == 1 and action[0] in views:
        await _ops_show(update, views[action[0]])
    elif len(action) == 3 and action[0] == "ap" and action[1] in ("pause", "resume") and action[2] in (*ops.STREAMS, "all"):
        await _ops_show(update, lambda: _autopilot_set(action[1], action[2]))


async def ops_alerts(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    state = ops.load_state()
    first_run = "active" not in state
    try:
        lines = ops.evaluate(state, await ops.collect(state, OLLAMA_BASE_URL))
    except Exception:
        log.exception("ops alert check failed")
        return
    if lines and OPS_ALERT_CHAT_ID:
        if first_run:
            lines.insert(0, "Server alerts are on. Open issues right now:")
        text = "\n\n".join(lines)
        try:
            for i in range(0, len(text), TG_MAX):
                await ctx.bot.send_message(
                    OPS_ALERT_CHAT_ID, text[i : i + TG_MAX], link_preview_options=LinkPreviewOptions(is_disabled=True)
                )
        except Exception as e:
            # State is not saved, so the same alerts are sent again on the next check.
            log.warning("could not send ops alerts to %s: %s", OPS_ALERT_CHAT_ID, e)
            return
    ops.save_state(state)


# ---------------------------------------------------------------------------
# /review /audit /ytstats - feedback on what the servers publish (review.py)
# ---------------------------------------------------------------------------
REVIEW_OFF = "Reviews are not set up: they need server ops (OPS_HOSTS and the ops SSH key) and a Groq or Gemini key."


def _rv_button(label: str, action: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(label, callback_data=f"rv:{action}")


def _review_markup(rec: dict) -> InlineKeyboardMarkup:
    rows = []
    if rec["kind"] == "youtube":
        if rec.get("status") in ("queued", "held"):
            rows.append([_rv_button("⛔ Don't publish", f"b:{rec['id']}")])
        elif rec.get("status") == "blocked":
            rows.append([_rv_button("▶️ Put back in the queue", f"u:{rec['id']}")])
        rows.append([_rv_button("🔁 Review again", f"yr:{rec['id']}")])
    elif len(f"rv:lr:{rec['slug']}:{rec['part']}") <= 64:  # Telegram's limit for button data
        rows.append([_rv_button("🔁 Review again", f"lr:{rec['slug']}:{rec['part']}")])
    return InlineKeyboardMarkup(rows)


async def _send_long(bot, chat_id: int, text: str, markup=None) -> None:
    """Send plain text, split between paragraphs when it is over Telegram's limit. Buttons go on the last part."""
    parts, current = [], ""
    for para in text.split("\n\n"):
        while len(para) > TG_MAX:
            parts, current, para = parts + ([current] if current else []) + [para[:TG_MAX]], "", para[TG_MAX:]
        if current and len(current) + len(para) + 2 > TG_MAX:
            parts, current = parts + [current], para
        else:
            current = f"{current}\n\n{para}" if current else para
    parts.append(current)
    for i, part in enumerate(parts):
        await bot.send_message(
            chat_id, part, reply_markup=markup if i == len(parts) - 1 else None,
            link_preview_options=LinkPreviewOptions(is_disabled=True),
        )


async def _review_work(bot, chat_id: int, job, working: str) -> None:
    """Run a slow review job (a coroutine returning text and buttons) and send what it produced."""
    note = await bot.send_message(chat_id, working)
    try:
        text, markup = await job
    except (ops.OpsError, review.ReviewError) as e:
        text, markup = f"That didn't work: {e}", None
    except Exception:
        log.exception("review job failed")
        text, markup = "That failed unexpectedly; the details are in the bot log.", None
    try:
        await note.delete()
    except Exception:
        pass
    await _send_long(bot, chat_id, text, markup)


async def _review_youtube(item_id: str):
    rec = await review.review_youtube(item_id)
    return review.format_review(rec), _review_markup(rec)


async def _review_lantern(slug: str, part: int):
    waiting = (await review.targets())["lantern"]
    built = next((p["built"] for p in waiting if (p["slug"], p["part"]) == (slug, part)), None)
    rec = await review.review_lantern(slug, part, built)
    return review.format_review(rec), _review_markup(rec)


async def _review_audit():
    rec = await review.audit()
    return review.format_audit(rec), InlineKeyboardMarkup([[_rv_button("🔁 Run the audit again", "auditr")]])


async def _review_stats():
    text, _ = await review.stats_report()
    return text, None


async def _review_menu():
    waiting = await review.targets()
    lines, rows = [], []
    for v in waiting["youtube"]:
        state = {"queued": "publishes", "held": "held, planned for", "blocked": "⛔ blocked, was due"}[v["status"]]
        result = review.verdict(v["review"]) if v["review"] else "not reviewed yet"
        lines.append(f"🎬 {v['title']}\n    {state} {review._when(v['publish_at'])}; {result}")
        rows.append([_rv_button(f"🎬 {v['title'][:40]}", f"y:{v['id']}")])
    for p in waiting["lantern"]:
        result = review.verdict(p["review"]) if p["review"] else "not reviewed yet"
        lines.append(f"🏮 {p['title']}, part {p['part']} of {p['of']}\n    built, waiting to be posted; {result}")
        if len(f"rv:l:{p['slug']}:{p['part']}") <= 64:
            rows.append([_rv_button(f"🏮 {p['title'][:32]}, part {p['part']}", f"l:{p['slug']}:{p['part']}")])
    text = "Waiting to go out\n\n" + ("\n".join(lines) if lines else "Nothing is waiting: no video is queued and no reel is built ahead.")
    text += "\n\nTap one to see its review. /review <video id> reviews any video, published ones too."
    rows.append([_rv_button("📊 Template audit", "audit"), _rv_button("📈 Channel numbers", "stats")])
    return text, InlineKeyboardMarkup(rows)


def _start_review(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, job, working: str) -> None:
    ctx.application.create_task(_review_work(ctx.bot, chat_id, job, working))


async def _show_or_review(ctx, chat_id: int, key: str, job, working: str, again: bool = False) -> None:
    rec = None if again else review.stored(key)
    if rec:
        return await _send_long(ctx.bot, chat_id, review.format_review(rec), _review_markup(rec))
    if review.LOCK.locked():
        working += " Another review is running, so this one starts after it."
    _start_review(ctx, chat_id, job(), working)


async def cmd_review(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    chat_id = update.effective_chat.id
    if not review.AVAILABLE:
        return await update.effective_message.reply_text(REVIEW_OFF)
    args = [a.lower() for a in ctx.args]
    if not args:
        return await _ops_show(update, _review_menu)
    if len(args) == 1 and re.fullmatch(r"[0-9a-f]{8,32}", args[0]):
        return await _show_or_review(
            ctx, chat_id, f"yt:{args[0]}", lambda: _review_youtube(args[0]),
            "Reviewing the video and its Short: script, title, description, thumbnail and frames. About two minutes…",
        )
    if len(args) == 2 and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", args[0]) and args[1].isdigit():
        return await _show_or_review(
            ctx, chat_id, review.lantern_key(args[0], int(args[1])), lambda: _review_lantern(args[0], int(args[1])),
            "Reviewing the reel: story, captions and contact sheet. About a minute…",
        )
    await update.effective_message.reply_text(
        "Usage:\n/review — what is waiting to go out\n/review <video id> — a YouTube video and its Short\n"
        "/review <story-slug> <part> — a Lanternlight reel"
    )


async def cmd_audit(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    if not review.AVAILABLE:
        return await update.effective_message.reply_text(REVIEW_OFF)
    await _audit(ctx, update.effective_chat.id, again=bool(ctx.args) and ctx.args[0].lower() in ("again", "run", "new"))


async def _audit(ctx, chat_id: int, again: bool) -> None:
    rec = None if again else review.load().get("audit")
    if rec:
        markup = InlineKeyboardMarkup([[_rv_button("🔁 Run the audit again", "auditr")]])
        return await _send_long(ctx.bot, chat_id, review.format_audit(rec), markup)
    _start_review(
        ctx, chat_id, _review_audit(),
        "Auditing every built video for problems the template repeats: titles, descriptions, chapters, tags, "
        "thumbnails and frames. This takes a few minutes…",
    )


async def cmd_ytstats(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    if not review.AVAILABLE:
        return await update.effective_message.reply_text(REVIEW_OFF)
    _start_review(ctx, update.effective_chat.id, _review_stats(), "Reading the channel numbers…")


async def _review_block(action: str, item_id: str) -> str:
    r = await ops.call(ops.AUTOPILOT_HOST, f"autopilot-{action}", item_id)
    if not r.get("changed"):
        return f"Nothing changed: that video is {r.get('status', 'not in the queue')}."
    review.update_stored(f"yt:{item_id}", status=r["status"])
    names = " and ".join(f"the {c['kind']} video" if c["kind"] == "long" else "its Short" for c in r["changed"])
    title = r["changed"][0]["title"]
    if action == "block":
        return f"⛔ Blocked {names}: \"{title}\" will not be published. \"Put back in the queue\" undoes this."
    return f"▶️ Back in the queue: {names}, \"{title}\"."


async def on_review(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if not allowed(update) or not review.AVAILABLE:
        return
    chat_id = query.message.chat_id
    action = (query.data or "").split(":")[1:]
    head, rest = (action[0], action[1:]) if action else ("", [])
    is_id = len(rest) == 1 and re.fullmatch(r"[0-9a-f]{8,32}", rest[0])
    is_part = len(rest) == 2 and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", rest[0]) and rest[1].isdigit()
    if head == "menu" and not rest:
        await _ops_show(update, _review_menu)
    elif head in ("y", "yr") and is_id:
        await _show_or_review(
            ctx, chat_id, f"yt:{rest[0]}", lambda: _review_youtube(rest[0]),
            "Reviewing the video and its Short: script, title, description, thumbnail and frames. About two minutes…",
            again=head == "yr",
        )
    elif head in ("l", "lr") and is_part:
        await _show_or_review(
            ctx, chat_id, review.lantern_key(rest[0], int(rest[1])), lambda: _review_lantern(rest[0], int(rest[1])),
            "Reviewing the reel: story, captions and contact sheet. About a minute…", again=head == "lr",
        )
    elif head in ("b", "u") and is_id:
        try:
            text = await _review_block("block" if head == "b" else "unblock", rest[0])
        except ops.OpsError as e:
            text = f"That didn't work: {e}"
        rec = review.stored(f"yt:{rest[0]}")
        if rec:
            try:
                await query.edit_message_reply_markup(_review_markup(rec))
            except BadRequest:
                pass
        await ctx.bot.send_message(chat_id, text)
    elif head in ("audit", "auditr") and not rest:
        await _audit(ctx, chat_id, again=head == "auditr")
    elif head == "stats" and not rest:
        _start_review(ctx, chat_id, _review_stats(), "Reading the channel numbers…")


async def review_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Review whatever is newly waiting to go out (one per run) and send the weekly channel numbers when due."""
    if not OPS_ALERT_CHAT_ID or review.LOCK.locked():
        return
    try:
        await review.next_auto()
        for rec in review.unsent():
            text = "New and waiting to go out; reviewed automatically.\n\n" + review.format_review(rec)
            await _send_long(ctx.bot, OPS_ALERT_CHAT_ID, text, _review_markup(rec))
            review.update_stored(rec["key"], sent=True)
        if review.weekly_due():
            text, snapshot = await review.stats_report(weekly=True)
            await _send_long(ctx.bot, OPS_ALERT_CHAT_ID, text)
            review.record_weekly(snapshot)
    except Exception as e:
        # Unsent reviews stay marked unsent and the weekly report stays due, so both are retried next run.
        log.warning("review job failed: %s: %s", type(e).__name__, e)


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
# Chats in file-selection mode: crawl found files, user is picking which to download.
# chat_id -> {"files": [{"url","name","size"}], "host": str, "expires": float}
DL_FILE_PICK: dict[int, dict] = {}


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


def _short_name(name: str, max_len: int = 50) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def _parse_selection(text: str, total: int) -> list[int] | None:
    """Parse user input like 'all', '1-18', '1 3 5 7-10'.  Returns sorted 0-based indices."""
    text = text.strip().lower()
    if text == "all":
        return list(range(total))
    indices: set[int] = set()
    for token in re.split(r"[\s,;]+", text):
        if not token:
            continue
        m = re.match(r"^(\d+)\s*[-\u2013\u2014.]+\s*(\d+)$", token)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if not (1 <= a <= total and 1 <= b <= total):
                return None
            indices.update(range(min(a, b) - 1, max(a, b)))
            continue
        if token.isdigit():
            n = int(token)
            if not 1 <= n <= total:
                return None
            indices.add(n - 1)
            continue
        return None
    return sorted(indices) if indices else None


def _dl_unconfigured_text() -> str:
    missing = []
    if not MYJD_AVAILABLE:
        missing.append("myjdapi not installed")
    if not (MYJD_EMAIL and MYJD_PASSWORD and MYJD_DEVICE):
        missing.append("MYJD_EMAIL/MYJD_PASSWORD/MYJD_DEVICE not set")
    return (
        "**/dl isn't configured yet.**\n\n"
        f"Missing: {escape_md(', '.join(missing))}.\n\n"
        "### One-time setup\n"
        "1. Run `docker compose up -d`, which starts JDownloader and the local Bot API server\n"
        "2. Open `http://<server>:5800` and sign in under Settings → My.JDownloader\n"
        "3. Put that account and the device name in `.env` (the `MYJD_*` values), then restart the bot"
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
    log.info("/dl from chat %s, args=%r", update.effective_chat.id, ctx.args)
    msg = update.effective_message

    if not DL_AVAILABLE:
        return await reply_formatted(update, _dl_unconfigured_text())

    # Links can come from the command args or from a message you reply to with /dl.
    text = " ".join(ctx.args)
    if msg.reply_to_message:
        text += " " + (msg.reply_to_message.text or "")
    urls = extract_urls(text)

    if not urls:
        return await msg.reply_text(
            "📥 Three ways to use /dl:\n\n"
            "1️⃣  /dl <link1> <link2> …\n"
            "    Direct host links → downloads + uploads immediately.\n\n"
            "2️⃣  /dl <page URL>\n"
            "    Scans the page (e.g. FitGirl) for links, lets you\n"
            "    pick a host, then pick specific files.\n\n"
            "3️⃣  Tap a host below, then send the link(s):",
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

    # Reserve the slot BEFORE any await to prevent concurrent-update races.
    DL_ACTIVE[chat_id] = {"task": None, "job": None, "status_msg": None}

    if not LOCAL_BOT_API:
        await msg.reply_text(
            "⚠️ No local Bot API server configured — uploads capped at "
            f"{DL_MAX_FILE_MB} MB. Larger files will download but can't be sent."
        )

    hosts = ", ".join(sorted({detect_host(u) for u in supported}))
    lines = [f"⬇️ Queuing {len(supported)} file(s) from {hosts}…"]
    if skipped:
        lines.append(f"(skipped {len(skipped)} link(s) not matching the selection/host)")
    status = await msg.reply_text("\n".join(lines))
    DL_ACTIVE[chat_id]["status_msg"] = status
    _launch_job(ctx, chat_id, status, supported)


def _launch_job(ctx, chat_id: int, status_msg, urls: list[str]) -> None:
    log.info("_launch_job: chat=%s urls=%d DL_ACTIVE_keys_before=%s", chat_id, len(urls), list(DL_ACTIVE.keys()))
    task = asyncio.create_task(_run_download_job(ctx, chat_id, urls, status_msg))
    entry = DL_ACTIVE.get(chat_id) or {}
    entry.update(task=task, job=None, status_msg=status_msg)
    DL_ACTIVE[chat_id] = entry
    log.info("_launch_job: DL_ACTIVE[%s] set, task started", chat_id)


# ---- page crawl -> group by host -> pick ------------------------------------
async def _start_crawl(ctx, chat_id: int, urls: list[str], msg) -> None:
    log.info("_start_crawl: chat=%s urls=%d DL_ACTIVE=%s DL_CRAWL=%s",
             chat_id, len(urls), list(DL_ACTIVE.keys()), list(DL_CRAWL.keys()))
    if chat_id in DL_ACTIVE:
        return await msg.reply_text("A download is already running here. Use /dlstatus or /dlcancel first.")
    if chat_id in DL_CRAWL:
        return await msg.reply_text("Already scanning a page here. Finish that pick or /dlcancel first.")
    # Reserve the slot BEFORE any await to prevent concurrent-update race conditions.
    DL_CRAWL[chat_id] = {"crawl_name": None, "groups": {}, "status_msg": None}
    hosts = " / ".join(sorted(set(SUPPORTED_HOSTS.values())))
    status = await msg.reply_text(f"🔍 Scanning page for {hosts} links…")
    DL_CRAWL[chat_id]["status_msg"] = status
    asyncio.create_task(_run_crawl(ctx, chat_id, urls, status))


async def _run_crawl(ctx, chat_id: int, urls: list[str], status) -> None:
    client = jd_client()
    crawl_name = None
    try:
        crawl_name = await asyncio.to_thread(client.add_crawl, urls)
        if chat_id in DL_CRAWL:
            DL_CRAWL[chat_id]["crawl_name"] = crawl_name

        # Poll until the SUPPORTED link count stabilises.
        # Phase 1 (first 30s): just gather, never settle — JD needs time to
        #   fetch the page, resolve filecrypt containers, and check hosts.
        # Phase 2 (30s–3min): settle when the supported count holds for 3 polls.
        #   If 0 supported links after 90s of stable total count, give up.
        WARMUP = 10          # polls before allowing settlement (~30s)
        GIVEUP_ZERO = 30     # polls before giving up with 0 supported (~90s)
        PROGRESS_EVERY = 5   # update the user every N polls (~15s)
        stable, last_sup, last_total, found = 0, -1, -1, []
        last_progress_sup = -1
        for poll_n in range(60):  # ~60 * 3s = 3 min ceiling
            await asyncio.sleep(3)
            if chat_id not in DL_CRAWL:  # cancelled
                return
            try:
                found = await asyncio.to_thread(client.crawl_links, crawl_name)
            except Exception as e:
                log.warning("crawl poll %d failed: %s", poll_n, e)
                continue
            sup_count = sum(1 for l in found if detect_host(l.get("url", "")))
            log.info("crawl poll %d: total=%d supported=%d", poll_n, len(found), sup_count)

            # Show live progress so the user knows it's working.
            if poll_n > 0 and poll_n % PROGRESS_EVERY == 0 and sup_count != last_progress_sup:
                elapsed = (poll_n + 1) * 3
                await _safe_edit(
                    status,
                    f"🔍 Scanning… found {sup_count} supported link(s) so far ({elapsed}s)",
                )
                last_progress_sup = sup_count

            if poll_n < WARMUP:
                last_sup, last_total = sup_count, len(found)
                continue

            if sup_count > 0 and sup_count == last_sup:
                stable += 1
                if stable >= 3:
                    break
            elif sup_count == 0 and len(found) == last_total and poll_n >= GIVEUP_ZERO:
                break  # page genuinely has no supported links
            else:
                stable = 0
            last_sup = sup_count
            last_total = len(found)

        supported = [l for l in found if detect_host(l.get("url", ""))]
        if not supported:
            try:
                await asyncio.to_thread(client.remove_crawl, crawl_name)
            except Exception:
                pass
            DL_CRAWL.pop(chat_id, None)
            return await _safe_edit(
                status,
                "No FuckingFast/DataNodes/FileKeeper links found on that page.\n"
                "Try pasting the direct file link(s) instead with /dl <link>.",
            )

        # Group the found links by host (keep name + size for file picker).
        groups: dict[str, list[dict]] = {}
        for l in supported:
            host = detect_host(l["url"])
            groups.setdefault(host, []).append({
                "url": l["url"],
                "name": l.get("name") or l["url"].rsplit("/", 1)[-1],
                "size": int(l.get("bytesTotal") or 0),
            })
        DL_CRAWL[chat_id]["groups"] = groups

        lines = ["Found links — pick a host to see its files:"]
        buttons = []
        for host in sorted(groups):
            n = len(groups[host])
            total = sum(f["size"] for f in groups[host])
            sz = f", ~{_human_bytes(total)}" if total else ""
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
            try:
                await asyncio.to_thread(client.remove_crawl, crawl_name)
            except Exception:
                pass
        DL_CRAWL.pop(chat_id, None)
        # Sanitise error — never leak API URLs or credentials to the user.
        err_str = str(e)
        if "api.jdownloader" in err_str or "unreachable" in err_str.lower():
            user_err = "Could not reach JDownloader (network issue). Try again in a minute."
        elif "device" in err_str.lower() and "not found" in err_str.lower():
            user_err = "JDownloader device not found. Is JDownloader running?"
        else:
            # Strip anything that looks like a URL with query params.
            import re as _re
            user_err = _re.sub(r"https?://\S+", "<url>", err_str)
        await _safe_edit(status, f"Scan failed: {user_err}")


async def on_dl_set(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """User tapped a host button on the crawl result. Show the numbered file list for selection."""
    query = update.callback_query
    await query.answer()
    if not allowed(update):
        return
    chat_id = update.effective_chat.id
    choice = (query.data or "").split(":", 1)[1] if ":" in (query.data or "") else ""
    log.info("on_dl_set: chat=%s choice=%s DL_CRAWL_keys=%s", chat_id, choice, list(DL_CRAWL.keys()))

    state = DL_CRAWL.pop(chat_id, None)
    client = jd_client()
    if state and state.get("crawl_name"):
        await asyncio.to_thread(client.remove_crawl, state["crawl_name"])

    if choice == "__cancel__":
        return await query.edit_message_text("Cancelled.")
    if not state:
        log.info("on_dl_set: no state for chat %s — expired", chat_id)
        return await query.edit_message_text("That scan expired. Send the page again with /dl.")

    files = state["groups"].get(choice)
    if not files:
        return await query.edit_message_text("No links for that host anymore. Try /dl again.")
    if chat_id in DL_ACTIVE:
        return await query.edit_message_text("A download is already running here. Use /dlstatus or /dlcancel first.")

    # Sort files: .rar parts first (by name), then everything else alphabetically.
    files.sort(key=lambda f: (0 if f["name"].endswith(".rar") else 1, f["name"]))

    # Build numbered file list.
    max_name = 50
    lines = [f"{choice} — {len(files)} file(s). Pick which to download:\n"]
    for i, f in enumerate(files, 1):
        sz = f"  ({_human_bytes(f['size'])})" if f["size"] else ""
        lines.append(f" {i:>2}. {_short_name(f['name'], max_name)}{sz}")
    lines.append(f"\nReply with: all · 1-{len(files)} · 1 3 5 7-10")
    text = "\n".join(lines)

    if len(text) > TG_MAX:
        max_name = 35
        lines = [f"{choice} — {len(files)} file(s):\n"]
        for i, f in enumerate(files, 1):
            sz = f" ({_human_bytes(f['size'])})" if f["size"] else ""
            lines.append(f"{i:>2}. {_short_name(f['name'], max_name)}{sz}")
        lines.append(f"\nReply: all · 1-{len(files)} · 1 3 5 7-10")
        text = "\n".join(lines)

    buttons = InlineKeyboardMarkup([
        [InlineKeyboardButton("⬇️ All files", callback_data="dlpick:all"),
         InlineKeyboardButton("Cancel", callback_data="dlpick:cancel")],
    ])
    DL_FILE_PICK[chat_id] = {
        "files": files,
        "host": choice,
        "expires": time.monotonic() + 600,
    }
    await query.edit_message_text(text[:TG_MAX], reply_markup=buttons)


async def on_dl_filepick(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline button on the file-pick list (All / Cancel)."""
    query = update.callback_query
    await query.answer()
    if not allowed(update):
        return
    chat_id = update.effective_chat.id
    choice = (query.data or "").split(":", 1)[1] if ":" in (query.data or "") else ""
    log.info("on_dl_filepick: chat=%s choice=%s DL_FILE_PICK_keys=%s", chat_id, choice, list(DL_FILE_PICK.keys()))
    state = DL_FILE_PICK.pop(chat_id, None)

    if choice == "cancel":
        return await query.edit_message_text("Cancelled.")
    if not state:
        return await query.edit_message_text("Selection expired. Send the page again with /dl.")
    if chat_id in DL_ACTIVE:
        return await query.edit_message_text("A download is already running here. Use /dlstatus or /dlcancel first.")

    urls = [f["url"] for f in state["files"]]
    await query.edit_message_text(f"Queuing all {len(urls)} file(s) from {state['host']}...")
    _launch_job(ctx, chat_id, query.message, urls)


async def _run_download_job(ctx, chat_id: int, urls: list[str], status) -> None:
    log.info("_run_download_job START: chat=%s urls=%d", chat_id, len(urls))
    client = jd_client()
    job: Job | None = None
    try:
        job = await asyncio.to_thread(client.add_job, urls)
        log.info("_run_download_job: add_job OK, pkg=%s dir=%s", job.package_name, job.read_dir)
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
                log.warning("_run_download_job: JD error: %s", snap.error)
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
            speed = f"{_human_bytes(snap.speed)}/s" if snap.speed else "—"
            text = (
                f"⬇️ Downloading {len(urls)} file(s)\n"
                f"{bar}  {snap.percent:.1f}%\n"
                f"{_human_bytes(snap.bytes_loaded)} / {_human_bytes(snap.bytes_total)}"
                f"  ·  {speed}  ·  ETA {_human_eta(snap.eta)}"
            )
            if snap.status and "download" not in snap.status.lower():
                text += f"\n⏳ {snap.status}"
            if text != last_edit:
                await _safe_edit(status, text)
                last_edit = text

        # Finished: upload every file in the job folder, then clean up.
        files = await asyncio.to_thread(client.finished_files, job)
        if not files:
            await _safe_edit(status, "Download finished but no files were found on disk.")
            return

        await _safe_edit(status, f"Download complete. Uploading {len(files)} file(s) to Telegram…")
        limit_bytes = DL_MAX_FILE_MB * 1024 * 1024
        sent, skipped, failed = 0, [], []
        for idx, fp in enumerate(files, 1):
            size = fp.stat().st_size
            if size > limit_bytes:
                skipped.append(f"{fp.name} ({_human_bytes(size)} > {DL_MAX_FILE_MB} MB limit)")
                continue
            await _safe_edit(
                status,
                f"Uploading {idx}/{len(files)}: {fp.name}\n({_human_bytes(size)})"
            )
            try:
                await ctx.bot.send_chat_action(chat_id, ChatAction.UPLOAD_DOCUMENT)
                await ctx.bot.send_document(
                    chat_id,
                    document=fp,
                    filename=fp.name,
                    caption=f"{fp.name}  ({_human_bytes(size)})",
                    read_timeout=DL_SEND_TIMEOUT,
                    write_timeout=DL_SEND_TIMEOUT,
                    disable_content_type_detection=True,
                )
                sent += 1
            except Exception as e:
                log.exception("send_document failed for %s", fp.name)
                failed.append(f"{fp.name}: {e}")

        total = sent + len(skipped) + len(failed)
        summary = [f"✅ Done — uploaded {sent}/{total} file(s)."]
        if skipped:
            summary.append(f"\n⚠️ Skipped (too large):")
            summary += [f"  • {x}" for x in skipped]
        if failed:
            summary.append(f"\n❌ Failed:")
            summary += [f"  • {x}" for x in failed]
        await _safe_edit(status, "\n".join(summary))

    except asyncio.CancelledError:
        log.info("_run_download_job: cancelled for chat %s", chat_id)
        if job:
            await asyncio.to_thread(client.cancel, job)
        raise
    except Exception as e:
        log.exception("download job failed for chat %s", chat_id)
        err_str = str(e)
        if "api.jdownloader" in err_str or "unreachable" in err_str.lower():
            user_err = "Lost connection to JDownloader. Try again in a minute."
        else:
            import re as _re
            user_err = _re.sub(r"https?://\S+", "<url>", err_str)
        await _safe_edit(status, f"Download failed: {user_err}")
    finally:
        log.info("_run_download_job DONE: chat=%s (cleaning up DL_ACTIVE)", chat_id)
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
    except RetryAfter as e:
        await asyncio.sleep(e.retry_after)
        try:
            await message.edit_text(text[:TG_MAX])
        except Exception:
            pass
    except Exception as exc:
        log.debug("edit_text failed: %s", exc)


async def cmd_dlstatus(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return await deny(update)
    chat_id = update.effective_chat.id
    log.info("/dlstatus from %s — DL_ACTIVE keys: %s", chat_id, list(DL_ACTIVE.keys()))
    entry = DL_ACTIVE.get(chat_id)
    if not entry:
        return await update.effective_message.reply_text("No active download in this chat.")
    job: Job | None = entry.get("job")
    if job is None:
        log.info("/dlstatus: entry exists but job is None (still queuing)")
        return await update.effective_message.reply_text("Download is being queued...")
    snap = await asyncio.to_thread(jd_client().poll, job)
    if not snap.present:
        log.info("/dlstatus: job exists but snap not present yet")
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
    had_pending = DL_FILE_PICK.pop(chat_id, None) is not None or had_pending
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
    user_text = (update.effective_message.text or "").strip()
    if not user_text:
        return

    # If this chat is in file-pick mode (crawl result), parse the number selection.
    pick = DL_FILE_PICK.get(chat_id)
    if pick:
        log.info("on_message file-pick: chat=%s text=%r", chat_id, user_text[:60])
        if pick["expires"] < time.monotonic():
            DL_FILE_PICK.pop(chat_id, None)
            return await update.effective_message.reply_text(
                "File selection timed out. Send /dl to start again."
            )
        selected = _parse_selection(user_text, len(pick["files"]))
        if selected is None:
            return await update.effective_message.reply_text(
                f"Couldn't parse that. Send: all, 1-{len(pick['files'])}, 1 3 5 7-10, or /dlcancel"
            )
        DL_FILE_PICK.pop(chat_id, None)
        urls = [pick["files"][i]["url"] for i in selected]
        if not urls:
            return await update.effective_message.reply_text("No files selected. Send /dl to start again.")
        if chat_id in DL_ACTIVE:
            return await update.effective_message.reply_text(
                "A download is already running here. Use /dlstatus or /dlcancel first."
            )
        status = await update.effective_message.reply_text(
            f"Queuing {len(urls)} of {len(pick['files'])} file(s) from {pick['host']}..."
        )
        _launch_job(ctx, chat_id, status, urls)
        return

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
    """One Markdown list of store deals; each store name links to its page."""
    cur = res["currency"]
    lines = [header, ""]
    for i, d in enumerate(res["deals"][:DEALS_MAX], 1):
        line = f"{i}. {_md_link(d['shop'], d['url'])} — **{_md_price(d['price'], cur)}**"
        if d["cut"] > 0:
            line += f" (-{d['cut']}%, was {_md_price(d['regular'], cur)})"
        if d.get("voucher"):
            line += f" · code `{str(d['voucher']).replace('`', '')}`"
        if d.get("drm"):
            line += f" · {escape_md(d['drm'])}"
        lines.append(line)
    if res["low"] is not None:
        when = f" on {res['low_date']}" if res.get("low_date") else ""
        lines += ["", f"All-time low: **{_md_price(res['low'], cur)}**{when}"]
    return lines


def _md_price(amount: float, currency: str) -> str:
    return escape_md(fmt_price(amount, currency))


def _md_link(label: str, url: str) -> str:
    """A Markdown link, or just the label when there is no usable address."""
    label = escape_md(label).replace("[", "(").replace("]", ")")
    if not re.match(r"https?://\S+$", url or ""):
        return label
    return f"[{label}]({url.replace('(', '%28').replace(')', '%29')})"


def _is_price(token: str) -> bool:
    try:
        return float(token) > 0
    except ValueError:
        return False


def deals_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=20, headers={"User-Agent": DEALS_UA}, follow_redirects=True)


def not_found_text(name: str, suggestions: list[str]) -> str:
    """Markdown for a search that matched no game."""
    text = f"No game found for **{escape_md(name)}**. Try a more exact title."
    if suggestions:
        text += "\n\nDid you mean:\n\n" + "\n".join(f"- {escape_md(s)}" for s in suggestions)
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

    lines: list[str] = [f"## 🎮 {escape_md(game['title'])}", ""]
    if game.get("steam_appid"):
        label = _md_link(f"Steam ({STEAM_CC})", steam["url"]) if steam else f"Steam ({STEAM_CC})"
        if not steam:
            lines.append(f"{label}: not sold in this region")
        elif not steam["currency"]:
            lines.append(f"{label}: free or no price listed")
        else:
            cur = steam["currency"]
            line = f"{label}: **{_md_price(steam['final'], cur)}**"
            if steam["final"] < steam["initial"]:
                cut = round(100 * (1 - steam["final"] / steam["initial"]))
                line += f" (-{cut}%, was {_md_price(steam['initial'], cur)})"
            lines.append(line)
        lines.append("")

    if market and market[2] and market[2]["deals"]:
        source, region, res = market
        lines += format_deal_block(f"### Best deals across stores\n{region} prices, via {source}", res)
    else:
        lines.append("No other store deals found.")

    if game.get("slug"):
        itad_url = f"https://isthereanydeal.com/game/{game['slug']}/info/"
    else:
        itad_url = "https://isthereanydeal.com/search/?q=" + quote(game["title"])
    lines += ["", _md_link("All stores and price history", itad_url)]

    if others:
        # Shown as code, which Telegram copies on tap, ready to paste and send.
        lines += ["", "Not this one? Try:", ""] + [f"- `/deals {t.replace('`', '')}`" for t in others]
    if notes:
        lines += [""] + [escape_md(note) for note in notes]
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


# ---------------------------------------------------------------------------
# Global error handler — prevents silent crashes
# ---------------------------------------------------------------------------
async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Log unhandled exceptions so they never disappear silently."""
    log.error("Unhandled exception while processing an update:", exc_info=ctx.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Something went wrong. The error has been logged; try again shortly."
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Graceful shutdown — cancel active /dl jobs so JD doesn't keep downloading
# ---------------------------------------------------------------------------
async def post_shutdown(app: Application) -> None:
    if DL_ACTIVE:
        log.info("Shutting down: cancelling %d active download(s)…", len(DL_ACTIVE))
        client = jd_client() if DL_AVAILABLE else None
        for chat_id, entry in list(DL_ACTIVE.items()):
            job = entry.get("job")
            if job and client:
                try:
                    client.cancel(job)
                except Exception:
                    pass
            task = entry.get("task")
            if task and not task.done():
                task.cancel()
        DL_ACTIVE.clear()


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
            BotCommand("servers", "Health of every instance"),
            BotCommand("autopilot", "Autopilot status, pause or resume a stream"),
            BotCommand("lantern", "Lanternlight queue status and retry"),
            BotCommand("ollama", "Local model server status"),
            BotCommand("review", "Feedback on videos and reels waiting to go out"),
            BotCommand("audit", "Problems the video template repeats"),
            BotCommand("ytstats", "YouTube views per video"),
            BotCommand("about", "About this bot"),
        ]
    )
    load_watches()
    if app.job_queue is not None:
        app.job_queue.run_repeating(check_watches, interval=WATCH_INTERVAL_HOURS * 3600, first=60)
        log.info("Price watch checks every %gh (%d loaded)", WATCH_INTERVAL_HOURS, len(WATCHES))
        if ops.OPS_AVAILABLE:
            app.job_queue.run_repeating(ops_alerts, interval=ops.ALERT_MINUTES * 60, first=90)
        log.info(
            "Server ops: %s (hosts: %s, alerts every %gm to chat %s)",
            "on" if ops.OPS_AVAILABLE else "off", ", ".join(ops.OPS_HOSTS) or "-", ops.ALERT_MINUTES, OPS_ALERT_CHAT_ID or "-",
        )
        if review.AVAILABLE:
            app.job_queue.run_repeating(review_job, interval=review.CHECK_MINUTES * 60, first=150)
        log.info("Video reviews: %s (new items checked every %gm)", "on" if review.AVAILABLE else "off", review.CHECK_MINUTES)
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
    if _dl_env > _dl_ceil:
        log.warning(
            "DL_MAX_FILE_MB=%d exceeds the %s ceiling of %d MB; clamped to %d",
            _dl_env, "local Bot API" if LOCAL_BOT_API else "public API", _dl_ceil, DL_MAX_FILE_MB,
        )
    log.info("Allowlist: %s", ALLOWED_USER_IDS or "(open to everyone)")


def main() -> None:
    builder = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
    )
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
    app.add_handler(CallbackQueryHandler(on_dl_filepick, pattern=r"^dlpick:"))
    app.add_handler(CommandHandler("servers", cmd_servers))
    app.add_handler(CommandHandler("autopilot", cmd_autopilot))
    app.add_handler(CommandHandler("lantern", cmd_lantern))
    app.add_handler(CommandHandler("ollama", cmd_ollama))
    app.add_handler(CallbackQueryHandler(on_ops, pattern=r"^ops:"))
    app.add_handler(CommandHandler("review", cmd_review))
    app.add_handler(CommandHandler("audit", cmd_audit))
    app.add_handler(CommandHandler("ytstats", cmd_ytstats))
    app.add_handler(CallbackQueryHandler(on_review, pattern=r"^rv:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_error_handler(error_handler)
    log.info("Starting polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
