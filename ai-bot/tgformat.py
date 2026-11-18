"""
Telegram formatting for model replies.

Models write Markdown, and Telegram shows it in one of two ways:

* Rich messages (Bot API 10.1+, sendRichMessage). Telegram renders Markdown itself,
  with real headings, lists, tables and formulas. telegram_markdown() first rewrites
  the text so Telegram's parser reads it the way the model meant it, and
  split_markdown() cuts a very long reply to fit Telegram's limits.
* Classic messages (sendMessage with parse_mode=HTML), used where rich messages are
  not available or Telegram rejects one. render() converts the Markdown to the few
  HTML tags classic messages allow and imitates the rest (tables, headings, lists).

FORMAT_RULES is appended to the system prompts. It tells the model how to lay a reply
out for a phone chat. Edit it to change how replies look.
"""
import html
import re
from itertools import groupby
from typing import NamedTuple

FORMAT_RULES = """\
Formatting rules. Your reply is shown in a Telegram chat, usually on a phone:
- Start with the answer itself, with no preamble.
- Keep paragraphs to 1-3 sentences.
- Write GitHub-style Markdown: **bold**, *italic*, `inline code`, fenced code blocks with a language tag, lists, "> " quotes and [text](url) links.
- In a long reply, open each section with a "## " heading, optionally led by one fitting emoji. A short reply gets no headings.
- Use a table only to compare several items across several properties, with at most 4 columns and short cells. Otherwise use bullets that each open with a **bold label**.
- Write formulas in LaTeX, inline as \\( ... \\) and on their own line as $$ ... $$. Write money as a plain amount such as $5.
- Never use HTML tags or images.
- Bold only key terms and the final result. Put commands, file names and values to copy in `code`.
- Nest lists at most one level deep.
- Keep code lines under 60 characters where the language allows."""


def styled(prompt: str) -> str:
    """A system prompt with the Telegram formatting rules appended."""
    return f"{prompt}\n\n{FORMAT_RULES}"


def escape_md(value) -> str:
    """Make a value (a title, a name, an error message) show literally inside Markdown the bot writes."""
    text = str(value).replace("&", "&amp;").replace("<", "&lt;")
    return re.sub(r"[\\`*_~|$#>]", r"\\\g<0>", text)


class Chunk(NamedTuple):
    """One Telegram message: `html` to send with parse_mode=HTML, `plain` as a fallback."""
    html: str
    plain: str


# Telegram allows 4096 characters per message, counted after formatting is removed.
MESSAGE_LIMIT = 4000
# Widest table that still fits a phone screen in monospace without wrapping.
MONO_WIDTH = 32

_BULLETS = ("•", "◦", "▪")
_RULE = "─" * 10

# Private-use characters stand in for text that later passes must not touch.
_SLOT_OPEN, _SLOT_CLOSE = "\ue000", "\ue001"
_ESCAPABLE = "\\`*_{}[]()#+-.!~|>$="
_ESCAPED_BASE = 0xE100
_RESERVED = re.compile(f"[{_SLOT_OPEN}{_SLOT_CLOSE}\ue100-\ue11f]")

_SLOT = re.compile(f"{_SLOT_OPEN}(\\d+){_SLOT_CLOSE}")
_ESCAPED_CHAR = re.compile("[\ue100-\ue11f]")
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
_MATH = re.compile(r"\\\((.+?)\\\)|\\\[(.+?)\\\]|\$\$(.+?)\$\$")
_BACKSLASH = re.compile(r"\\([\\`*_{}\[\]()#+\-.!~|>$=])")
_URL = r"(?:https?://|tg://|mailto:)(?:[^\s()<>]|\([^\s()<>]*\))+"
_LINK = re.compile(r"!?\[([^\[\]\n]+)\]\(\s*<?(" + _URL + r")>?(?:\s+\"[^\"]*\")?\s*\)")
_AUTOLINK = re.compile(r"<((?:https?://|mailto:)[^\s<>]+)>")
_BARE_URL = re.compile(r"https?://[^\s<>*]+")
_TAG = re.compile(r"</?(?:b|i|s|code|pre|a|blockquote)\b[^>]*>")

_DUNDER = re.compile(r"(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)")
# (pattern, HTML opening tag, HTML closing tag, Markdown marker)
_EMPHASIS = (
    (re.compile(r"\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*"), "<b><i>", "</i></b>", "***"),
    (re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*"), "<b>", "</b>", "**"),
    (_DUNDER, "<b>", "</b>", "**"),
    (re.compile(r"~~(?=\S)(.+?)(?<=\S)~~"), "<s>", "</s>", "~~"),
    (re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])"), "<i>", "</i>", "*"),
    (re.compile(r"(?<!\w)_(?=[^\s_])(.+?)(?<=[^\s_])_(?!\w)"), "<i>", "</i>", "*"),
)

_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*([^\s`]*)[^`]*$")
_MATH_FENCES = {"\\[": "\\]", "$$": "$$"}
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)(?:\s+#+)?$")
# A short line that is bold from end to end is a section title.
_TITLE = re.compile(r"^\s{0,3}\*\*[^*]{1,60}\*\*:?$")
_HRULE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_QUOTE = re.compile(r"^\s{0,3}>")
_QUOTE_MARKS = re.compile(r"^\s*(?:>\s?)+")
_BULLET = re.compile(r"^(\s*)[-*+•]\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")
_TASK = re.compile(r"^\[([ xX])\]\s+(.*)$")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


class _Line(NamedTuple):
    html: str
    kind: str = "text"  # "text", "blank", "heading" or "item" (a list item)
    wrap: tuple | None = None  # (block id, opening tag, closing tag) for <pre> and <blockquote>


_BLANK = _Line("", "blank")


def render(markdown: str, limit: int = MESSAGE_LIMIT) -> list[Chunk]:
    """Convert a model reply to Telegram HTML, split into messages of at most `limit` characters."""
    lines = _tidy(_parse(_RESERVED.sub("", markdown.replace("\r\n", "\n"))))
    return [_assemble(group) for group in _pack(lines, limit)]


# ---------------------------------------------------------------------------
# Inline formatting: one line of Markdown -> HTML
# ---------------------------------------------------------------------------
def _plain(text: str) -> str:
    return html.unescape(_TAG.sub("", text))


def _length(text: str) -> int:
    """Length as Telegram counts it (UTF-16 code units)."""
    return len(text.encode("utf-16-le")) // 2


def _restore_escaped(text: str, as_html: bool = True) -> str:
    def char(m: re.Match) -> str:
        c = _ESCAPABLE[ord(m.group()) - _ESCAPED_BASE]
        return html.escape(c, quote=False) if as_html else c

    return _ESCAPED_CHAR.sub(char, text)


def _spans(text: str, wrap, leaf) -> str:
    """Find bold, italic and strikethrough spans, leftmost first.

    `wrap(style, inner)` renders one span and `leaf(text)` the text between spans.
    Spans are rendered recursively, so the output always nests properly.
    """
    out = []
    while text:
        first = None
        for style in _EMPHASIS:
            for m in style[0].finditer(text):
                # __init__ and friends are Python names, not bold text.
                if style[0] is _DUNDER and re.fullmatch(r"\w+", m.group(1)):
                    continue
                if first is None or m.start() < first[0].start():
                    first = (m, style)
                break
        if first is None:
            break
        m, style = first
        out.append(_spans(text[: m.start()], wrap, leaf) + wrap(style, _spans(m.group(1), wrap, leaf)))
        text = text[m.end():]
    out.append(leaf(text))
    return "".join(out)


def _emphasis(text: str) -> str:
    """Bold, italic and strikethrough as HTML tags. Entities such as &lt; show as the character they name."""
    return _spans(
        text,
        lambda style, inner: style[1] + inner + style[2],
        lambda t: html.escape(html.unescape(t), quote=False),
    )


def _inline(text: str) -> str:
    slots: list[str] = []

    def hold(rendered: str) -> str:
        slots.append(rendered)
        return f"{_SLOT_OPEN}{len(slots) - 1}{_SLOT_CLOSE}"

    def code(m: re.Match) -> str:
        body = m.group(2)
        if body.startswith(" ") and body.endswith(" ") and body.strip():
            body = body[1:-1]
        return hold(f"<code>{html.escape(body, quote=False)}</code>")

    def math(m: re.Match) -> str:
        body = (m.group(1) or m.group(2) or m.group(3)).strip()
        if _SLOT_OPEN in body:  # already holds a code span, and code cannot nest
            return m.group()
        return hold(f"<code>{html.escape(body, quote=False)}</code>")

    def link(m: re.Match) -> str:
        url = html.escape(_restore_escaped(m.group(2), as_html=False), quote=True)
        return hold(f'<a href="{url}">{_emphasis(m.group(1))}</a>')

    def autolink(m: re.Match) -> str:
        return hold(html.escape(_restore_escaped(m.group(1), as_html=False), quote=False))

    def bare_url(m: re.Match) -> str:
        return hold(html.escape(_restore_escaped(m.group(), as_html=False), quote=False))

    text = _CODE_SPAN.sub(code, text)
    text = _MATH.sub(math, text)
    text = _BACKSLASH.sub(lambda m: chr(_ESCAPED_BASE + _ESCAPABLE.index(m.group(1))), text)
    text = _LINK.sub(link, text)
    text = _AUTOLINK.sub(autolink, text)
    text = _BARE_URL.sub(bare_url, text)
    text = _emphasis(text)
    while _SLOT.search(text):
        text = _SLOT.sub(lambda m: slots[int(m.group(1))], text)
    return _restore_escaped(text)


# ---------------------------------------------------------------------------
# Blocks: Markdown text -> list of rendered lines
# ---------------------------------------------------------------------------
def _indent(whitespace: str) -> int:
    return len(whitespace.expandtabs(4))


def _list_depth(stack: list[int], indent: int) -> int:
    while stack and stack[-1] > indent:
        stack.pop()
    if not stack or stack[-1] < indent:
        stack.append(indent)
    return min(len(stack) - 1, 3)


def _unbold(text: str) -> str:
    """Drop bold markers around text that is about to be made bold anyway."""
    return re.sub(r"^\*\*([^*]+)\*\*$", r"\1", text.strip())


def _text_line(raw: str, stack: list[int]) -> _Line:
    """A heading, rule, list item or paragraph line. `stack` tracks open list levels."""
    if not raw.strip():
        return _BLANK
    if _HRULE.match(raw):
        stack.clear()
        return _Line(_RULE)
    m = _HEADING.match(raw)
    if m:
        stack.clear()
        return _Line(f"<b>{_inline(_unbold(m.group(1)))}</b>", "heading")
    if _TITLE.match(raw):
        stack.clear()
        return _Line(_inline(raw.strip()), "heading")
    m = _BULLET.match(raw)
    if m:
        depth = _list_depth(stack, _indent(m.group(1)))
        body, glyph = m.group(2), _BULLETS[min(depth, len(_BULLETS) - 1)]
        task = _TASK.match(body)
        if task:
            body, glyph = task.group(2), "☐" if task.group(1) == " " else "☑"
        return _Line(f"{'   ' * depth}{glyph} {_inline(body)}", "item")
    m = _NUMBERED.match(raw)
    if m:
        depth = _list_depth(stack, _indent(m.group(1)))
        return _Line(f"{'   ' * depth}{m.group(2)}. {_inline(m.group(3))}", "item")
    indent = _indent(raw[: len(raw) - len(raw.lstrip())])
    if stack and indent >= 2:
        # Continuation of a list item: line it up under that item's text.
        depth = min(max(sum(1 for level in stack if level < indent) - 1, 0), 3)
        return _Line(f"{'   ' * depth}  {_inline(raw.strip())}")
    stack.clear()
    return _Line(_inline(raw.strip()))


def _cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    return [cell.strip().replace("\\|", "|") for cell in _CELL_SPLIT.split(row)]


def _table(header: list[str], rule: list[str], rows: list[list[str]], block: int) -> list[_Line]:
    """Telegram has no tables: small ones become aligned monospace, larger ones become lists."""
    columns = len(header)
    rows = [(row + [""] * columns)[:columns] for row in rows]
    grid = [[_plain(_inline(cell)) for cell in row] for row in [header] + rows]
    widths = [max(len(row[j]) for row in grid) for j in range(columns)]

    if sum(widths) + 2 * (columns - 1) <= MONO_WIDTH and all(c.isascii() for row in grid for c in row):
        right = [cell.endswith(":") and not cell.startswith(":") for cell in rule]

        def aligned(row: list[str]) -> str:
            padded = (c.rjust(w) if r else c.ljust(w) for c, w, r in zip(row, widths, right))
            return "  ".join(padded).rstrip()

        text = [aligned(grid[0]), aligned(["-" * w for w in widths])] + [aligned(r) for r in grid[1:]]
        wrap = (block, "<pre>", "</pre>")
        return [_Line(html.escape(line, quote=False), wrap=wrap) for line in text]

    if columns == 2:
        lines = [_Line(f"<i>{_inline(header[0])} — {_inline(header[1])}</i>")]
        for first, second in rows:
            lines.append(_Line(f"• <b>{_inline(_unbold(first) or '—')}</b> — {_inline(second)}"))
        return lines

    lines = []
    for row in rows:
        lines.append(_Line(f"• <b>{_inline(_unbold(row[0]) or '—')}</b>"))
        for name, cell in zip(header[1:], row[1:]):
            if cell:
                lines.append(_Line(f"   {_inline(name)}: {_inline(cell)}"))
    return lines


def _parse(markdown: str) -> list[_Line]:
    source = markdown.split("\n")
    out: list[_Line] = []
    stack: list[int] = []
    block = 0
    i = 0
    while i < len(source):
        raw = source[i].rstrip()
        stripped = raw.strip()

        fence = _FENCE.match(raw)
        if fence or stripped in _MATH_FENCES:
            block += 1
            if fence:
                indent, marker, language = _indent(fence.group(1)), fence.group(2), fence.group(3)
            else:
                indent, marker, language = 0, _MATH_FENCES[stripped], ""
            body = []
            i += 1
            while i < len(source):
                candidate = source[i].strip()
                if fence and len(candidate) >= len(marker) and set(candidate) == {marker[0]}:
                    break
                if not fence and candidate == marker:
                    break
                line = source[i].rstrip()
                lead = len(line) - len(line.lstrip(" "))
                body.append(line[min(lead, indent):])
                i += 1
            i += 1
            while body and not body[0].strip():
                body.pop(0)
            while body and not body[-1].strip():
                body.pop()
            language = re.sub(r"[^\w+#.-]", "", language)
            if language:
                tags = (f'<pre><code class="language-{html.escape(language)}">', "</code></pre>")
            else:
                tags = ("<pre>", "</pre>")
            out += [_Line(html.escape(line, quote=False), wrap=(block, *tags)) for line in body]
            continue

        if "|" in raw and i + 1 < len(source) and _TABLE_RULE.match(source[i + 1]):
            header, rule = _cells(raw), _cells(source[i + 1])
            if len(header) >= 2 and len(header) == len(rule):
                block += 1
                rows = []
                i += 2
                while i < len(source) and source[i].strip() and "|" in source[i]:
                    rows.append(_cells(source[i]))
                    i += 1
                out += _table(header, rule, rows, block)
                stack.clear()
                continue

        if _QUOTE.match(raw):
            block += 1
            wrap = (block, "<blockquote>", "</blockquote>")
            quoted = []
            inner_stack: list[int] = []
            while i < len(source) and _QUOTE.match(source[i]):
                quoted.append(_text_line(_QUOTE_MARKS.sub("", source[i].rstrip()), inner_stack))
                i += 1
            while quoted and quoted[0].kind == "blank":
                quoted.pop(0)
            while quoted and quoted[-1].kind == "blank":
                quoted.pop()
            out += [_Line(line.html, wrap=wrap) for line in quoted]
            continue

        out.append(_text_line(raw, stack))
        i += 1
    return out


def _tidy(lines: list[_Line]) -> list[_Line]:
    """Normalise spacing: one blank line at most, a gap above headings, none around blocks
    or between the items of a list."""
    out: list[_Line] = []
    for line in lines:
        previous = out[-1] if out else None
        if line.kind == "blank":
            if previous is None or previous.kind in ("blank", "heading") or previous.wrap:
                continue
        elif line.kind == "item":
            if len(out) >= 2 and previous.kind == "blank" and out[-2].kind == "item":
                out.pop()
        elif line.kind == "heading":
            if previous is not None and previous.kind != "blank" and not previous.wrap:
                out.append(_BLANK)
        elif line.wrap and previous is not None and previous.kind == "blank":
            out.pop()
        out.append(line)
    while out and out[-1].kind == "blank":
        out.pop()
    return out


# ---------------------------------------------------------------------------
# Splitting into messages
# ---------------------------------------------------------------------------
def _pack(lines: list[_Line], limit: int) -> list[list[_Line]]:
    """Group lines into messages, preferring to break at a blank line."""
    messages: list[list[_Line]] = []
    current: list[tuple[_Line, int]] = []
    size = 0

    def close(count: int) -> None:
        nonlocal current, size
        messages.append([line for line, _ in current[:count]])
        current = current[count:]
        while current and current[0][0].kind == "blank":
            current.pop(0)
        size = sum(cost for _, cost in current)

    for line in lines:
        text = _plain(line.html)
        if _length(text) > limit:
            # A single line too long for one message: fall back to unformatted pieces.
            step = limit // 2
            pieces = [line._replace(html=html.escape(text[k : k + step], quote=False)) for k in range(0, len(text), step)]
        else:
            pieces = [line]
        for piece in pieces:
            cost = _length(_plain(piece.html)) + 1
            while current and size + cost > limit:
                seen, cut = 0, len(current)
                for index, (candidate, candidate_cost) in enumerate(current):
                    if candidate.kind == "blank" and seen >= limit // 2:
                        cut = index
                    seen += candidate_cost
                close(cut)
            current.append((piece, cost))
            size += cost
    if current:
        close(len(current))
    return [m for m in messages if any(line.kind != "blank" for line in m)]


def _assemble(lines: list[_Line]) -> Chunk:
    while lines and lines[-1].kind == "blank":
        lines = lines[:-1]
    parts = []
    for wrap, group in groupby(lines, key=lambda line: line.wrap):
        body = "\n".join(line.html for line in group)
        parts.append(f"{wrap[1]}{body}{wrap[2]}" if wrap else body)
    rendered = "\n".join(parts)
    return Chunk(rendered, _plain(rendered))


# ---------------------------------------------------------------------------
# Rich messages: Telegram renders the Markdown, so it only has to be made safe
# ---------------------------------------------------------------------------
# Telegram allows 32,768 characters and 500 blocks (paragraphs, list items, table rows)
# per rich message. These leave room for the escapes telegram_markdown() adds.
RICH_CHAR_LIMIT = 30_000
RICH_BLOCK_LIMIT = 450

# HTML tags Telegram understands inside Markdown and that are harmless to keep.
_SAFE_TAG = re.compile(
    r"</?(?:b|strong|i|em|u|ins|s|strike|del|sub|sup|mark|br|details|summary|tg-spoiler)(?:\s+open)?\s*/?>", re.I
)
_TAG_START = re.compile(r"<(?=[A-Za-z/!?])")
_RICH_ESCAPE = re.compile(r"\\[\\`*_{}\[\]()#+\-.!~|>$=&]")
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)")
_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]|\$\$(.+?)\$\$")
_DOLLAR_PAIR = re.compile(r"(?<![\\$\w])\$(?=[^\s$])([^$\n]*?)(?<=[^\s\\$])\$(?![\d$])")
_PROSE_WORD = re.compile(r"\b(?:and|or|to|is|in|of|the|for|per|if|at|by|on|as|vs)\b", re.I)
_IMAGE = re.compile(r"!\[([^\[\]\n]*)\]\(\s*<?([^\s()<>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_LOOSE_UNDERSCORE = re.compile(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])")
_LINE_PREFIX = re.compile(r"^(\s*(?:>\s?)*\s*(?:[-*+]\s+(?:\[[ xX]\]\s+)?|\d{1,9}[.)]\s+|#{1,6}\s+)?)(.*)$")


def telegram_markdown(markdown: str) -> str:
    """Rewrite Markdown so Telegram's rich-message parser reads it the way the model meant it.

    Telegram's parser is looser than GitHub's and treats things as markup that models write
    as plain text: "$20-$30" becomes a formula, "List<String>" loses its tag, "__init__"
    turns bold, a stray "*" starts italics, and a link to a missing image rejects the whole
    message. Text that only looks like markup is escaped here; real markup is kept.
    """
    out = []
    fence = None  # marker of the code block we are inside
    formula_end = None  # line that closes the formula block we are inside
    table = False
    source = _RESERVED.sub("", markdown.replace("\r\n", "\n")).split("\n")
    for i, raw in enumerate(source):
        stripped = raw.strip()
        if fence:
            out.append(raw)
            if len(stripped) >= len(fence) and set(stripped) == {fence[0]}:
                fence = None
            continue
        if formula_end:
            last = stripped.endswith(formula_end)
            body = stripped[: -len(formula_end)].rstrip() if last else raw
            if body.strip():
                out.append(_BARE_DOLLAR.sub(r"\\$", body))
            if last:
                out.append("$$")
                formula_end = None
            continue
        opening = _FENCE.match(raw)
        if opening:
            fence = opening.group(2)
            out.append(raw)
            continue
        # A formula block: "$$" or "\[" opens it and the matching mark on a later line closes it.
        start = stripped[:2]
        if start in _MATH_FENCES and _MATH_FENCES[start] not in stripped[2:]:
            formula_end = _MATH_FENCES[start]
            out += [line for line in ("$$", _BARE_DOLLAR.sub(r"\\$", stripped[2:].lstrip())) if line]
            continue
        if _HRULE.match(raw):
            # Directly under a paragraph, "---" would turn that paragraph into a heading.
            if out and out[-1].strip():
                out.append("")
            out.append(raw)
            continue
        if not stripped or _TABLE_RULE.match(raw):
            table = table and bool(stripped)
            out.append(raw)
            continue
        if "|" in raw and i + 1 < len(source) and _TABLE_RULE.match(source[i + 1]):
            table = True
        elif "|" not in raw:
            table = False
        prefix, body = _LINE_PREFIX.match(raw).groups()
        out.append(prefix + _safe_inline(body, table))
    return "\n".join(out)


_BARE_DOLLAR = re.compile(r"(?<!\\)\$")


def _formula(body: str, mark: str) -> str:
    """A formula between Telegram's marks: "$" inside a sentence, "$$" on its own."""
    body = body.strip()
    if "$" in body:
        # Money inside a formula: a single dollar sign would end the formula right there,
        # even an escaped one. Between double marks an escaped dollar sign is safe.
        return "$$" + _BARE_DOLLAR.sub(r"\\$", body) + "$$"
    return f"{mark}{body}{mark}"


def _is_formula(text: str) -> bool:
    """Whether the text between two dollar signs is maths rather than prices or shell variables."""
    if re.search(r"\\[A-Za-z]", text):  # a LaTeX command such as \frac
        return True
    return (
        len(text) <= 60
        and re.fullmatch(r"[A-Za-z0-9\s+\-*/=<>()\[\]{}^_,.|!']+", text) is not None
        and text[-1] not in "+-*/=<>([{,.|^_"
        and re.search(r"[A-Za-z^=]", text) is not None
        and not re.search(r"[A-Za-z]{4,}", text)
        and not _PROSE_WORD.search(text)
    )


def _safe_inline(text: str, table_row: bool) -> str:
    slots: list[str] = []

    def hold(kept: str) -> str:
        slots.append(kept)
        return f"{_SLOT_OPEN}{len(slots) - 1}{_SLOT_CLOSE}"

    def leaf(plain: str) -> str:
        plain = plain.replace("*", "\\*").replace("==", "\\=\\=").replace("$", "\\$")
        plain = _LOOSE_UNDERSCORE.sub(r"\\_", plain)
        if not table_row:
            plain = plain.replace("||", "\\|\\|")
        return _TAG_START.sub("&lt;", plain)

    def emphasis(part: str) -> str:
        return _spans(part, lambda style, inner: style[3] + inner + style[3], leaf)

    def image(m: re.Match) -> str:
        label, url = m.group(1) or "image", m.group(2)
        return f"[{label}]({url})" if re.match(r"https?://", url) else label

    text = _CODE_SPAN.sub(lambda m: hold(m.group()), text)
    text = _INLINE_MATH.sub(lambda m: hold(_formula(m.group(1), "$")), text)
    text = _DISPLAY_MATH.sub(lambda m: hold(_formula(m.group(1) or m.group(2), "$$")), text)
    text = _RICH_ESCAPE.sub(lambda m: hold(m.group()), text)
    text = _DOLLAR_PAIR.sub(lambda m: hold(m.group()) if _is_formula(m.group(1)) else m.group(), text)
    text = _IMAGE.sub(image, text)
    text = _LINK.sub(lambda m: hold(f"[{emphasis(m.group(1))}]({m.group(2)})"), text)
    text = _AUTOLINK.sub(lambda m: hold(m.group(1)), text)
    text = _BARE_URL.sub(lambda m: hold(m.group()), text)
    text = _SAFE_TAG.sub(lambda m: hold(m.group()), text)
    text = emphasis(text)
    while _SLOT.search(text):
        text = _SLOT.sub(lambda m: slots[int(m.group(1))], text)
    return text


def split_markdown(markdown: str, max_chars: int = RICH_CHAR_LIMIT, max_blocks: int = RICH_BLOCK_LIMIT) -> list[str]:
    """Cut a reply into pieces that each fit one rich message.

    Cuts fall on a blank line where there is one in the second half of a piece. A code
    block that has to be cut is closed in one piece and reopened in the next.
    """
    pieces: list[str] = []
    current: list[str] = []
    size = blocks = 0
    gap = None  # (line count, size, blocks) at the last blank line outside a code block
    fence = None  # (marker, opening line) of the code block we are inside
    step = max(max_chars // 2, 1)

    for raw in markdown.replace("\r\n", "\n").split("\n"):
        for line in [raw[k : k + step] for k in range(0, len(raw), step)] or [""]:
            cost = 0 if fence or not line.strip() else 1
            while current and (size + len(line) + 1 > max_chars or blocks + cost > max_blocks):
                if gap and (gap[1] >= max_chars // 2 or gap[2] >= max_blocks // 2):
                    count, used_size, used_blocks = gap
                    pieces.append("\n".join(current[:count]))
                    current, size, blocks, gap = current[count:], size - used_size, blocks - used_blocks, None
                else:
                    if fence and current == [fence[1]]:
                        break  # only the reopened code block is left, so there is nothing to cut
                    pieces.append("\n".join(current + [fence[0]] if fence else current))
                    current, size, blocks, gap = [], 0, 0, None
                    if fence:
                        current, size, blocks = [fence[1]], len(fence[1]) + 1, 1
            current.append(line)
            size += len(line) + 1
            blocks += cost
            stripped = line.strip()
            if fence:
                if len(stripped) >= len(fence[0]) and set(stripped) == {fence[0][0]}:
                    fence = None
            elif _FENCE.match(line):
                fence = (_FENCE.match(line).group(2), line)
            elif not stripped:
                gap = (len(current), size, blocks)
    pieces.append("\n".join(current))
    return [piece.strip("\n") for piece in pieces if piece.strip()]
