import json
import re
from typing import Any

_SURROGATE = re.compile("[\ud800-\udfff]")


def norm(s: Any) -> str:
    """Text as the page shows it: whitespace runs collapse to one space, ends are trimmed."""
    if isinstance(s, bool):
        s = "true" if s else "false"
    return re.sub(r"[ \t\n\r\f\ufeff]+", " ", "" if s is None else str(s)).strip()


def fold(s: str) -> str:
    """For matching only, never shown: whitespace collapsed and lowercased."""
    return re.sub(r"[\s\ufeff]+", " ", s).lower()


def clip(s: str, n: int) -> str:
    return f"{s[:n]}…" if len(s) > n else s


def quote(s: str) -> str:
    return _SURROGATE.sub(lambda m: f"\\u{ord(m.group()):04x}", json.dumps(s, ensure_ascii=False))


def well_formed(s: str) -> str:
    return _SURROGATE.sub("\ufffd", s)

# Icon fonts (Material Icons etc.) put their glyph names in the DOM as text: "vpn_key", "hotel Stays".
_ICON_WORD = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_LEADING_ICON = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*\s+(?=[A-Z0-9])")
_TRAILING_ICON = re.compile(r"\s+[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")  # needs an underscore: "expand_more"


def is_icon_text(t: str) -> bool:
    """A text line that is only an icon glyph name: one lowercase token like "vpn_key" or "help"."""
    return len(t) <= 30 and bool(_ICON_WORD.fullmatch(t)) and ("_" in t or len(t) >= 4)


def clean_name(name: str) -> str:
    """Drop icon glyph names from the ends of a control's name, for display and for icon-free locators:
    "hotel Stays" -> "Stays", "airline_seat_recline_normal Economy expand_more" -> "Economy"."""
    out = name
    m = _LEADING_ICON.match(out)
    if m and out[m.end():]:
        out = out[m.end():]
    while True:
        m = _TRAILING_ICON.search(out)
        if not m or not out[:m.start()]:
            return out
        out = out[:m.start()]