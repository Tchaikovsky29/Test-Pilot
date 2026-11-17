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
