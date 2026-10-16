from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def safe_text(value: object) -> str:
    """Collapse every whitespace run to one space and strip (production ``safe_text``)."""

    return _WHITESPACE.sub(" ", str(value or "")).strip()
