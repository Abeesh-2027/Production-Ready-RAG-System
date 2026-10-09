from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)  # also turns NBSP into a normal space
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]

    # PDFs hard-wrap every line, so a single "\n" usually does NOT mean a new
    # paragraph. A heuristic: a line that ends a sentence (or is very short, like
    # a heading) starts a new paragraph after it; otherwise join with a space.
    paragraphs: list[str] = []
    current: list[str] = []
    for i, line in enumerate(lines):
        if not line:  # a real blank line is a paragraph break
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
        next_line = next((l for l in lines[i + 1:] if l), "")
        if _ends_paragraph(line, next_line):
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(p for p in paragraphs if p)


def _ends_paragraph(line: str, next_line: str) -> bool:
    """True if `line` looks like the last line of a paragraph.

    * Lines ending in sentence punctuation end a paragraph.
    * A short line without a trailing comma is probably a heading - but only if the
      NEXT line starts a new thought (capital letter, digit or bullet). A wrapped
      continuation line starts in lowercase, so we keep joining those.
    """
    if line.endswith((".", "!", "?", '."', ".”")):
        return True
    if len(line) < 60 and not line.endswith((",", ";", ":")):
        return not next_line or not next_line[0].islower()
    return False
