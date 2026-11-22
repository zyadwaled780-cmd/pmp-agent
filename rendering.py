"""Safe rendering of text for the Gradio UI.

Stored data is never modified for display; these helpers only produce *display* strings.
* ``escape_md``                 - untrusted plain text (task titles, owners, comments, member and sender names):
                                  every markdown / HTML control character is escaped, newlines are flattened.
* ``sanitize_model_markdown``   - model-written reports: headings, bullets and bold are kept, but images, links,
                                  raw HTML and bare URLs are removed (they could exfiltrate data or phish).
"""
import re

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~&])")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]*>")
_URL = re.compile(r"(?:https?|ftp|javascript|data):[^\s)>\]]+", re.I)


def escape_md(text, limit: int = 300) -> str:
    s = " ".join(str(text if text is not None else "").split())
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return _MD_SPECIAL.sub(r"\\\1", s)


def sanitize_model_markdown(text, limit: int = 6000) -> str:
    s = str(text if text is not None else "")[:limit]
    s = _IMAGE.sub("[image removed]", s)
    s = _LINK.sub(r"\1", s)
    s = _HTML_TAG.sub("", s)
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return _URL.sub("[link removed]", s)
