import re


_USERNAME = re.compile(
    r"\[\[user.*?\]\]|\[\[user talk.*?\]\]|user:\w+", re.IGNORECASE
)
_URL = re.compile(r"http\S+|www\.\S+")
_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")
_WIKI_TEMPLATE = re.compile(r"\{\{.*?\}\}")
_WIKI_BOLD_ITALIC = re.compile(r"'{2,5}")
_WIKI_HEADING = re.compile(r"=+\s*(.*?)\s*=+")
_STRETCHED_CHARS = re.compile(r"(.)\1{2,}")
_WHITESPACE = re.compile(r"\s+")



def preprocess(text: str | None) -> str:
    
    if not text or not text.strip():
        return "[UNK]"

    text = text.lower()
    text = _USERNAME.sub(" ", text)
    text = _URL.sub("[UNK]", text)
    text = _WIKI_LINK.sub(r"\1", text)        # [[link|display]] → display
    text = _WIKI_TEMPLATE.sub(" ", text)       # {{template}} → removed
    text = _WIKI_BOLD_ITALIC.sub("", text)     # ''italic''/'''bold''' → removed
    text = _WIKI_HEADING.sub(r"\1", text)      # == Heading == → Heading
    text = _STRETCHED_CHARS.sub(r"\1\1", text) # looool → lool (collapse to 2)
    text = _WHITESPACE.sub(" ", text).strip()

    return text if text else "[UNK]"