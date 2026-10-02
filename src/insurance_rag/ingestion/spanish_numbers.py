"""Parse Spanish cardinal and ordinal number words as used in provision titles.

Older laws number their articles in words ("Artículo dieciocho", "Artículo ciento veintiuno",
"Artículo sexto bis") and every law numbers its additional, transitional and final provisions
with ordinals ("Disposición adicional decimotercera"). Retrieval and citation need the numeric
form, so this module turns those words into integers.
"""

import re
import unicodedata

_UNITS = {
    "cero": 0,
    "un": 1,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiun": 21,
    "veintiuno": 21,
    "veintiuna": 21,
    "veintidos": 22,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
}
_TENS = {
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}
_HUNDREDS = {
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "trescientos": 300,
    "cuatrocientos": 400,
    "quinientos": 500,
    "seiscientos": 600,
    "setecientos": 700,
    "ochocientos": 800,
    "novecientos": 900,
}

# Ordinal stems without the gender ending (-o/-a); "tercer"/"primer" are apocopated forms.
_ORDINAL_UNITS = {
    "primer": 1,
    "segund": 2,
    "tercer": 3,
    "cuart": 4,
    "quint": 5,
    "sext": 6,
    "septim": 7,
    "setim": 7,
    "octav": 8,
    "noven": 9,
}
_ORDINAL_TENS = {
    "decim": 10,
    "vigesim": 20,
    "trigesim": 30,
    "cuadragesim": 40,
    "quincuagesim": 50,
    "sexagesim": 60,
    "septuagesim": 70,
    "octogesim": 80,
    "nonagesim": 90,
}
_ORDINAL_SPECIAL = {"undecim": 11, "duodecim": 12, "centesim": 100}

_LATIN_SUFFIXES = (
    "bis",
    "ter",
    "quater",
    "quinquies",
    "sexies",
    "septies",
    "octies",
    "nonies",
    "decies",
)

_TOKEN_RE = re.compile(r"[a-z]+|\d+")


def normalize(text: str) -> str:
    """Lowercase and strip accents, keeping only ASCII letters, digits and spaces."""
    decomposed = unicodedata.normalize("NFKD", text.replace("\xa0", " "))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower()


def parse_cardinal(words: str) -> int | None:
    """Parse a cardinal written in words, e.g. "ciento veintiuno" -> 121."""
    tokens = [t for t in normalize(words).split() if t != "y"]
    if not tokens:
        return None
    total = 0
    for token in tokens:
        if token in _HUNDREDS:
            value = _HUNDREDS[token]
        elif token in _TENS:
            value = _TENS[token]
        elif token in _UNITS:
            value = _UNITS[token]
        else:
            return None
        total += value
    return total


def _parse_ordinal_word(word: str) -> int | None:
    stem = word[:-1] if word.endswith(("o", "a")) else word
    if stem in _ORDINAL_SPECIAL:
        return _ORDINAL_SPECIAL[stem]
    if stem in _ORDINAL_UNITS:
        return _ORDINAL_UNITS[stem]
    if stem in _ORDINAL_TENS:
        return _ORDINAL_TENS[stem]
    # Fused compounds such as "decimotercera" or "vigesimoprimera"; the linking vowel is elided
    # before another vowel ("decimoctava").
    for prefix, tens in _ORDINAL_TENS.items():
        if not word.startswith(prefix):
            continue
        for rest_word in (word[len(prefix) + 1 :], word[len(prefix) :]):
            rest = _parse_ordinal_word(rest_word) if len(rest_word) > 1 else None
            if rest is not None and rest < 10:
                return tens + rest
    return None


def parse_ordinal(words: str) -> int | None:
    """Parse an ordinal written in words, e.g. "vigésimo primera" or "decimotercera" -> 13/21."""
    tokens = normalize(words).split()
    if not tokens:
        return None
    total = 0
    for token in tokens:
        value = _parse_ordinal_word(token)
        if value is None:
            return None
        total += value
    return total


def parse_number(words: str) -> int | None:
    """Parse a number given as digits, cardinal words or ordinal words."""
    text = normalize(words).strip().rstrip(".")
    text = re.sub(r"[.ºª°o]+$", "", text) if re.fullmatch(r"\d+[.ºª°o]*", text) else text
    if text.isdigit():
        return int(text)
    cardinal = parse_cardinal(text)
    return cardinal if cardinal is not None else parse_ordinal(text)


def parse_provision_number(title: str) -> str | None:
    """Return the canonical number of a provision from its title.

    >>> parse_provision_number("Artículo dieciocho")
    '18'
    >>> parse_provision_number("Artículo 10 bis")
    '10 bis'
    >>> parse_provision_number("Disposición adicional decimotercera")
    '13'
    >>> parse_provision_number("Disposición final única")
    'única'
    """
    text = normalize(title)
    text = re.sub(r"^(articulo|disposicion (adicional|transitoria|derogatoria|final))\s*", "", text)
    text = text.strip().rstrip(".")
    if not text:
        return None
    if text in {"unico", "unica"}:
        return "única" if text == "unica" else "único"

    tokens: list[str] = _TOKEN_RE.findall(text)
    suffix = ""
    if tokens and tokens[-1] in _LATIN_SUFFIXES:
        suffix = " " + tokens.pop()
    elif len(tokens) > 1 and re.search(r"\b[a-z]\)$", text):
        # Lettered articles such as "Artículo setenta y seis. a)".
        suffix = " " + tokens.pop()
    if not tokens:
        return None
    if tokens[0].isdigit():
        return tokens[0] + suffix
    number = parse_number(" ".join(tokens))
    return None if number is None else f"{number}{suffix}"
