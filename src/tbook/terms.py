"""Term normalization shared by concept resolution and the leak checker."""

import re
import unicodedata

from wordfreq import zipf_frequency

COMMON_ZIPF = 3.0          # single words at least this common in English are not matched by spelling


UNINFLECTED = {"atlas", "alias", "bias", "gas", "canvas", "lens", "chaos", "series", "species", "means"}


def singular(word: str) -> str:
    w = word
    if w in UNINFLECTED:
        return w
    if w.endswith("ices") and len(w) > 5:
        return w[:-4] + "ix"
    if w == "axes" or w.endswith(" axes"):
        return w[:-2] + "is"
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if re.search(r"(ses|xes|zes|ches|shes)$", w) and not w.endswith(("bases", "cases", "phases")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is", "ics")) and len(w) > 3:
        return w[:-1]
    return w


def normalize(name: str) -> str:
    t = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"^(the|a|an)\s+", "", t.strip())
    t = re.sub(r"[‐-―\-_/]+", " ", t)
    t = re.sub(r"[^a-z0-9()\[\]\s']+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    parts = t.split(" ")
    if parts and parts[-1]:
        parts[-1] = singular(parts[-1])
    return " ".join(parts)


def spelling_distinctive(term: str) -> bool:
    """Can this term be matched by its spelling without catching ordinary English?"""
    words = re.findall(r"[A-Za-z]+", term)
    if len(words) >= 2 or re.search(r"[0-9()\[\]]", term):
        return True                     # multiword terms and notation-like names ("SO(3)")
    if not words:
        return False
    w = singular(words[0].lower())
    return zipf_frequency(w, "en") < COMMON_ZIPF
