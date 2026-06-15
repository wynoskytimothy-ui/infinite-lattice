"""
wiki_teacher.py - a LIVE teacher: auto-source a term's definition from Wikipedia.

This is the production `teacher(term)` for the self-learning loop - no human
glossary, no API key. The engine hands it a gap term; it returns Wikipedia's
definition (the lead extract), which carries the bridging vocabulary the gold
docs use. Results are cached to disk (wiki_cache.json) so runs are reproducible
and we never re-fetch. Swappable for an LLM API or UMLS the same way.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

_CACHE_PATH = Path(__file__).resolve().parent / "wiki_cache.json"
_UA = {"User-Agent": "lattice-rag-distiller/0.1 (research)"}


def _load_cache():
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache):
    _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _fetch_summary(title):
    url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
           + urllib.parse.quote(title.replace(" ", "_")))
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA),
                                    timeout=12) as r:
            d = json.load(r)
        if d.get("type") == "disambiguation":
            return ""
        return (d.get("extract") or "").strip()
    except Exception:
        return ""


def _search_title(term):
    """map a term (often an abbreviation/plural) to its best Wikipedia article."""
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "opensearch", "search": term, "limit": 1, "namespace": 0,
         "format": "json"})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA),
                                    timeout=12) as r:
            d = json.load(r)
        return d[1][0] if len(d) > 1 and d[1] else None
    except Exception:
        return None


def define(term, cache=None):
    """Wikipedia definition for `term`, trying a few title normalisations.
    Returns '' if no usable page. Uses/updates the on-disk cache."""
    own = cache is None
    if own:
        cache = _load_cache()
    if term in cache:
        return cache[term]
    variants = [term, term.upper(), term.capitalize()]
    if term.endswith("s") and len(term) > 3:
        s = term[:-1]
        variants += [s, s.upper(), s.capitalize()]
    out = ""
    for t in variants:
        out = _fetch_summary(t)
        if out:
            break
    if not out:                                    # abbreviation/disambig -> search
        title = _search_title(term)
        if title:
            out = _fetch_summary(title)
    cache[term] = out
    if own:
        _save_cache(cache)
    return out


def define_many(terms):
    """Define a list of terms, persisting the cache once."""
    cache = _load_cache()
    out = {}
    for t in terms:
        out[t] = define(t, cache)
    _save_cache(cache)
    return out


if __name__ == "__main__":
    samples = ["copeptin", "lats1", "rxrs", "bcl2", "yap", "mds", "th17",
               "nickel", "spores", "colchicine", "ucb", "pge2", "aspirin",
               "sequestration", "ppar", "foxo3a", "biomaterials", "chabaudi",
               "myelodysplastic", "drosophila"]
    defs = define_many(samples)
    got = sum(1 for d in defs.values() if d)
    print(f"Wikipedia teacher: defined {got}/{len(samples)} sample gap terms\n")
    for t, d in defs.items():
        print(f"  {t:14s} {'OK ' if d else 'no '} {d[:90]}")
