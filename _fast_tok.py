"""Lightweight tokenizer worker — imports ONLY `re` so ProcessPool workers spawn fast on Windows
(no numpy/aethos re-import per worker). Tokenizer is a VERBATIM copy of aethos_append_index.words;
the fast-ingest identical-index gate catches any drift."""
import re

_TOK = re.compile(r"[a-z][a-z0-9]+")
_STOP = set("a an and are as at be by for from has he in is it its of on that the "
            "to was were will with this these those which who we our you your they "
            "their them not no can may also been being have had but or if than then "
            "so such into over under more most some any all".split())
POS_HEAD, POS_BOOST = 14, 1.6


def words(text):
    return [w for w in _TOK.findall(text.lower()) if w not in _STOP and len(w) > 2]


def tok_bag_chunk(texts):
    """Tokenize + per-doc bag (positional tf). Returns flat (tokens, weights, items_per_doc) for the chunk.
    Doing the bag here moves BOTH heavy Python stages (tokenize + bag) into the worker process."""
    flat_t, flat_w, nitems = [], [], []
    for text in texts:
        tl = [w for w in _TOK.findall(text.lower()) if w not in _STOP and len(w) > 2]
        bag = {}; bget = bag.get
        for i, w in enumerate(tl):
            bag[w] = bget(w, 0.0) + (POS_BOOST if i < POS_HEAD else 1.0)
        nitems.append(len(bag))
        for w, wt in bag.items():
            flat_t.append(w); flat_w.append(wt)
    return flat_t, flat_w, nitems
