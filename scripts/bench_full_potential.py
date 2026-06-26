#!/usr/bin/env python3
import os, sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lattice_retriever_v1.trigger_formula_codec import decode_trigger_formula, encode_trigger_formula

def cmp(label, data):
    w, m = encode_trigger_formula(data)
    assert decode_trigger_formula(w) == data
    z = len(zlib.compress(data, 9))
    win = "lattice" if len(w) < z else "zlib"
    print(f"{label}: raw={len(data):,} lattice={len(w):,} ({len(data)/len(w):.1f}x) mode={m['mode']} zlib={z:,} ({len(data)/z:.1f}x) winner={win}")

cmp("single 100KB", bytes([42]) * 100000)
cmp("two-branch 100KB", bytes([1, 2]) * 50000)
cmp("10-digit 100KB", bytes(range(10)) * 10000)
cmp("english phrase", b"the cat sat on the mat " * 200)
cmp("random 10KB", bytes(os.urandom(10000)))
