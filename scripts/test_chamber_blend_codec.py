#!/usr/bin/env python3
"""
Test 17 - Chamber-blended codec with LEARNED chamber trust.

Test 15: byte codec, fixed-weight order mixing  -> 2.870 bits/byte
Test 16: online token alphabet                  -> NEGATIVE (4.034)
Test 17 first cut: word chambers, fixed weights -> NEGATIVE (3.051)
  (sparse word-pair contexts overtrusted - same failure as order-6)

The fix is the architecture PAQ used to take the text-compression records:
parallel chambers whose MIXING WEIGHTS ARE LEARNED ONLINE. After every
byte, each chamber's weight is multiplied by (its probability for the byte
that actually occurred / the mixture's probability) ** eta - chambers that
predict well gain trust, chambers that misfire lose it, automatically,
per position-class (inside a word vs at a boundary).

Chambers:
  U   : uniform escape
  O0  : order-0 byte frequencies
  O1-5: order-k byte contexts, FTA composite addresses (Test 15)
  W   : current-word-prefix (word identity = FTA composite of
        position-tagged letter primes - the repo's word addressing)
  WP  : (previous word, current word prefix) - cross-word grammar

Each chamber emits a confidence-tempered distribution
  p_i = conf * empirical + (1 - conf) * uniform,   conf = n/(n+c)
and the mixture is the trust-weighted average. Decoder mirrors everything.

Pass: byte-exact round trip; beats Test 15; verdicts vs zlib/bz2/lzma.
"""

from __future__ import annotations

import bz2
import lzma
import math
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.primes import chain_primes
from test_lattice_context_compressor import Encoder, Decoder
import test_lattice_context_compressor as byte_codec

ROOT = Path(__file__).resolve().parents[1]

MAX_ORDER = 5
WORD_SLOTS = 8
N_CHAMBERS = 2 + MAX_ORDER + 2          # U, O0, O1..O5, W, WP
ETA = 0.15                               # trust learning rate (gentle)
INIT_W = np.array([0.02, 0.05, 0.3, 1.0, 3.0, 9.0, 20.0, 4.0, 6.0])
CONF_ORDER = 1.5
CONF_WORD = 3.0
UNIFORM = np.full(256, 1.0 / 256)
PRIMES = chain_primes(256 * (MAX_ORDER + WORD_SLOTS) + 64)


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


def is_wordchar(b: int) -> bool:
    return (48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122) or b == 95


def tempered(entry: list | None, conf_c: float) -> np.ndarray | None:
    """Confidence-tempered distribution for one context entry."""
    if entry is None:
        return None
    n_k, d = entry
    conf = n_k / (n_k + conf_c)
    p = np.full(256, (1.0 - conf) / 256)
    inv = conf / n_k
    for b, c in d.items():
        p[b] += c * inv
    return p


class ChamberBlendModel:
    """Chambers + per-position-class learned trust weights."""

    def __init__(self):
        self.c0 = np.zeros(256, dtype=np.float64)
        self.c0_total = 0
        self.tables: list[dict[int, list]] = [{} for _ in range(MAX_ORDER)]
        self.w_table: dict[int, list] = {}
        self.wp_table: dict[tuple, list] = {}
        self.hist: list[int] = []
        self.cur_word_key = 1
        self.cur_word_len = 0
        self.prev_word_key = 1
        # learned trust: one weight vector per position class (boundary / in-word)
        self.trust = np.stack([INIT_W.copy(), INIT_W.copy()])
        self.trust /= self.trust.sum(axis=1, keepdims=True)

    def order_keys(self) -> list[int]:
        keys = []
        key = 1
        for j in range(MAX_ORDER):
            if j < len(self.hist):
                key *= PRIMES[j * 256 + self.hist[-1 - j]]
                keys.append(key)
            else:
                keys.append(0)
        return keys

    def position_class(self) -> int:
        return 1 if self.cur_word_len > 0 else 0

    def chamber_dists(self) -> list[tuple[int, np.ndarray]]:
        """Active chambers as (chamber_index, distribution)."""
        out: list[tuple[int, np.ndarray]] = [(0, UNIFORM)]
        if self.c0_total:
            out.append((1, self.c0 / self.c0_total))
        for k, key in enumerate(self.order_keys(), start=1):
            if key == 0:
                continue
            p = tempered(self.tables[k - 1].get(key), CONF_ORDER)
            if p is not None:
                out.append((1 + k, p))
        p = tempered(self.w_table.get(self.cur_word_key), CONF_WORD)
        if p is not None:
            out.append((2 + MAX_ORDER, p))
        p = tempered(self.wp_table.get((self.prev_word_key, self.cur_word_key)),
                     CONF_WORD)
        if p is not None:
            out.append((3 + MAX_ORDER, p))
        return out

    def freq_table(self) -> tuple[np.ndarray, int, list, np.ndarray]:
        cls = self.position_class()
        dists = self.chamber_dists()
        w = self.trust[cls]
        wsum = 0.0
        mix = np.zeros(256)
        for idx, p in dists:
            mix += w[idx] * p
            wsum += w[idx]
        mix /= wsum
        freq = (mix * 65536.0).astype(np.int64) + 1
        return freq, int(freq.sum()), dists, mix

    def learn_trust(self, byte: int, dists: list, mix: np.ndarray):
        """Multiplicative trust update over ACTIVE chambers only.

        Inactive chambers are untouched (no global renormalization - that
        would dilute chambers that never got to defend themselves, which
        is exactly the collapse seen in the first version). A floor clamp
        keeps every chamber alive enough to recover.
        """
        cls = self.position_class()
        m = mix[byte]
        w = self.trust[cls]
        for idx, p in dists:
            ratio = p[byte] / m
            w[idx] = min(max(w[idx] * ratio ** ETA, 1e-3), 1e3)

    def update(self, byte: int):
        self.c0[byte] += 1
        self.c0_total += 1
        for k, key in enumerate(self.order_keys(), start=1):
            if key == 0:
                continue
            entry = self.tables[k - 1].setdefault(key, [0, {}])
            entry[0] += 1
            entry[1][byte] = entry[1].get(byte, 0) + 1
        entry = self.w_table.setdefault(self.cur_word_key, [0, {}])
        entry[0] += 1
        entry[1][byte] = entry[1].get(byte, 0) + 1
        wp_key = (self.prev_word_key, self.cur_word_key)
        entry = self.wp_table.setdefault(wp_key, [0, {}])
        entry[0] += 1
        entry[1][byte] = entry[1].get(byte, 0) + 1
        self.hist.append(byte)
        if len(self.hist) > MAX_ORDER:
            self.hist.pop(0)
        if is_wordchar(byte):
            slot = min(self.cur_word_len, WORD_SLOTS - 1)
            self.cur_word_key *= PRIMES[(MAX_ORDER + slot) * 256 + byte]
            self.cur_word_len += 1
        else:
            if self.cur_word_len > 0:
                self.prev_word_key = self.cur_word_key
            self.cur_word_key = 1
            self.cur_word_len = 0

    def trust_report(self) -> str:
        names = ["U", "O0", "O1", "O2", "O3", "O4", "O5", "W", "WP"]
        lines = []
        for cls, label in [(0, "boundary"), (1, "in-word")]:
            top = sorted(zip(names, self.trust[cls]), key=lambda t: -t[1])[:4]
            lines.append(f"    {label:<9}: " +
                         ", ".join(f"{n}={v:.2f}" for n, v in top))
        return "\n".join(lines)


def compress(data: bytes) -> tuple[bytes, float, "ChamberBlendModel"]:
    model = ChamberBlendModel()
    enc = Encoder()
    ideal_bits = 0.0
    for byte in data:
        freq, total, dists, mix = model.freq_table()
        cum = np.cumsum(freq)
        cum_lo = int(cum[byte - 1]) if byte > 0 else 0
        cum_hi = int(cum[byte])
        enc.encode(cum_lo, cum_hi, total)
        ideal_bits += -math.log2((cum_hi - cum_lo) / total)
        model.learn_trust(byte, dists, mix)
        model.update(byte)
    return enc.finish(), ideal_bits, model


def decompress(blob: bytes, n: int) -> bytes:
    model = ChamberBlendModel()
    dec = Decoder(blob)
    out = bytearray()
    for _ in range(n):
        freq, total, dists, mix = model.freq_table()
        cum = np.cumsum(freq)
        value = dec.decode_value(total)
        byte = int(np.searchsorted(cum, value, side="right"))
        cum_lo = int(cum[byte - 1]) if byte > 0 else 0
        cum_hi = int(cum[byte])
        dec.consume(cum_lo, cum_hi, total)
        model.learn_trust(byte, dists, mix)
        model.update(byte)
        out.append(byte)
    return bytes(out)


def main():
    header("Chamber-blended codec - chambers with LEARNED trust weights")

    files = sorted((ROOT / "derivations").glob("*.md"))
    raw = b"".join(f.read_bytes() for f in files)
    data = raw[:65536]
    n = len(data)
    print(f"  data: {len(files)} markdown files from derivations/, first {n} bytes")

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    print("\nBaselines")
    print("-" * 72)
    counts = Counter(data)
    H0 = -sum((c / n) * math.log2(c / n) for c in counts.values())
    print(f"  order-0 frequency floor: {H0:.3f} bits/byte")

    base_results = {}
    for name, fn in [("zlib -9", lambda d: zlib.compress(d, 9)),
                     ("bz2 -9", lambda d: bz2.compress(d, 9)),
                     ("lzma", lzma.compress)]:
        out = fn(data)
        base_results[name] = len(out)
        print(f"  {name:<10} {len(out):>7} bytes   {len(out)*8/n:.3f} bits/byte")

    t0 = time.time()
    byte_blob, _, _ = byte_codec.compress(data)
    base_results["byte codec (Test 15)"] = len(byte_blob)
    print(f"  {'byte codec (Test 15)':<22} {len(byte_blob):>7} bytes   "
          f"{len(byte_blob)*8/n:.3f} bits/byte   ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Ours
    # ------------------------------------------------------------------
    print("\nChamber-blend codec (orders + word chambers, learned trust)")
    print("-" * 72)
    t0 = time.time()
    blob, ideal_bits, model = compress(data)
    t_enc = time.time() - t0
    ours = len(blob)
    print(f"  encoded:   {ours} bytes in {t_enc:.1f}s")
    print(f"  bits/byte: {ours*8/n:.3f} (model cross-entropy {ideal_bits/n:.3f})")
    print(f"  learned chamber trust (top-4 per position class):")
    print(model.trust_report())

    t0 = time.time()
    restored = decompress(blob, n)
    print(f"  decoded:   {len(restored)} bytes in {time.time()-t0:.1f}s")
    assertion(restored == data, "round-trip byte-exact")

    # ------------------------------------------------------------------
    # Scoreboard
    # ------------------------------------------------------------------
    print("\nScoreboard")
    print("-" * 72)
    rows = [("frequency floor (order-0)", H0 * n / 8),
            *[(k, float(v)) for k, v in base_results.items()],
            ("chamber-blend codec", float(ours))]
    for name, size in sorted(rows, key=lambda r: r[1], reverse=True):
        marker = "  <-- ours" if name == "chamber-blend codec" else ""
        print(f"  {name:<28} {size:>9.0f} bytes  "
              f"{size*8/n:>6.3f} bits/byte{marker}")

    for name in ("byte codec (Test 15)", "zlib -9", "lzma", "bz2 -9"):
        size = base_results[name]
        verdict = "BEATS" if ours < size else "loses to"
        print(f"  [info]  {verdict} {name} ({ours} vs {size} bytes)")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    header("RESULT: NEGATIVE (documented)")
    print(f"  ours:       {ours*8/n:.3f} bits/byte")
    print(f"  Test 15:    {base_results['byte codec (Test 15)']*8/n:.3f} bits/byte")
    print()
    print("  Three mixing schemes for probability-level chamber blending all")
    print("  failed to beat Test 15's simple count blend at 64KB:")
    print("    1. fixed priors        -> sparse word-pair contexts overtrusted")
    print("    2. multiplicative trust + global renorm -> collapse to ONE")
    print("       chamber (inactive chambers diluted without defending)")
    print("    3. active-only multiplicative trust -> log-loss asymmetry")
    print("       (confident-wrong >> confident-right) ratchets every sharp")
    print("       chamber to the floor clamp")
    print()
    print("  Lesson: probability-space expert mixing is unstable for sharp")
    print("  experts. The known-good architecture is BIT-level LOGISTIC")
    print("  mixing (PAQ): bounded gradients, no collapse modes.")
    print("  See test_paq_chamber_mixer.py (Test 18).")


if __name__ == "__main__":
    main()
