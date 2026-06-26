#!/usr/bin/env python3
"""
Compare lattice storage codecs vs standard compressors.

  python scripts/bench_compression_benchmarks.py
  python scripts/bench_compression_benchmarks.py --sample-kb 500 --alphabet 200
  python scripts/bench_compression_benchmarks.py --corpus random,repetitive,english
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
import random
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_retriever_v1.dot_blob_codec import compress_to_dot_blob, reconstruct_from_dot_blob
from lattice_retriever_v1.personal_lattice_codec import (
    PersonalKey,
    decode_personal,
    encode_personal,
)
from lattice_retriever_v1.storage_codec import encode_storage, decode_storage


@dataclass
class BenchRow:
    name: str
    corpus: str
    raw_bytes: int
    compressed_bytes: int
    bare_lumber_bytes: int | None
    ratio: float
    encode_ms: float
    decode_ms: float
    roundtrip_ok: bool

    def as_dict(self) -> dict:
        return {
            "codec": self.name,
            "corpus": self.corpus,
            "raw_bytes": self.raw_bytes,
            "compressed_bytes": self.compressed_bytes,
            "bare_lumber_bytes": self.bare_lumber_bytes,
            "ratio_x": round(self.ratio, 3),
            "encode_ms": round(self.encode_ms, 2),
            "decode_ms": round(self.decode_ms, 2),
            "roundtrip_ok": self.roundtrip_ok,
        }


def _ratio(raw: int, comp: int) -> float:
    return raw / comp if comp else 0.0


def _bench_std(name: str, corpus: str, data: bytes, compress_fn, decompress_fn) -> BenchRow:
    t0 = time.perf_counter()
    comp = compress_fn(data)
    enc_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    back = decompress_fn(comp)
    dec_ms = (time.perf_counter() - t1) * 1000
    return BenchRow(
        name=name,
        corpus=corpus,
        raw_bytes=len(data),
        compressed_bytes=len(comp),
        bare_lumber_bytes=None,
        ratio=_ratio(len(data), len(comp)),
        encode_ms=enc_ms,
        decode_ms=dec_ms,
        roundtrip_ok=back == data,
    )


def _bench_lattice_dot(corpus: str, data: bytes) -> BenchRow:
    t0 = time.perf_counter()
    _, ledger, wire = compress_to_dot_blob(data)
    enc_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    back = reconstruct_from_dot_blob(wire)
    dec_ms = (time.perf_counter() - t1) * 1000
    return BenchRow(
        name="lattice_dot_blob",
        corpus=corpus,
        raw_bytes=len(data),
        compressed_bytes=len(wire),
        bare_lumber_bytes=ledger.bare_lumber_bytes,
        ratio=_ratio(len(data), len(wire)),
        encode_ms=enc_ms,
        decode_ms=dec_ms,
        roundtrip_ok=back == data,
    )


def _bench_lattice_lumber_only(corpus: str, data: bytes) -> BenchRow:
    _, ledger, _ = compress_to_dot_blob(data)
    return BenchRow(
        name="lattice_bare_lumber_only",
        corpus=corpus,
        raw_bytes=len(data),
        compressed_bytes=ledger.bare_lumber_bytes,
        bare_lumber_bytes=ledger.bare_lumber_bytes,
        ratio=_ratio(len(data), ledger.bare_lumber_bytes),
        encode_ms=0.0,
        decode_ms=0.0,
        roundtrip_ok=False,
    )


def _bench_lattice_lst1(corpus: str, data: bytes) -> BenchRow:
    t0 = time.perf_counter()
    wire, ledger, _ = encode_storage(data, min_cohesion=0.8)
    enc_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    back = decode_storage(wire)
    dec_ms = (time.perf_counter() - t1) * 1000
    return BenchRow(
        name="lattice_lst1_patterns",
        corpus=corpus,
        raw_bytes=len(data),
        compressed_bytes=len(wire),
        bare_lumber_bytes=ledger.alphabet_bytes,
        ratio=_ratio(len(data), len(wire)),
        encode_ms=enc_ms,
        decode_ms=dec_ms,
        roundtrip_ok=back == data,
    )


def _bench_lattice_personal(corpus: str, data: bytes, key: PersonalKey) -> BenchRow:
    t0 = time.perf_counter()
    _, _, wire = encode_personal(data, key)
    enc_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    back = decode_personal(wire, key)
    dec_ms = (time.perf_counter() - t1) * 1000
    return BenchRow(
        name="lattice_personal_keyed",
        corpus=corpus,
        raw_bytes=len(data),
        compressed_bytes=len(wire),
        bare_lumber_bytes=None,
        ratio=_ratio(len(data), len(wire)),
        encode_ms=enc_ms,
        decode_ms=dec_ms,
        roundtrip_ok=back == data,
    )


def generate_corpus(kind: str, n: int, alphabet: int, seed: int) -> bytes:
    rng = random.Random(seed)
    if kind == "repetitive":
        hot = list(range(min(10, alphabet)))
        chunk = bytes(rng.choice(hot) for _ in range(24))
        return (chunk * (n // len(chunk) + 1))[:n]
    if kind == "random_bounded":
        return bytes(rng.randint(0, alphabet - 1) for _ in range(n))
    if kind == "random_byte":
        return bytes(rng.randint(0, 255) for _ in range(n))
    if kind == "english_like":
        words = (
            "the a of and to in is you that it he for was on are as with his they "
            "be at one have this from or had by hot but some what there we can out "
            "other were all your when up use word how said an each she which do "
            "their time if will way about many then them would write like so these "
            "her long make thing see him two has look more day go did come number "
            "sound no most people my over know water than call first who may down "
            "side been now find".split()
        )
        out = bytearray()
        while len(out) < n:
            w = rng.choice(words)
            out.extend(w.encode())
            out.append(32)
        return bytes(out[:n])
    if kind == "single_repeat":
        unit = b"lattice compression benchmark test data "
        return (unit * (n // len(unit) + 1))[:n]
    raise ValueError(f"unknown corpus {kind}")


def run_corpus(corpus: str, data: bytes, key: PersonalKey, *, skip_lst1: bool) -> list[BenchRow]:
    rows: list[BenchRow] = []
    rows.append(
        BenchRow(
            name="raw",
            corpus=corpus,
            raw_bytes=len(data),
            compressed_bytes=len(data),
            bare_lumber_bytes=None,
            ratio=1.0,
            encode_ms=0.0,
            decode_ms=0.0,
            roundtrip_ok=True,
        )
    )
    rows.append(_bench_std("zlib-9", corpus, data, lambda d: zlib.compress(d, 9), zlib.decompress))
    rows.append(
        _bench_std(
            "gzip-9",
            corpus,
            data,
            lambda d: gzip.compress(d, compresslevel=9),
            gzip.decompress,
        )
    )
    rows.append(_bench_std("bz2-9", corpus, data, lambda d: bz2.compress(d, 9), bz2.decompress))
    rows.append(_bench_std("lzma-9", corpus, data, lambda d: lzma.compress(d, preset=9), lzma.decompress))

    try:
        import zstandard as zstd

        rows.append(
            _bench_std(
                "zstd-9",
                corpus,
                data,
                lambda d, _z=zstd: _z.ZstdCompressor(level=9).compress(d),
                lambda c, _z=zstd: _z.ZstdDecompressor().decompress(c),
            )
        )
    except ImportError:
        pass
    try:
        import lz4.frame as lz4f

        rows.append(
            _bench_std(
                "lz4-9",
                corpus,
                data,
                lambda d, _l=lz4f: _l.compress(d, compression_level=9),
                lz4f.decompress,
            )
        )
    except ImportError:
        pass

    rows.append(_bench_lattice_dot(corpus, data))
    rows.append(_bench_lattice_lumber_only(corpus, data))
    if not skip_lst1:
        rows.append(_bench_lattice_lst1(corpus, data))
    rows.append(_bench_lattice_personal(corpus, data, key))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Lattice vs standard compression benchmarks")
    p.add_argument("--sample-kb", type=int, default=256, help="Sample size KB (default 256)")
    p.add_argument("--alphabet", type=int, default=200)
    p.add_argument(
        "--corpus",
        default="repetitive,random_bounded,random_byte,english_like,single_repeat",
        help="Comma-separated corpus kinds",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--key", default="bench-personal-key")
    p.add_argument("--skip-lst1", action="store_true", help="Skip slow pattern codec")
    args = p.parse_args()

    n = args.sample_kb * 1024
    key = PersonalKey.from_passphrase(args.key)
    kinds = [k.strip() for k in args.corpus.split(",") if k.strip()]

    all_rows: list[dict] = []
    summary: list[dict] = []

    for kind in kinds:
        data = generate_corpus(kind, n, args.alphabet, args.seed)
        rows = run_corpus(kind, data, key, skip_lst1=args.skip_lst1)
        all_rows.extend(r.as_dict() for r in rows)
        std_rows = [r for r in rows if r.name in ("zlib-9", "gzip-9", "bz2-9", "lzma-9", "zstd-9", "lz4-9")]
        best_classic = max(std_rows, key=lambda r: r.ratio) if std_rows else None
        dot = next(r for r in rows if r.name == "lattice_dot_blob")
        lumber = next(r for r in rows if r.name == "lattice_bare_lumber_only")
        lst1 = next((r for r in rows if r.name == "lattice_lst1_patterns"), None)
        summary.append(
            {
                "corpus": kind,
                "raw_bytes": len(data),
                "n_unique_symbols": len(set(data)),
                "best_classic": best_classic.name if best_classic else None,
                "best_classic_ratio_x": round(best_classic.ratio, 3) if best_classic else None,
                "lattice_dot_blob_ratio_x": round(dot.ratio, 3),
                "lattice_lst1_ratio_x": round(lst1.ratio, 3) if lst1 else None,
                "bare_lumber_ratio_x": round(lumber.ratio, 3),
                "lattice_beats_best_classic": dot.ratio > (best_classic.ratio if best_classic else 0),
            }
        )

    report = {
        "sample_kb": args.sample_kb,
        "alphabet": args.alphabet,
        "corpora": kinds,
        "summary": summary,
        "rows": all_rows,
        "notes": [
            "bare_lumber_only is structure index (not lossless alone)",
            "lattice_dot_blob and lattice_personal_keyed are lossless roundtrip",
            "zstd/lz4 rows omitted if packages not installed",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
