#!/usr/bin/env python3
"""
Root-cause finder on the lattice - a glass-box diagnostic.

Each CAUSE is indexed by its symptom signature. Given observed symptoms, the
engine ranks causes by idf-weighted symptom CONVERGENCE (the meet): the cause
that explains the most symptoms, weighted by how DIAGNOSTIC each is (a rare
symptom pins the diagnosis; a common one is noise - the "rarest means most"
rule). Every diagnosis comes with its evidence chain: which symptoms matched,
which one was decisive, and how confident (the score gap to the runner-up).

Continual learning: a new resolved incident is one appended cause - the system
gets better at diagnosis with every case, no retraining, no forgetting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words

# cause -> its symptom signature (a small knowledge base of failure modes)
CAUSES = {
    "DB connection pool exhausted":
        "requests timeout waiting for a database connection; pool exhausted; max "
        "connections reached; queries queue and hang; latency spikes under load",
    "Memory leak":
        "memory usage grows steadily over hours; out of memory OOM errors; a "
        "restart temporarily fixes it; heap keeps climbing; swap thrashing",
    "Disk full":
        "no space left on device; writes fail; log files grew unbounded; inode "
        "or storage at 100 percent; database cannot write",
    "TLS certificate expired":
        "TLS SSL handshake failed; certificate expired or untrusted; HTTPS "
        "connections refused; clients cannot connect securely",
    "Database deadlock":
        "deadlock detected; lock wait timeout; transactions blocked on each "
        "other; concurrent writes hang; rollback storms",
    "Rate limited by upstream":
        "HTTP 429 too many requests; rate limit or quota exceeded; upstream "
        "throttled our calls; backoff and retries",
    "Cache stampede":
        "thundering herd after cache expiry; simultaneous cache misses; backend "
        "overloaded by a load spike; recompute storm",
    "Clock skew":
        "time drift between hosts; clock skew; authentication tokens rejected as "
        "expired; JWT or kerberos failures; NTP not syncing",
    "Network partition":
        "nodes unreachable; split brain; consensus quorum lost; replication lag "
        "grows; cluster members flapping",
    "Thread pool exhausted":
        "thread pool exhausted; no available worker threads; request queue "
        "backlog; rejected execution; latency under concurrency",
    "Bad deploy / config drift":
        "regression started right after a deployment; a config change broke it; "
        "rollback restores service; new version introduced the fault",
    "GC pause":
        "long garbage collection pauses; stop the world; JVM latency spikes; tail "
        "latency p99 jumps; heap pressure",
    "Cascading failure":
        "a downstream dependency timed out; retries amplified the load; failure "
        "cascaded across services; outage spread",
    "DNS resolution failure":
        "name resolution failed; unknown host; DNS lookup timeouts; intermittent "
        "connectivity to services by hostname",
}


class RootCauseFinder:
    def __init__(self, causes):
        self.idx = AppendOnlyLatticeIndex()
        for name, sig in causes.items():
            self.idx.add(name, sig)
        self.sig = dict(causes)

    def add_incident(self, cause, symptoms):
        """append a newly resolved incident (continual learning, O(1))."""
        if cause in self.idx.alive:
            return
        self.idx.add(cause, symptoms)
        self.sig[cause] = symptoms

    def diagnose(self, symptoms, k=3):
        N = max(1, len(self.idx.alive))
        scores = self.idx._score(symptoms)
        ranked = sorted(scores, key=lambda d: scores[d], reverse=True)[:k]
        qwords = set(words(symptoms))
        out = []
        for cause in ranked:
            matched = sorted(qwords & set(words(self.sig[cause])),
                             key=lambda w: -self._idf(w))
            decisive = matched[0] if matched else None
            out.append((cause, scores[cause], matched, decisive))
        gap = (scores[ranked[0]] - scores[ranked[1]]) if len(ranked) > 1 else scores.get(ranked[0], 0)
        conf = gap / scores[ranked[0]] if ranked and scores[ranked[0]] else 0.0
        return out, conf

    def _idf(self, w):
        p = self.idx.token_prime.get(("w", w))
        return self.idx._idf(p, max(1, len(self.idx.alive))) if p else 0.0


def show(rcf, symptoms):
    print(f"\nOBSERVED: {symptoms}")
    diag, conf = rcf.diagnose(symptoms)
    for i, (cause, sc, matched, decisive) in enumerate(diag):
        tag = "  <= ROOT CAUSE" if i == 0 else ""
        print(f"   {i+1}. {cause:<32} score {sc:5.2f}{tag}")
        print(f"        evidence: {', '.join(matched[:6]) or '-'}")
        if i == 0 and decisive:
            print(f"        decisive symptom: '{decisive}' (idf {rcf._idf(decisive):.1f}, "
                  f"the rarest/most diagnostic)")
    print(f"   confidence: {conf*100:.0f}% (score gap to runner-up)")


def main():
    rcf = RootCauseFinder(CAUSES)
    print(f"root-cause finder: {len(CAUSES)} known failure modes indexed\n" + "=" * 60)

    # 1) a clear case: convergent rare symptoms pin one cause
    show(rcf, "requests timing out, database connections all in use, pool exhausted")

    # 2) ambiguity: a common symptom alone is not diagnostic...
    show(rcf, "latency is high and responses are slow")
    # ...but adding the RARE symptom disambiguates (rarest means most)
    show(rcf, "latency is high, slow responses, long stop the world GC pauses")

    # 3) continual learning: a NEW failure mode the system has never seen
    print("\n" + "=" * 60 + "\nCONTINUAL LEARNING: a novel incident is appended (O(1), no retrain)")
    rcf.add_incident("Kafka consumer lag",
                     "consumer group lag growing; messages piling up in the topic; "
                     "offsets falling behind; rebalancing storms; partition stuck")
    show(rcf, "messages piling up, consumer group lag growing, offsets behind")


if __name__ == "__main__":
    main()
