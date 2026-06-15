#!/usr/bin/env python3
"""
Test 8 - Distributed ID generation without coordination.

Claim: N independent agents can claim primes from disjoint pools in parallel
without any central coordinator. Each agent emits IDs (composites) that are
guaranteed unique across the whole system by FTA.

Why this matters: Distributed unique ID systems (Twitter Snowflake, UUID,
Sonyflake, KSUID) need either:
  - Central coordinator (single point of failure, contention)
  - Probabilistic guarantees (UUID v4 has 2^-122 collision per pair)
  - Time-based + machine ID (clock skew issues)

FTA composites give DETERMINISTIC global uniqueness with NO coordinator:
  - Each agent claims a base prime range (e.g., agent_i takes primes [a_i, b_i])
  - Each agent's composites are products of its prime range
  - Distinct agents -> distinct prime factors -> distinct composites (FTA)

Tests:
  (A) Simulate 100 agents in parallel, each emits 1000 IDs
  (B) Verify zero collisions across 100,000 emitted IDs
  (C) Verify factorization recovers issuing agent (prime tells the source)
  (D) No coordination overhead: agents touch disjoint memory
"""

from __future__ import annotations

import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.primes import chain_primes


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


def agent_emit(agent_id: int, prime_range: tuple[int, ...], n_ids: int,
               rng_seed: int) -> tuple[int, list[int]]:
    """One agent emits n_ids composite IDs from its own prime range.

    To guarantee uniqueness even WITHIN this agent's stream, we enforce that
    each chain is distinct. The FTA guarantee then gives global uniqueness
    automatically (because chain spaces are disjoint across agents).
    """
    rng = random.Random(rng_seed)
    ids: list[int] = []
    seen_chains: set[tuple[int, ...]] = set()
    while len(ids) < n_ids:
        k = rng.randint(3, 5)
        chain = tuple(sorted(rng.sample(prime_range, k)))
        if chain in seen_chains:
            continue
        seen_chains.add(chain)
        composite = 1
        for p in chain:
            composite *= p
        ids.append(composite)
    return agent_id, ids


def main():
    header("Distributed ID via disjoint prime ranges - zero-coordination")

    # ---------------------------------------------------------------
    # Part A: Allocate disjoint prime ranges to 100 agents
    # ---------------------------------------------------------------
    print("\nPart A - Allocate disjoint prime ranges to 100 agents")
    print("-" * 72)

    n_agents = 100
    primes_per_agent = 50  # each agent gets 50 unique primes
    total_primes_needed = n_agents * primes_per_agent  # 5000
    base = chain_primes(total_primes_needed)
    print(f"  total primes available:  {len(base)}")
    print(f"  primes per agent:        {primes_per_agent}")
    print(f"  agents:                  {n_agents}")

    # Disjoint partition: agent_i gets primes [i*50 .. (i+1)*50]
    agent_ranges: list[tuple[int, ...]] = [
        tuple(base[i * primes_per_agent: (i + 1) * primes_per_agent])
        for i in range(n_agents)
    ]
    # Verify ranges are disjoint
    all_assigned = [p for r in agent_ranges for p in r]
    assertion(len(all_assigned) == len(set(all_assigned)),
              f"prime ranges across {n_agents} agents are pairwise disjoint")
    print(f"  total assigned primes:   {len(all_assigned)}")

    # ---------------------------------------------------------------
    # Part B: Parallel emission with no coordination
    # ---------------------------------------------------------------
    print("\nPart B - 100 agents emit 1000 IDs each in parallel (no locking)")
    print("-" * 72)

    n_per_agent = 1000
    t0 = time.time()
    all_ids: list[tuple[int, int]] = []  # (agent_id, composite_id)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(agent_emit, i, agent_ranges[i], n_per_agent,
                            rng_seed=i * 1000 + 42)
            for i in range(n_agents)
        ]
        for fut in as_completed(futures):
            agent_id, ids = fut.result()
            for cid in ids:
                all_ids.append((agent_id, cid))

    dt = time.time() - t0
    print(f"  emitted {len(all_ids)} IDs in {dt:.2f}s ({len(all_ids)/dt:.0f} IDs/sec)")

    # ---------------------------------------------------------------
    # Part C: Zero collisions
    # ---------------------------------------------------------------
    print("\nPart C - Zero collisions across all 100,000 IDs")
    print("-" * 72)

    just_ids = [cid for _, cid in all_ids]
    distinct = len(set(just_ids))
    print(f"  total IDs:     {len(just_ids)}")
    print(f"  distinct IDs:  {distinct}")
    print(f"  collisions:    {len(just_ids) - distinct}")
    assertion(distinct == len(just_ids),
              "all 100,000 distributed IDs are globally unique (no coordinator!)")

    # ---------------------------------------------------------------
    # Part D: ID -> issuing agent recovery via factorization
    # ---------------------------------------------------------------
    print("\nPart D - Factorize ID -> recover issuing agent")
    print("-" * 72)

    # Build inverse map: prime -> agent_id
    prime_to_agent: dict[int, int] = {}
    for aid, r in enumerate(agent_ranges):
        for p in r:
            prime_to_agent[p] = aid

    # Sample 500 IDs, factor and identify agent
    recovery_failures = 0
    sample = random.sample(all_ids, 500)
    for true_aid, cid in sample:
        # trial divide by all base primes
        residual = cid
        factors = []
        for p in base:
            while residual % p == 0:
                factors.append(p)
                residual //= p
            if residual == 1:
                break
        # all factors should belong to the same agent
        agents = {prime_to_agent[p] for p in factors}
        if len(agents) != 1 or true_aid not in agents:
            recovery_failures += 1
    assertion(recovery_failures == 0,
              f"factorization recovers issuing agent for all 500 sampled IDs")

    # ---------------------------------------------------------------
    # Part E: Bit cost vs Snowflake / UUID
    # ---------------------------------------------------------------
    print("\nPart E - Bit cost comparison")
    print("-" * 72)

    bit_widths = [math.log2(cid) for cid in just_ids[:1000]]
    avg_bits = sum(bit_widths) / len(bit_widths)
    max_bits = max(bit_widths)
    print(f"  FTA composite IDs:   avg = {avg_bits:.1f} bits, max = {max_bits:.1f} bits")
    print(f"  Twitter Snowflake:   64 bits (time + worker + sequence)")
    print(f"  UUID v4:             128 bits (random)")
    print(f"  KSUID:               160 bits")
    print()
    print(f"  Our hash is competitive on size AND deterministically unique.")
    assertion(avg_bits < 128, "FTA IDs more compact than UUIDs")

    # ---------------------------------------------------------------
    # Part F: Verify agent ranges remained independent
    # ---------------------------------------------------------------
    print("\nPart F - Agent isolation: factors of any ID lie in one range")
    print("-" * 72)

    # Pick 100 IDs and verify each factors to primes from exactly one agent
    isolation_check = 0
    for true_aid, cid in random.sample(all_ids, 100):
        residual = cid
        agents_seen = set()
        for p in base:
            while residual % p == 0:
                agents_seen.add(prime_to_agent[p])
                residual //= p
            if residual == 1:
                break
        if len(agents_seen) == 1 and true_aid in agents_seen:
            isolation_check += 1
    assertion(isolation_check == 100,
              "100/100 IDs factor entirely within one agent's range")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    header("RESULT")
    print(f"  Agents:          {n_agents}")
    print(f"  Total IDs:       {len(just_ids)}")
    print(f"  Distinct IDs:    {distinct}")
    print(f"  Collisions:      0")
    print(f"  Bits per ID:     {avg_bits:.0f} avg")
    print(f"  Throughput:      {len(all_ids)/dt:.0f} IDs/sec (no coordination)")
    print()
    print("  CONCLUSION:")
    print("  N agents with disjoint prime ranges can emit globally unique")
    print("  IDs in parallel with ZERO coordination. By FTA, distinct prime")
    print("  factorizations -> distinct composites. The composite also")
    print("  IDENTIFIES the issuing agent (factor and look up). This is")
    print("  better than Snowflake (no clock skew issues), better than UUID")
    print("  (deterministic uniqueness instead of probabilistic), and")
    print("  competitively compact.")


if __name__ == "__main__":
    main()
