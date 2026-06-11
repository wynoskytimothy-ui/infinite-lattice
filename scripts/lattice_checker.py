#!/usr/bin/env python3
"""
lattice_checker.py - checker learner using RecursiveLattice for pattern memory.

Replaces the flat WinPattern dict from recursive_checker.py with a real
recursive hierarchy. Patterns promote across multiple levels:

  L0: 32 base square primes (registered at init)
  L1: winning chains promoted from PROMOTION_POOL  (count >= PROMOTE_L1)
  L2: combinations of L1 patterns that recur       (count >= PROMOTE_L2)
  L3: combinations of L2 patterns                  (count >= PROMOTE_L3)

Same wing_transform / swap_meet / triple_equalization at every level.
walk_up(prime) finds containing higher-level patterns. walk_down(prime)
recovers the base chain.

Test: same 700 self-play episodes as the bucket-Q and flat-pattern baselines.
"""

from __future__ import annotations

import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_games import _CHK_PRIME_TO_POS, _CHK_SQUARE_PRIMES, CheckersGame
from aethos_recursive_lattice import RecursiveLattice


# =====================================================================
# Loss anomaly (same shape as recursive_checker.py)
# =====================================================================

@dataclass
class LossAnomaly:
    losing_chain: tuple[int, ...]
    winning_response: tuple[int, ...]
    context: frozenset[int]
    losing_side: str
    fire_count: int = 0
    signal: float = 0.0
    resolved: bool = False
    resolving_prime: int | None = None


# =====================================================================
# Agent
# =====================================================================

class LatticeCheckerAgent:
    PROMOTE_L1 = 2       # chain seen 2+ times -> level-1 promotion
    PROMOTE_L2 = 2       # L1 combo seen 2+ times -> level-2
    PROMOTE_L3 = 2       # L2 combo seen 2+ times -> level-3
    ANOMALY_THRESHOLD = 1.0
    PATTERN_MIN_SHARED = 2
    ANOMALY_MIN_SHARED = 3
    PATTERN_LAST_K = 5   # last K of winner becomes pattern key (was 8)
    LAST_K_MOVES = 5

    def __init__(self, alpha=0.3, gamma=0.95, eps=0.25):
        self.game = CheckersGame()
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        # The recursive lattice IS the pattern store
        self.lattice = RecursiveLattice()
        for prime in _CHK_SQUARE_PRIMES:
            self.lattice.register_base(prime, label=f"sq{_CHK_PRIME_TO_POS[prime]}")

        # Promotion-gating counters
        self.chain_counts: Counter = Counter()
        self.chain_promoted: dict[tuple, int] = {}  # (chain, side) -> L1 prime
        self.l1_combo_counts: Counter = Counter()
        self.l1_combo_promoted: dict[tuple, int] = {}
        self.l2_combo_counts: Counter = Counter()
        self.l2_combo_promoted: dict[tuple, int] = {}

        # Anomaly store
        self.anomalies: dict[tuple, LossAnomaly] = {}

        self.stats = Counter()

    # ---- Q helpers ----
    def _state_feats(self, state):
        return self.game.abstract_state(state)

    def _move_sig(self, state, move):
        return self.game.move_signature(state, move)

    def _value(self, state, move):
        return self.Q[(self._state_feats(state), self._move_sig(state, move))]

    def _candidates(self, state):
        moves = self.game.legal_moves(state)
        if not moves: return moves
        max_c = max(len(m.captures) for m in moves)
        cands = [m for m in moves if len(m.captures) == max_c]
        prom = [m for m in cands if m.promoted]
        return prom if prom else cands

    # ---- Pattern matching using the lattice ----
    def _matching_l1_patterns(self, side_chain: tuple[int, ...], side: str):
        my_set = set(side_chain)
        out = []
        for chain_key, prime in self.chain_promoted.items():
            chain, chain_side = chain_key
            if chain_side != side: continue
            shared = len(my_set & set(chain))
            if shared >= self.PATTERN_MIN_SHARED:
                out.append((shared, prime, chain))
        out.sort(key=lambda x: -x[0])
        return out

    def _deepest_containing_pattern(self, prime: int) -> int:
        """Walk up; return the prime at the deepest level reachable."""
        up = self.lattice.walk_up(prime)
        if not up:
            return prime
        # pick the one with highest level
        deepest = prime
        deepest_lvl = self.lattice.resolve(prime).level
        for p in up:
            lvl = self.lattice.resolve(p).level
            if lvl > deepest_lvl:
                deepest = p
                deepest_lvl = lvl
        return deepest

    def _continuations(self, prime: int, used_primes: set[int]) -> list[int]:
        """walk_down to base, return primes not yet used in current play."""
        base_chain = self.lattice.walk_down(prime)
        return [p for p in base_chain if p not in used_primes]

    def _is_anomalous_extension(self, side_chain, next_prime, side):
        ext = set(side_chain) | {next_prime}
        for a in self.anomalies.values():
            if a.resolved or a.signal < self.ANOMALY_THRESHOLD: continue
            if a.losing_side != side: continue
            if len(set(a.losing_chain) & ext) >= self.ANOMALY_MIN_SHARED:
                return True
        return False

    # ---- Action selection ----
    def act(self, state, training: bool, side_chain: tuple[int, ...]):
        cands = self._candidates(state)
        if not cands: return None
        side = state.turn

        # 1. Try hierarchical pattern match
        if not (training and random.random() < self.eps):
            matched = self._matching_l1_patterns(side_chain, side)
            if matched:
                _, l1_prime, _ = matched[0]
                # Walk up to the deepest containing pattern (L2/L3 if available)
                deepest = self._deepest_containing_pattern(l1_prime)
                deepest_level = self.lattice.resolve(deepest).level
                # Continuations from the deepest pattern's full base chain
                used = set(side_chain)
                continuations = self._continuations(deepest, used)
                for next_p in continuations:
                    for m in cands:
                        if _CHK_SQUARE_PRIMES[m.source] == next_p:
                            self.stats["pattern_use"] += 1
                            if deepest_level > 1:
                                self.stats["hierarchy_use"] += 1
                            return m

        # 2. Anomaly avoidance (side-specific)
        if self.anomalies:
            before = len(cands)
            safe = [m for m in cands
                    if not self._is_anomalous_extension(side_chain, _CHK_SQUARE_PRIMES[m.source], side)]
            if safe:
                if len(safe) < before:
                    self.stats["anomaly_avoided"] += before - len(safe)
                cands = safe

        # 3. eps-greedy Q
        if training and random.random() < self.eps:
            return random.choice(cands)
        return max(cands, key=lambda m: (
            self._value(state, m),
            len(m.captures),
            int(m.promoted),
        ))

    # ---- Q update ----
    def _update_q(self, s, a, r, s_next, terminal):
        key = (self._state_feats(s), self._move_sig(s, a))
        old = self.Q[key]
        if terminal:
            target = r
        else:
            nxt = self._candidates(s_next)
            if not nxt:
                target = r
            else:
                opp_best = max(self._value(s_next, m) for m in nxt)
                target = r - self.gamma * opp_best
        self.Q[key] = old + self.alpha * (target - old)

    # ---- Multi-level promotion ----
    def _record_chain(self, chain: tuple[int, ...], side: str):
        chain_key = (chain, side)
        self.chain_counts[chain_key] += 1
        if (self.chain_counts[chain_key] >= self.PROMOTE_L1
                and chain_key not in self.chain_promoted):
            try:
                new_prime = self.lattice.promote(
                    chain, label=f"L1_{side}_n{len(self.chain_promoted)}"
                )
                self.chain_promoted[chain_key] = new_prime
                self.stats["promoted_L1"] += 1
                resolved = self._try_resolve(new_prime, frozenset(chain))
                self.stats["resolved"] += resolved
                return new_prime
            except RuntimeError:
                self.stats["pool_exhausted"] += 1
        return None

    def _record_l1_combo(self, l1_primes: list[int]):
        if len(l1_primes) < 2: return None
        combo = tuple(sorted(set(l1_primes)))
        if len(combo) < 2: return None
        self.l1_combo_counts[combo] += 1
        if (self.l1_combo_counts[combo] >= self.PROMOTE_L2
                and combo not in self.l1_combo_promoted):
            try:
                new_prime = self.lattice.promote(
                    combo, label=f"L2_combo_n{len(self.l1_combo_promoted)}"
                )
                self.l1_combo_promoted[combo] = new_prime
                self.stats["promoted_L2"] += 1
                self._try_resolve(new_prime, frozenset(combo))
                return new_prime
            except RuntimeError:
                self.stats["pool_exhausted"] += 1
        return None

    def _record_l2_combo(self, l2_primes: list[int]):
        if len(l2_primes) < 2: return None
        combo = tuple(sorted(set(l2_primes)))
        if len(combo) < 2: return None
        self.l2_combo_counts[combo] += 1
        if (self.l2_combo_counts[combo] >= self.PROMOTE_L3
                and combo not in self.l2_combo_promoted):
            try:
                new_prime = self.lattice.promote(
                    combo, label=f"L3_combo_n{len(self.l2_combo_promoted)}"
                )
                self.l2_combo_promoted[combo] = new_prime
                self.stats["promoted_L3"] += 1
                return new_prime
            except RuntimeError:
                self.stats["pool_exhausted"] += 1
        return None

    def _record_loss(self, losing_chain, winning_response, losing_side):
        ctx = frozenset(losing_chain) | frozenset(winning_response)
        key = (losing_chain, winning_response, losing_side)
        if key in self.anomalies:
            a = self.anomalies[key]
            a.fire_count += 1
            a.context = frozenset(a.context | ctx)
        else:
            a = LossAnomaly(
                losing_chain=losing_chain,
                winning_response=winning_response,
                context=ctx,
                losing_side=losing_side,
                fire_count=1,
            )
            self.anomalies[key] = a
        ctx_mass = sum(math.log1p(float(p)) for p in a.context)
        a.signal += 0.35 * math.log1p(a.fire_count) + 0.05 * ctx_mass

    def _try_resolve(self, new_prime, new_chain: frozenset[int]) -> int:
        n = 0
        ctx = new_chain | {new_prime}
        for a in self.anomalies.values():
            if a.resolved: continue
            if len(a.context & ctx) >= 2:
                a.resolved = True
                a.resolving_prime = new_prime
                n += 1
        return n

    # ---- Learn from a finished game ----
    def learn_from_game(self, transitions, winner, chains):
        if winner in ("X", "O"):
            loser = "O" if winner == "X" else "X"
            # Dedupe: a chain in the lattice is a set of distinct prime anchors
            winner_last = tuple(sorted(set(chains[winner][-self.PATTERN_LAST_K:])))
            loser_last = tuple(sorted(set(chains[loser][-self.LAST_K_MOVES:])))

            # Promote winner's last-K chain (L1)
            new_l1 = self._record_chain(winner_last, winner)

            # Find all L1 patterns that share enough primes with this game's winner chain
            winner_set = set(chains[winner])
            l1_in_game: list[int] = []
            for chain_key, prime in self.chain_promoted.items():
                chain, side = chain_key
                if side != winner: continue
                if len(set(chain) & winner_set) >= self.PATTERN_MIN_SHARED:
                    l1_in_game.append(prime)

            # If multiple L1s in this game, record their combo (maybe promote L2)
            new_l2 = None
            if len(l1_in_game) >= 2:
                new_l2 = self._record_l1_combo(l1_in_game)

            # If multiple L2s have ever been promoted, check combos
            if new_l2 is not None and len(self.l1_combo_promoted) >= 2:
                # try L3 promotion from L2 combos
                l2_combo = list(self.l1_combo_promoted.values())[-3:]
                if new_l2 in l2_combo:
                    self._record_l2_combo(l2_combo)

            self._record_loss(loser_last, winner_last, losing_side=loser)

        # Q-table updates
        for side, s, a, s_next in transitions:
            w_next = self.game.winner(s_next)
            terminal = w_next is not None
            r = 0.0
            if terminal:
                if w_next == side: r = 1.0
                elif w_next is not None and w_next != "draw": r = -1.0
            self._update_q(s, a, r, s_next, terminal)


# =====================================================================
# Episode / eval
# =====================================================================

MAX_PLIES = 200


def play_episode(agent: LatticeCheckerAgent, training: bool = True) -> str:
    state = agent.game.initial_state()
    transitions = []
    chains: dict[str, list[int]] = {"X": [], "O": []}
    plies = 0
    winner: str | None = None
    while plies < MAX_PLIES:
        winner = agent.game.winner(state)
        if winner is not None: break
        side = state.turn
        m = agent.act(state, training, tuple(chains[side]))
        if m is None:
            winner = "O" if side == "X" else "X"
            break
        s_next = agent.game.apply_move(state, m)
        transitions.append((side, state, m, s_next))
        chains[side].append(_CHK_SQUARE_PRIMES[m.source])
        state = s_next
        plies += 1
    if winner is None: winner = "draw"
    if training:
        agent.learn_from_game(transitions, winner, chains)
    return winner


def random_policy(game, state):
    moves = game.legal_moves(state)
    return random.choice(moves) if moves else None


def eval_vs_random(agent: LatticeCheckerAgent, games: int, side: str) -> dict:
    w = d = l = 0
    for _ in range(games):
        state = agent.game.initial_state()
        chains: dict[str, list[int]] = {"X": [], "O": []}
        plies = 0
        winner: str | None = None
        while plies < MAX_PLIES:
            winner = agent.game.winner(state)
            if winner is not None: break
            cur = state.turn
            if cur == side:
                m = agent.act(state, training=False, side_chain=tuple(chains[cur]))
            else:
                m = random_policy(agent.game, state)
            if m is None: break
            chains[cur].append(_CHK_SQUARE_PRIMES[m.source])
            state = agent.game.apply_move(state, m)
            plies += 1
        if winner is None: winner = "draw"
        if winner == side: w += 1
        elif winner == "draw": d += 1
        else: l += 1
    return {"w": w, "d": d, "l": l}


# =====================================================================
# Main
# =====================================================================

def main():
    random.seed(42)
    print("=" * 78)
    print("LATTICE CHECKER LEARNER - RecursiveLattice as pattern memory")
    print("=" * 78)
    print()
    print("Pattern hierarchy:")
    print("  L0: 32 base square primes (registered at init)")
    print("  L1: winning chains promoted from PROMOTION_POOL")
    print("  L2: combinations of L1 patterns that recur")
    print("  L3: combinations of L2 patterns")
    print()
    print("walk_up(prime) finds containing higher-level patterns at decision time.")
    print("=" * 78)

    agent = LatticeCheckerAgent(alpha=0.3, gamma=0.95, eps=0.25)
    checkpoints = [50, 200, 400, 700]
    t0 = time.time()
    for i in range(1, max(checkpoints) + 1):
        play_episode(agent, training=True)
        if i % 100 == 0:
            agent.eps = max(0.05, 0.25 * (1.0 - i / max(checkpoints)))
        if i in checkpoints:
            vX = eval_vs_random(agent, 40, "X")
            vO = eval_vs_random(agent, 40, "O")
            dt = time.time() - t0
            s = agent.lattice.stats()
            lc = s["level_counts"]
            print(f"  ep={i:>4} ({dt:>5.1f}s)  |Q|={len(agent.Q):>5}  "
                  f"L1={agent.stats['promoted_L1']:>2} "
                  f"L2={agent.stats['promoted_L2']:>2} "
                  f"L3={agent.stats['promoted_L3']:>2}   "
                  f"vs random X w/d/l={vX['w']:>2}/{vX['d']}/{vX['l']:>2}   "
                  f"O w/d/l={vO['w']:>2}/{vO['d']}/{vO['l']:>2}")
            print(f"           lattice levels: {lc}   "
                  f"pattern_uses={agent.stats['pattern_use']}  "
                  f"hierarchy_uses={agent.stats['hierarchy_use']}  "
                  f"anomalies_resolved={agent.stats['resolved']}")

    print()
    print("=" * 78)
    print("FINAL EVAL (100 games each side)")
    print("=" * 78)
    agent.eps = 0.0
    fX = eval_vs_random(agent, 100, "X")
    fO = eval_vs_random(agent, 100, "O")
    print(f"  as X: {fX['w']:>3}/100 ({fX['w']}%)   "
          f"[recursive_checker: 56%, v1 raw-Q: 67%]")
    print(f"  as O: {fO['w']:>3}/100 ({fO['w']}%)   "
          f"[recursive_checker: 82%, v1 raw-Q: 72%]")

    print()
    print("=" * 78)
    print("RECURSIVE LATTICE STATE")
    print("=" * 78)
    s = agent.lattice.stats()
    print(f"  total_nodes     : {s['total_nodes']}")
    print(f"  level_counts    : {s['level_counts']}")
    print(f"  max_level       : {s['max_level']}")
    print(f"  pool_used       : {s['pool_used']} / {s['pool_used'] + s['pool_remaining']}")
    print(f"  loss anomalies  : {len(agent.anomalies)}")
    print(f"  resolved        : {agent.stats['resolved']}")

    # Show one tree if L2+ exists
    by_level: dict[int, list] = defaultdict(list)
    for node in agent.lattice.nodes.values():
        if node.is_promoted:
            by_level[node.level].append(node)
    for lvl in sorted(by_level.keys(), reverse=True):
        if by_level[lvl]:
            top = by_level[lvl][0]
            print(f"\n--- Top tree (root level {lvl}, prime {top.prime}) ---")
            tree = agent.lattice.render_tree(top.prime).split("\n")
            for line in tree[:18]:
                print(f"  {line}")
            if len(tree) > 18:
                print(f"  ... ({len(tree) - 18} more lines)")
            break


if __name__ == "__main__":
    main()
