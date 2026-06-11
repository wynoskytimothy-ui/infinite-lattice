#!/usr/bin/env python3
"""
recursive_checker.py - checker learner with the 4-mechanism recursive loop.

Implements your statement:
  1. Bilateral learning - both winner and loser chains carry signal
  2. Anomaly priming   - losing chains accumulate signal (loss gets a prime later)
  3. Pattern promotion - recurring win-chains get new primes from PROMOTION_POOL
  4. Sub-lattice spawn - each promoted prime hosts its own continuation index

Architecture modeled on:
  aethos_promotion.PromotionRegistry  (pool allocation + frequency gating)
  core.learning_engine.BadCorrelationStore  (signal accumulation + resolution)

Baseline to beat: v1 raw-state Q hit 67% vs random as X over 700 episodes.
"""

from __future__ import annotations

import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

from aethos_games import _CHK_SQUARE_PRIMES, CheckersGame
from aethos_promotion import PROMOTION_POOL


# =====================================================================
# Pattern memory (analog of PromotionRegistry, specialized for chains)
# =====================================================================

@dataclass
class WinPattern:
    chain: tuple[int, ...]
    side: str
    count: int = 1
    promoted_prime: int | None = None
    # sub-lattice: continuations Counter + metadata
    sub_lattice: dict = field(default_factory=dict)


# =====================================================================
# Loss anomaly (analog of BadCorrelationStore entry)
# =====================================================================

@dataclass
class LossAnomaly:
    losing_chain: tuple[int, ...]
    winning_response: tuple[int, ...]
    context: frozenset[int]
    losing_side: str  # which side made this losing chain
    fire_count: int = 0
    signal: float = 0.0
    resolved: bool = False
    resolving_prime: int | None = None


# =====================================================================
# Recursive agent
# =====================================================================

class RecursiveCheckerAgent:
    PROMOTE_AT = 3  # chain win-count threshold for promotion
    ANOMALY_THRESHOLD = 1.0  # signal threshold to actively avoid
    PATTERN_MIN_SHARED = 3  # primes shared for pattern match
    ANOMALY_MIN_SHARED = 3  # primes shared for anomaly trigger
    LAST_K_MOVES = 5  # only the last K moves of a loser become an anomaly key
    PATTERN_LAST_K = 8  # only the last K winning moves promote as pattern

    def __init__(self, alpha: float = 0.3, gamma: float = 0.95, eps: float = 0.25):
        self.game = CheckersGame()
        self.Q: dict = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        # Pattern store (keyed by (chain, side))
        self.patterns: dict[tuple, WinPattern] = {}
        self.next_promo_idx = 0
        # Anomaly store (keyed by (losing_chain, winning_response))
        self.anomalies: dict[tuple, LossAnomaly] = {}
        self.stats = Counter()

    # ---- Q-table helpers ----
    def _state_feats(self, state):
        return self.game.abstract_state(state)

    def _move_sig(self, state, move):
        return self.game.move_signature(state, move)

    def _value(self, state, move):
        return self.Q[(self._state_feats(state), self._move_sig(state, move))]

    def _candidates(self, state):
        moves = self.game.legal_moves(state)
        if not moves:
            return moves
        max_c = max(len(m.captures) for m in moves)
        cands = [m for m in moves if len(m.captures) == max_c]
        prom = [m for m in cands if m.promoted]
        return prom if prom else cands

    # ---- Pattern matching (use the sub-lattice continuations) ----
    def _matching_patterns(self, side_chain: tuple[int, ...], side: str) -> list[WinPattern]:
        my_set = set(side_chain)
        scored: list[tuple[int, WinPattern]] = []
        for wp in self.patterns.values():
            if wp.promoted_prime is None:
                continue
            if wp.side != side:
                continue
            shared = len(my_set & set(wp.chain))
            if shared >= self.PATTERN_MIN_SHARED:
                scored.append((shared, wp))
        scored.sort(key=lambda x: (-x[0], -x[1].count))
        return [wp for _, wp in scored]

    # ---- Anomaly check (side-specific: only filter by our own past losses) ----
    def _is_anomalous_extension(self, side_chain: tuple[int, ...], next_prime: int, side: str) -> bool:
        ext = set(side_chain) | {next_prime}
        for a in self.anomalies.values():
            if a.resolved or a.signal < self.ANOMALY_THRESHOLD:
                continue
            if a.losing_side != side:
                continue
            if len(set(a.losing_chain) & ext) >= self.ANOMALY_MIN_SHARED:
                return True
        return False

    # ---- Action selection ----
    def act(self, state, training: bool, side_chain: tuple[int, ...]):
        cands = self._candidates(state)
        if not cands:
            return None
        side = state.turn

        # 1. Pattern hit -> use sub-lattice continuations
        if not (training and random.random() < self.eps):
            matched = self._matching_patterns(side_chain, side)
            if matched:
                best = matched[0]
                conts = best.sub_lattice.get("continuations", Counter())
                if conts:
                    for next_p, _ in conts.most_common(5):
                        for m in cands:
                            if _CHK_SQUARE_PRIMES[m.source] == next_p:
                                self.stats["pattern_use"] += 1
                                return m

        # 2. Anomaly avoidance (side-specific - only filter by our own losses)
        if self.anomalies:
            before = len(cands)
            safe = [
                m for m in cands
                if not self._is_anomalous_extension(side_chain, _CHK_SQUARE_PRIMES[m.source], side)
            ]
            if safe:
                if len(safe) < before:
                    self.stats["anomaly_avoided"] += before - len(safe)
                cands = safe

        # 3. Eps-greedy Q with structural prior tiebreak
        if training and random.random() < self.eps:
            return random.choice(cands)
        return max(cands, key=lambda m: (
            self._value(state, m),
            len(m.captures),
            int(m.promoted),
        ))

    # ---- Q update ----
    def _update_q(self, s, a, r, s_next, terminal: bool):
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

    # ---- Pattern record + promotion ----
    def _record_pattern(self, chain: tuple[int, ...], side: str):
        key = (chain, side)
        if key in self.patterns:
            wp = self.patterns[key]
            wp.count += 1
        else:
            wp = WinPattern(chain=chain, side=side, count=1)
            self.patterns[key] = wp
        # populate sub-lattice continuations (the chain itself is the win sequence)
        sl = wp.sub_lattice
        sl.setdefault("continuations", Counter())
        for p in chain:
            sl["continuations"][p] += 1

        # Maybe promote
        if wp.promoted_prime is None and wp.count >= self.PROMOTE_AT:
            if self.next_promo_idx < len(PROMOTION_POOL):
                wp.promoted_prime = PROMOTION_POOL[self.next_promo_idx]
                self.next_promo_idx += 1
                sl["promoted_at_count"] = wp.count
                self.stats["promoted"] += 1
                # Resolve old anomalies sharing context with this new prime's chain
                resolved = self._try_resolve(wp.promoted_prime, frozenset(wp.chain))
                if resolved:
                    self.stats["resolved"] += resolved

    # ---- Anomaly record ----
    def _record_loss(self, losing_chain: tuple[int, ...],
                     winning_response: tuple[int, ...],
                     losing_side: str):
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
        # BadCorrelationStore-style signal growth (rescaled for checker primes)
        ctx_mass = sum(math.log1p(float(p)) for p in a.context)
        a.signal += 0.35 * math.log1p(a.fire_count) + 0.05 * ctx_mass

    # ---- Resolve anomalies when a new prime explains them ----
    def _try_resolve(self, new_prime: int, new_chain: frozenset[int]) -> int:
        n = 0
        ctx_with_new = new_chain | {new_prime}
        for a in self.anomalies.values():
            if a.resolved:
                continue
            if len(a.context & ctx_with_new) >= 2:
                a.resolved = True
                a.resolving_prime = new_prime
                n += 1
        return n

    # ---- Learn from a finished game ----
    def learn_from_game(self, transitions, winner: str, chains: dict):
        if winner in ("X", "O"):
            loser = "O" if winner == "X" else "X"
            # Keep only the decisive last-K moves of each side
            winner_last = tuple(sorted(chains[winner][-self.PATTERN_LAST_K:]))
            loser_last = tuple(sorted(chains[loser][-self.LAST_K_MOVES:]))
            self._record_pattern(winner_last, winner)
            self._record_loss(loser_last, winner_last, losing_side=loser)
        for side, s, a, s_next in transitions:
            w_next = self.game.winner(s_next)
            terminal = w_next is not None
            r = 0.0
            if terminal:
                if w_next == side:
                    r = 1.0
                elif w_next is not None and w_next != "draw":
                    r = -1.0
            self._update_q(s, a, r, s_next, terminal)


# =====================================================================
# Episode + eval
# =====================================================================

MAX_PLIES = 200


def play_episode(agent: RecursiveCheckerAgent, training: bool = True) -> str:
    state = agent.game.initial_state()
    transitions = []
    chains: dict[str, list[int]] = {"X": [], "O": []}
    plies = 0
    winner: str | None = None
    while plies < MAX_PLIES:
        winner = agent.game.winner(state)
        if winner is not None:
            break
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
    if winner is None:
        winner = "draw"
    if training:
        agent.learn_from_game(transitions, winner, chains)
    return winner


def random_policy(game, state):
    moves = game.legal_moves(state)
    return random.choice(moves) if moves else None


def eval_vs_random(agent: RecursiveCheckerAgent, games: int, agent_side: str) -> dict:
    w = d = l = 0
    for _ in range(games):
        state = agent.game.initial_state()
        chains: dict[str, list[int]] = {"X": [], "O": []}
        plies = 0
        winner: str | None = None
        while plies < MAX_PLIES:
            winner = agent.game.winner(state)
            if winner is not None:
                break
            side = state.turn
            if side == agent_side:
                m = agent.act(state, training=False, side_chain=tuple(chains[side]))
            else:
                m = random_policy(agent.game, state)
            if m is None:
                break
            chains[side].append(_CHK_SQUARE_PRIMES[m.source])
            state = agent.game.apply_move(state, m)
            plies += 1
        if winner is None:
            winner = "draw"
        if winner == agent_side:
            w += 1
        elif winner == "draw":
            d += 1
        else:
            l += 1
    return {"w": w, "d": d, "l": l}


# =====================================================================
# Main
# =====================================================================

def main():
    random.seed(42)
    print("=" * 72)
    print("RECURSIVE CHECKER LEARNER")
    print("=" * 72)
    print()
    print("Mechanisms:")
    print("  1. Bilateral learning - winner AND loser chains are signal")
    print("  2. Pattern promotion - chains seen >= 4 times -> prime from PROMOTION_POOL")
    print("  3. Loss anomaly store - BadCorrelationStore-style signal accumulation")
    print("  4. Anomaly resolution - new primes collapse matched anomalies")
    print("  5. Sub-lattice - each promoted pattern hosts continuation Counter")
    print()
    print("Baseline: v1 raw-state Q hit 67% vs random as X after 700 episodes.")
    print("=" * 72)
    print()

    agent = RecursiveCheckerAgent(alpha=0.3, gamma=0.95, eps=0.25)
    checkpoints = [50, 200, 400, 700]
    eps_floor = 0.05
    t0 = time.time()
    for i in range(1, max(checkpoints) + 1):
        play_episode(agent, training=True)
        if i % 100 == 0:
            agent.eps = max(eps_floor, 0.25 * (1.0 - i / max(checkpoints)))
        if i in checkpoints:
            vX = eval_vs_random(agent, 40, "X")
            vO = eval_vs_random(agent, 40, "O")
            dt = time.time() - t0
            print(f"  ep={i:>4} ({dt:.1f}s)  |Q|={len(agent.Q):>5}  eps={agent.eps:.2f}   "
                  f"vs random X w/d/l={vX['w']}/{vX['d']}/{vX['l']}   "
                  f"O w/d/l={vO['w']}/{vO['d']}/{vO['l']}")
            print(f"           promoted={agent.stats['promoted']:>2}  "
                  f"anomalies={len(agent.anomalies):>4}  "
                  f"resolved={agent.stats['resolved']:>2}  "
                  f"pattern_uses={agent.stats['pattern_use']:>4}  "
                  f"anomaly_avoided={agent.stats['anomaly_avoided']:>4}")

    print()
    print("=" * 72)
    print("FINAL EVALUATION (100 games each side)")
    print("=" * 72)
    agent.eps = 0.0
    fX = eval_vs_random(agent, 100, "X")
    fO = eval_vs_random(agent, 100, "O")
    print(f"  as X: {fX['w']:>3}/100 wins ({fX['w']}%)   [baseline v1: 67%]")
    print(f"  as O: {fO['w']:>3}/100 wins ({fO['w']}%)")

    print()
    print("=" * 72)
    print("LEARNED STRUCTURE")
    print("=" * 72)
    print(f"  Patterns recorded:                  {len(agent.patterns)}")
    print(f"  Promoted to new primes:             {agent.stats['promoted']}")
    print(f"  Loss anomalies:                     {len(agent.anomalies)}")
    print(f"  Anomalies resolved by new primes:   {agent.stats['resolved']}")
    print(f"  Pattern uses during play:           {agent.stats['pattern_use']}")
    print(f"  Moves avoided as anomalous:         {agent.stats['anomaly_avoided']}")

    promoted = sorted(
        (w for w in agent.patterns.values() if w.promoted_prime is not None),
        key=lambda w: -w.count,
    )
    if promoted:
        print()
        print("--- Top 5 promoted patterns (with sub-lattice continuations) ---")
        for wp in promoted[:5]:
            conts = wp.sub_lattice.get("continuations", Counter())
            top_conts = ", ".join(f"{p}(x{c})" for p, c in conts.most_common(4))
            print(f"  prime={wp.promoted_prime}  side={wp.side}  wins={wp.count}  "
                  f"len(chain)={len(wp.chain)}")
            print(f"           sub-lattice top continuations: {top_conts}")


if __name__ == "__main__":
    main()
