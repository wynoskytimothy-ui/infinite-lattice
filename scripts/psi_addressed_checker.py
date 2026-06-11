#!/usr/bin/env python3
"""
psi_addressed_checker.py - Q-table keyed on Psi.coord, no buckets.

Tests the claim: with septillions of correlations available via the lattice,
mapping correctly (Psi as address) bounds memory by the number of equalized
nodes, not by trajectory count.

Q-table key:
    state  -> 4-tuple of Psi.coord readouts, one per piece-type chain
              computed via wing_transform from aethos_complex_plane
    action -> the chamber signature (1..32) + capture + promotion

Memory growth measured at multiple checkpoints. Sub-linear |Q| growth over
episodes => lattice equalization is collapsing trajectories as predicted.
Linear |Q| growth => Psi addresses are too unique at this game size.
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_complex_plane import wing_transform
from aethos_games import _CHK_SQUARE_PRIMES, CheckersGame
from aethos_lattice import BranchKind


# =====================================================================
# Psi address computation (cached)
# =====================================================================

@lru_cache(maxsize=2_000_000)
def _primes_of(c: int) -> tuple[int, ...]:
    return tuple(p for p in _CHK_SQUARE_PRIMES if c % p == 0)


@lru_cache(maxsize=2_000_000)
def psi_address(xm: int, xk: int, om: int, ok: int, turn: str, ply: int) -> tuple:
    """Compute the lattice address (z, zeta) per piece-type chain.

    Returns a 4-tuple, one entry per (branch, chain). Each entry is
    (branch_id, X_int, Y_int, zeta_int) where the (X,Y,zeta) is the integer
    Psi.coord from wing_transform on that chain with n=ply, wing=1.

    Empty chains use a (branch_id, 0, 0, 0) sentinel so the slot is preserved.
    Perspective-flip: side-to-move always sees its own pieces on VA1/VA2,
    so symmetric positions across the X/O turn axis collapse to the same key.
    """
    xm_c = _primes_of(xm)
    xk_c = _primes_of(xk)
    om_c = _primes_of(om)
    ok_c = _primes_of(ok)

    if turn == "X":
        slots = ((xm_c, BranchKind.VA1), (xk_c, BranchKind.VA2),
                 (om_c, BranchKind.VA3), (ok_c, BranchKind.VA4))
    else:
        slots = ((om_c, BranchKind.VA1), (ok_c, BranchKind.VA2),
                 (xm_c, BranchKind.VA3), (xk_c, BranchKind.VA4))

    coords = []
    for chain, branch in slots:
        if not chain:
            coords.append((int(branch), 0, 0, 0))
            continue
        psi = wing_transform(branch, chain, ply, wing=1)
        coords.append((int(branch), int(psi.z.real), int(psi.z.imag), int(psi.zeta)))
    return tuple(coords)


# =====================================================================
# Agent
# =====================================================================

class PsiCheckerAgent:
    def __init__(self, alpha: float = 0.3, gamma: float = 0.95, eps: float = 0.25):
        self.game = CheckersGame()
        self.Q: dict = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        # Track key stats
        self.unique_states: set = set()
        self.unique_actions: set = set()
        # Track collisions: when two different (xm, xk, om, ok, turn, ply)
        # tuples hash to the same Psi address
        self.collision_witnesses: dict[tuple, set[tuple]] = defaultdict(set)

    def _state_addr(self, state, ply: int) -> tuple:
        return psi_address(state.xm, state.xk, state.om, state.ok, state.turn, ply)

    def _candidates(self, state):
        moves = self.game.legal_moves(state)
        if not moves: return moves
        max_c = max(len(m.captures) for m in moves)
        cands = [m for m in moves if len(m.captures) == max_c]
        prom = [m for m in cands if m.promoted]
        return prom if prom else cands

    def value(self, state, move, ply):
        return self.Q[(self._state_addr(state, ply), self.game.move_signature(state, move))]

    def act(self, state, training: bool, ply: int):
        cands = self._candidates(state)
        if not cands: return None
        if training and random.random() < self.eps:
            return random.choice(cands)
        return max(cands, key=lambda m: (
            self.value(state, m, ply),
            len(m.captures),
            int(m.promoted),
        ))

    def update(self, s, a, r, s_next, terminal: bool, ply: int, ply_next: int):
        addr = self._state_addr(s, ply)
        sig = self.game.move_signature(s, a)
        key = (addr, sig)

        # Track collisions: did two distinct positions land on the same addr?
        raw_state = (s.xm, s.xk, s.om, s.ok, s.turn, ply)
        prev = self.collision_witnesses[addr]
        if prev and raw_state not in prev:
            # We've seen this address from a different raw position -- collapse!
            self.unique_states.add(addr)  # still one unique addr
        prev.add(raw_state)

        self.unique_states.add(addr)
        self.unique_actions.add(sig)

        old = self.Q[key]
        if terminal:
            target = r
        else:
            nxt = self._candidates(s_next)
            if not nxt:
                target = r
            else:
                opp_best = max(self.value(s_next, m, ply_next) for m in nxt)
                target = r - self.gamma * opp_best
        self.Q[key] = old + self.alpha * (target - old)

    def collision_stats(self) -> dict:
        total_raw = sum(len(s) for s in self.collision_witnesses.values())
        unique_addr = len(self.collision_witnesses)
        collapses = sum(1 for s in self.collision_witnesses.values() if len(s) > 1)
        return {"unique_addr": unique_addr, "total_raw_positions": total_raw,
                "addresses_with_collisions": collapses}


# =====================================================================
# Episode / eval
# =====================================================================

MAX_PLIES = 200


def play_episode(agent: PsiCheckerAgent, training: bool = True) -> str:
    state = agent.game.initial_state()
    transitions = []
    ply = 0
    winner = None
    while ply < MAX_PLIES:
        winner = agent.game.winner(state)
        if winner is not None: break
        side = state.turn
        m = agent.act(state, training, ply)
        if m is None:
            winner = "O" if side == "X" else "X"
            break
        s_next = agent.game.apply_move(state, m)
        transitions.append((side, state, m, s_next, ply))
        state = s_next
        ply += 1
    if winner is None:
        winner = "draw"
    if training:
        for side, s, a, s_next, p in transitions:
            w_next = agent.game.winner(s_next)
            terminal = w_next is not None
            r = 0.0
            if terminal:
                if w_next == side: r = 1.0
                elif w_next is not None and w_next != "draw": r = -1.0
            agent.update(s, a, r, s_next, terminal, p, p + 1)
    return winner


def random_policy(game, state):
    moves = game.legal_moves(state)
    return random.choice(moves) if moves else None


def eval_vs_random(agent: PsiCheckerAgent, games: int, side: str) -> dict:
    w = d = l = 0
    for _ in range(games):
        state = agent.game.initial_state()
        ply = 0
        winner = None
        while ply < MAX_PLIES:
            winner = agent.game.winner(state)
            if winner is not None: break
            if state.turn == side:
                m = agent.act(state, training=False, ply=ply)
            else:
                m = random_policy(agent.game, state)
            if m is None: break
            state = agent.game.apply_move(state, m)
            ply += 1
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
    print("PSI-ADDRESSED CHECKER LEARNER")
    print("=" * 78)
    print()
    print("Q-key:  state_addr = wing_transform(branch, chain, n=ply, wing=1).coord")
    print("        for each (VA1=my men, VA2=my kings, VA3=opp men, VA4=opp kings)")
    print("        action    = lattice chamber (1..32) + capture_bucket + promoted")
    print()
    print("Comparison baselines:")
    print("  v1 raw-composite Q (700 ep)   : |Q| = 253,712  vs-random X 67%")
    print("  v2 bucket-abstracted Q (700 ep): |Q| =   1,088  vs-random X ~56%")
    print("  Psi-addressed Q (700 ep)      : measure |Q|, growth rate, win rate")
    print("=" * 78)
    print()

    agent = PsiCheckerAgent(alpha=0.3, gamma=0.95, eps=0.25)
    # More checkpoints to see the growth curve
    checkpoints = [25, 50, 100, 200, 400, 700]
    growth_log = []
    t0 = time.time()
    for i in range(1, max(checkpoints) + 1):
        play_episode(agent, training=True)
        if i % 100 == 0:
            agent.eps = max(0.05, 0.25 * (1.0 - i / max(checkpoints)))
        if i in checkpoints:
            vX = eval_vs_random(agent, 30, "X")
            vO = eval_vs_random(agent, 30, "O")
            dt = time.time() - t0
            q_size = len(agent.Q)
            n_states = len(agent.unique_states)
            n_acts = len(agent.unique_actions)
            growth_log.append((i, q_size, n_states, n_acts, dt))
            print(f"  ep={i:>4} ({dt:>5.1f}s)  |Q|={q_size:>5}  states={n_states:>5}  "
                  f"actions={n_acts:>4}   "
                  f"vs random X w/d/l={vX['w']:>2}/{vX['d']}/{vX['l']:>2}   "
                  f"O w/d/l={vO['w']:>2}/{vO['d']}/{vO['l']:>2}")

    print()
    print("=" * 78)
    print("MEMORY GROWTH CURVE")
    print("=" * 78)
    print(f"  {'episode':>8} {'|Q|':>8} {'unique_states':>15} {'|Q|/ep':>8} {'states/ep':>10}")
    for ep, q, st, _, _ in growth_log:
        print(f"  {ep:>8} {q:>8} {st:>15} {q/ep:>8.1f} {st/ep:>10.1f}")

    # Compute growth analysis
    if len(growth_log) >= 2:
        first = growth_log[1]  # skip 25 ep (too early)
        last = growth_log[-1]
        rate_early = first[1] / first[0]
        rate_late = last[1] / last[0]
        slope_change = rate_late / rate_early
        print()
        print(f"  rate early (ep {first[0]:>3}): {rate_early:6.2f} entries/episode")
        print(f"  rate late  (ep {last[0]:>3}): {rate_late:6.2f} entries/episode")
        print(f"  rate ratio: {slope_change:.2f}")
        if slope_change < 0.8:
            print("  -> SUB-LINEAR growth: lattice equalization compressing state.")
        elif slope_change > 1.1:
            print("  -> SUPER-LINEAR growth: late games encountering more novelty.")
        else:
            print("  -> ROUGHLY LINEAR growth: little equalization at this scale.")

    # Collision stats
    print()
    print("=" * 78)
    print("LATTICE EQUALIZATION STATS")
    print("=" * 78)
    cs = agent.collision_stats()
    print(f"  Distinct (xm,xk,om,ok,turn,ply) positions seen: {cs['total_raw_positions']:>6}")
    print(f"  Distinct Psi addresses (Q-key prefixes):        {cs['unique_addr']:>6}")
    print(f"  Addresses that captured 2+ raw positions:       {cs['addresses_with_collisions']:>6}")
    if cs['total_raw_positions'] > 0:
        collapse_ratio = cs['unique_addr'] / cs['total_raw_positions']
        print(f"  Collapse ratio (addresses / raw positions):     {collapse_ratio:.3f}")
        compression = cs['total_raw_positions'] / max(cs['unique_addr'], 1)
        print(f"  Effective compression (raw / addresses):        {compression:.2f}x")

    print()
    print("=" * 78)
    print("FINAL EVALUATION (100 games each side)")
    print("=" * 78)
    agent.eps = 0.0
    fX = eval_vs_random(agent, 100, "X")
    fO = eval_vs_random(agent, 100, "O")
    print(f"  as X: {fX['w']:>3}/100 ({fX['w']}%)   [baseline v1 raw-Q: 67%]")
    print(f"  as O: {fO['w']:>3}/100 ({fO['w']}%)   [baseline v1 raw-Q: 72%]")


if __name__ == "__main__":
    main()
