#!/usr/bin/env python3
"""
Tic-tac-toe v2 — using prime composites + 2-way + 3-way meet algebra.

How each piece of the formula is used:

  Prime composites (ICN encoding)
    C_X = product of X's primes; C_O = product of O's primes.
    Add a piece: C *= p. Played(p): C % p == 0. Factor C -> chain A.
    State key = (C_X, C_O, turn) - one tuple of integers per position.

  2-way correlation  (swap_meet)
    bank(a)@p == bank(p)@a in your lattice.
    Composite form: C is order-invariant -> all permutations of a position
    collapse to ONE Q-table entry. Free ~5x state-space cut.

  3-way correlation  (triple_equalization, missing-variable rule)
    The 8 winning lines on a 3x3 board ARE 8 triple-meet nodes.
    Threat detection = missing_member(line, owned_subset):
      own {3,5}, line is {3,5,7}, missing -> 7   (the winning move)
    Computed directly from aethos_complex_plane.missing_member.

  Promotion
    A winning ply sequence (p1, p2, ..., pk) is logged as a chain.
    Frequent winning openings get a fresh prime above the board pool;
    the chain itself is preserved as the "macro" expansion.
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_complex_plane import (
    canon_complex,
    missing_member,
    swap_meet,
    triple_equalization,
)
from aethos_lattice import BranchKind
from core.primes import chain_primes


# ---------------------------------------------------------------------------
# Prime layout
# ---------------------------------------------------------------------------

SQUARE_PRIMES: tuple[int, ...] = chain_primes(9)            # 3..29
POS_TO_PRIME = {i: p for i, p in enumerate(SQUARE_PRIMES)}
PRIME_TO_POS = {p: i for i, p in POS_TO_PRIME.items()}
ALL_PRIMES = frozenset(SQUARE_PRIMES)
ALL_PRODUCT = 1
for _p in SQUARE_PRIMES:
    ALL_PRODUCT *= _p

# 8 winning lines as ORDERED prime triples (a<p<q for missing_member)
_LINES_POS = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)
WIN_LINES: tuple[tuple[int, int, int], ...] = tuple(
    tuple(sorted(POS_TO_PRIME[i] for i in line)) for line in _LINES_POS
)
WIN_PRODUCTS = tuple(a * p * q for (a, p, q) in WIN_LINES)


# ---------------------------------------------------------------------------
# Prime-composite board
# ---------------------------------------------------------------------------

class PrimeBoard:
    """State = (C_X, C_O, turn). Order-invariant by construction."""

    __slots__ = ("cx", "co", "turn")

    def __init__(self, cx: int = 1, co: int = 1, turn: str = "X"):
        self.cx = cx
        self.co = co
        self.turn = turn

    def chain(self, side: str) -> tuple[int, ...]:
        c = self.cx if side == "X" else self.co
        return tuple(p for p in SQUARE_PRIMES if c % p == 0)

    def unused_primes(self) -> tuple[int, ...]:
        c = self.cx * self.co
        return tuple(p for p in SQUARE_PRIMES if c % p != 0)

    def winner(self) -> str | None:
        # Win = some line product divides player's composite
        for prod in WIN_PRODUCTS:
            if self.cx % prod == 0:
                return "X"
            if self.co % prod == 0:
                return "O"
        if self.cx * self.co == ALL_PRODUCT:
            return "draw"
        return None

    def play(self, prime: int) -> "PrimeBoard":
        if (self.cx * self.co) % prime == 0:
            raise ValueError(f"prime {prime} already played")
        if self.turn == "X":
            return PrimeBoard(self.cx * prime, self.co, "O")
        return PrimeBoard(self.cx, self.co * prime, "X")

    def key(self) -> tuple[int, int, str]:
        return (self.cx, self.co, self.turn)

    def n(self) -> int:
        # ply count = number of distinct primes dividing C_X*C_O
        c = self.cx * self.co
        return sum(1 for p in SQUARE_PRIMES if c % p == 0)


# ---------------------------------------------------------------------------
# Meet-driven heuristics (the 3-way correlation in action)
# ---------------------------------------------------------------------------

def lines_owned_two_of_three(c_side: int) -> list[tuple[tuple[int, int, int], int]]:
    """For each line, return (line, missing_prime) where player owns exactly 2/3."""
    out = []
    for line, prod in zip(WIN_LINES, WIN_PRODUCTS):
        owned = tuple(p for p in line if c_side % p == 0)
        if len(owned) == 2:
            # missing-variable rule (triple meet): the unique missing anchor
            missing = missing_member(line, owned)
            out.append((line, int(missing)))
    return out


def winning_move(board: PrimeBoard) -> int | None:
    """If side-to-move can complete a line this turn, return that prime."""
    c_me = board.cx if board.turn == "X" else board.co
    c_opp = board.co if board.turn == "X" else board.cx
    for _line, m in lines_owned_two_of_three(c_me):
        if c_opp % m != 0:  # not blocked by opp
            return m
    return None


def blocking_move(board: PrimeBoard) -> int | None:
    """If opp threatens to complete next turn, return the prime that blocks."""
    c_me = board.cx if board.turn == "X" else board.co
    c_opp = board.co if board.turn == "X" else board.cx
    for _line, m in lines_owned_two_of_three(c_opp):
        if c_me % m != 0:
            return m
    return None


# ---------------------------------------------------------------------------
# Q-learning agent with meet-prior
# ---------------------------------------------------------------------------

class MeetQAgent:
    def __init__(self, alpha=0.4, gamma=0.95, eps=0.20, use_meet_prior=True):
        self.Q: dict[tuple, float] = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.use_meet_prior = use_meet_prior
        # Promotion log: winning move sequences (chains) -> count
        self.win_sequences: Counter[tuple[int, ...]] = Counter()

    def value(self, board: PrimeBoard, p: int) -> float:
        return self.Q[(board.key(), p)]

    def best_action(self, board: PrimeBoard) -> int:
        # Meet-prior overrides exploration when a decisive move exists
        if self.use_meet_prior:
            wm = winning_move(board)
            if wm is not None:
                return wm
            bm = blocking_move(board)
            if bm is not None:
                return bm
        legals = list(board.unused_primes())
        return max(legals, key=lambda p: self.value(board, p))

    def act(self, board: PrimeBoard, training: bool) -> int:
        if self.use_meet_prior:
            wm = winning_move(board)
            if wm is not None:
                return wm
            bm = blocking_move(board)
            if bm is not None:
                return bm
        legals = list(board.unused_primes())
        if training and random.random() < self.eps:
            return random.choice(legals)
        return max(legals, key=lambda p: self.value(board, p))

    def update(self, s: PrimeBoard, a: int, r: float, s_next: PrimeBoard) -> None:
        old = self.value(s, a)
        if s_next.winner() is not None:
            target = r
        else:
            opp_best = max(self.value(s_next, p) for p in s_next.unused_primes())
            target = r - self.gamma * opp_best
        self.Q[(s.key(), a)] = old + self.alpha * (target - old)


# ---------------------------------------------------------------------------
# Self-play / eval (mirrors v1 so the comparison is apples-to-apples)
# ---------------------------------------------------------------------------

def reward(winner: str | None, mover: str) -> float:
    if winner is None or winner == "draw":
        return 0.0
    return 1.0 if winner == mover else -1.0


def self_play_episode(agent: MeetQAgent) -> None:
    board = PrimeBoard()
    transitions: list[tuple[str, PrimeBoard, int, PrimeBoard]] = []
    x_moves: list[int] = []
    o_moves: list[int] = []
    while board.winner() is None:
        side = board.turn
        a = agent.act(board, training=True)
        s_next = board.play(a)
        transitions.append((side, board, a, s_next))
        (x_moves if side == "X" else o_moves).append(a)
        board = s_next
    winner = board.winner()
    for side, s, a, s_next in transitions:
        r = reward(winner, side) if s_next.winner() is not None else 0.0
        agent.update(s, a, r, s_next)
    # Promotion log: record winning side's move sequence as a chain
    if winner == "X":
        agent.win_sequences[tuple(sorted(x_moves))] += 1
    elif winner == "O":
        agent.win_sequences[tuple(sorted(o_moves))] += 1


def evaluate_vs_random(agent: MeetQAgent, games: int, agent_side: str) -> dict:
    w = d = l = 0
    for _ in range(games):
        b = PrimeBoard()
        while b.winner() is None:
            if b.turn == agent_side:
                a = agent.best_action(b)
            else:
                a = random.choice(b.unused_primes())
            b = b.play(a)
        wr = b.winner()
        if wr == agent_side: w += 1
        elif wr == "draw":   d += 1
        else:                l += 1
    return {"w": w, "d": d, "l": l}


def minimax_value(b: PrimeBoard, memo: dict) -> int:
    w = b.winner()
    if w == "draw": return 0
    if w is not None: return -1
    k = b.key()
    if k in memo: return memo[k]
    best = -2
    for p in b.unused_primes():
        v = -minimax_value(b.play(p), memo)
        if v > best: best = v
    memo[k] = best
    return best


def evaluate_vs_minimax(agent: MeetQAgent, games: int, agent_side: str) -> dict:
    memo: dict = {}
    w = d = l = 0
    for _ in range(games):
        b = PrimeBoard()
        while b.winner() is None:
            if b.turn == agent_side:
                a = agent.best_action(b)
            else:
                legals = list(b.unused_primes())
                a = min(legals, key=lambda p: minimax_value(b.play(p), memo))
            b = b.play(a)
        wr = b.winner()
        if wr == agent_side: w += 1
        elif wr == "draw":   d += 1
        else:                l += 1
    return {"w": w, "d": d, "l": l}


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def demo_meet_algebra() -> None:
    print("=" * 72)
    print("MEET ALGEBRA -> game mechanics")
    print("=" * 72)
    print(f"Square primes: {SQUARE_PRIMES}")
    print(f"ALL_PRODUCT = {ALL_PRODUCT}  (factor it back -> SQUARE_PRIMES)\n")

    print("8 winning lines as triple-meet nodes:")
    for line, prod in zip(WIN_LINES, WIN_PRODUCTS):
        print(f"  line={line}  line_product={prod}")

    print("\n--- 2-way meet (order invariance): swap_meet(3, 13) ---")
    L, R = swap_meet(3, 13)
    print(f"  bank(3)@n=13:  z={L.z}  zeta={L.zeta}")
    print(f"  bank(13)@n=3:  z={R.z}  zeta={R.zeta}")
    print(f"  equal? {L.coord == R.coord}     <- composite form C=3*13=39 is identical either order")

    print("\n--- 3-way meet (triple_equalization) on winning row (3,5,7) ---")
    eq = triple_equalization(3, 5, 7)
    for label, (n_w, psi) in eq.items():
        print(f"  {label} subset @ n={int(n_w)}: z={psi.z}  zeta={psi.zeta:.0f}")
    coords = {psi.coord for _, psi in eq.values()}
    print(f"  all three collapse to ONE node: {coords}")

    print("\n--- Threat detection via missing_member ---")
    # X owns {3,5} on the top row. What completes it? -> 7
    line = (3, 5, 7)
    owned = (3, 5)
    m = missing_member(line, owned)
    print(f"  X owns {owned}, line={line}  ->  missing prime = {int(m)}  (winning move)")
    # X owns {3,7}. Missing = 5 (center of top row)
    owned = (3, 7)
    m = missing_member(line, owned)
    print(f"  X owns {owned}, line={line}  ->  missing prime = {int(m)}")

    print("\n--- Composite-key state demo ---")
    b = PrimeBoard()
    seq = [3, 13, 7, 5, 11]  # X:3, O:13, X:7, O:5, X:11 (X completes top row 3,5,7? no, X has {3,7,11})
    for p in seq:
        side = b.turn
        b = b.play(p)
        print(f"  {side} plays {p:2}   C_X={b.cx:>8}  C_O={b.co:>8}   n={b.n()}")
    print(f"  winner = {b.winner()}")


def train_and_compare() -> None:
    random.seed(42)
    print("\n" + "=" * 72)
    print("TRAIN: composite-state + meet-prior agent")
    print("=" * 72)
    agent = MeetQAgent(alpha=0.4, gamma=0.95, eps=0.30, use_meet_prior=True)

    checkpoints = [500, 2500, 10000, 30000]
    for i in range(1, max(checkpoints) + 1):
        self_play_episode(agent)
        if i % 1000 == 0:
            agent.eps = max(0.05, 0.30 * (1.0 - i / max(checkpoints)))
        if i in checkpoints:
            vR_X = evaluate_vs_random(agent, 400, "X")
            vR_O = evaluate_vs_random(agent, 400, "O")
            print(
                f"  ep={i:>6}  |Q|={len(agent.Q):>5}  eps={agent.eps:.2f}   "
                f"vs random  X w/d/l={vR_X['w']}/{vR_X['d']}/{vR_X['l']}   "
                f"O w/d/l={vR_O['w']}/{vR_O['d']}/{vR_O['l']}"
            )

    print("\n--- Gate: vs perfect minimax ---")
    agent.eps = 0.0
    vM_X = evaluate_vs_minimax(agent, 50, "X")
    vM_O = evaluate_vs_minimax(agent, 50, "O")
    print(f"  agent as X w/d/l = {vM_X['w']}/{vM_X['d']}/{vM_X['l']}")
    print(f"  agent as O w/d/l = {vM_O['w']}/{vM_O['d']}/{vM_O['l']}")
    learned = vM_X['l'] == 0 and vM_O['l'] == 0
    print(f"  LEARNED: {learned}")

    print("\n--- Promotion candidates: top winning chains (sorted prime tuples) ---")
    top = agent.win_sequences.most_common(8)
    next_prime_id = 31  # first unused prime above board pool
    print(f"  fresh prime pool: starts at {next_prime_id}, 37, 41, 43, ...")
    for rank, (chain, count) in enumerate(top, 1):
        promoted = next_prime_id
        # advance to next prime (simple sieve step for demo)
        x = next_prime_id + 2
        while True:
            ok = True
            d = 3
            while d * d <= x:
                if x % d == 0:
                    ok = False
                    break
                d += 2
            if ok: break
            x += 2
        next_prime_id = x
        print(f"  #{rank}  chain={chain}  wins={count}  -> promoted prime = {promoted}")


def main() -> None:
    demo_meet_algebra()
    train_and_compare()


if __name__ == "__main__":
    main()
