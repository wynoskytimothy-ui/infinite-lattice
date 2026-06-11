#!/usr/bin/env python3
"""
quad_fork_learner.py - tic-tac-toe with the full 1/2/3/4-way meet hierarchy.

How each order is used:

  1-way   prime divisibility            p in chain  <=>  C % p == 0
  2-way   commutative product           C is order-invariant (swap_meet)
  3-way   missing_member                triple-meet completes a winning line
  4-way   fork detector                 a prime that becomes the missing third
                                        of TWO lines simultaneously

Priority list (all from the meet hierarchy):
  1. winning_move      3-way: I own 2/3 of a line, take the missing prime
  2. blocking_move     3-way: opp owns 2/3, take their missing prime
  3. fork_move         4-way: my move creates 2 simultaneous 3-way threats
                       AND opp has no winning threat afterwards
  4. safe filter       exclude moves that allow opp to win or fork next ply
  5. Q-learning        picks among safe moves for strategic depth

Gate: vs perfect minimax, never lose as either side.
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_complex_plane import missing_member, swap_meet, triple_equalization
from core.primes import chain_primes


# ---------------------------------------------------------------------------
# Prime layout
# ---------------------------------------------------------------------------

SQUARE_PRIMES: tuple[int, ...] = chain_primes(9)
POS_TO_PRIME = {i: p for i, p in enumerate(SQUARE_PRIMES)}
PRIME_TO_POS = {p: i for i, p in POS_TO_PRIME.items()}
ALL_PRIMES = frozenset(SQUARE_PRIMES)
ALL_PRODUCT = 1
for _p in SQUARE_PRIMES:
    ALL_PRODUCT *= _p

_LINES_POS = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)
WIN_LINES: tuple[tuple[int, int, int], ...] = tuple(
    tuple(sorted(POS_TO_PRIME[i] for i in line)) for line in _LINES_POS
)
WIN_PRODUCTS = tuple(a * p * q for (a, p, q) in WIN_LINES)

# Prime "meet degree": how many winning lines each square participates in.
# Derived purely from the meet structure -> center=4, corners=3, sides=2.
PRIME_DEGREE: Counter[int] = Counter()
for _line in WIN_LINES:
    for _p in _line:
        PRIME_DEGREE[_p] += 1


# ---------------------------------------------------------------------------
# Composite-state board
# ---------------------------------------------------------------------------

class PrimeBoard:
    __slots__ = ("cx", "co", "turn")

    def __init__(self, cx: int = 1, co: int = 1, turn: str = "X"):
        self.cx = cx
        self.co = co
        self.turn = turn

    def me_opp(self) -> tuple[int, int]:
        return (self.cx, self.co) if self.turn == "X" else (self.co, self.cx)

    def unused_primes(self) -> tuple[int, ...]:
        c = self.cx * self.co
        return tuple(p for p in SQUARE_PRIMES if c % p != 0)

    def winner(self) -> str | None:
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


# ---------------------------------------------------------------------------
# Meet algebra — 3-way (threats) and 4-way (forks)
# ---------------------------------------------------------------------------

def threats(c_me: int, c_opp: int) -> list[tuple[tuple[int, int, int], int]]:
    """3-way meets: lines where c_me owns 2/3 and the missing prime is free.

    Returns [(line, missing_prime)] using aethos_complex_plane.missing_member.
    """
    out: list[tuple[tuple[int, int, int], int]] = []
    for line in WIN_LINES:
        owned = tuple(p for p in line if c_me % p == 0)
        if len(owned) == 2:
            m = int(missing_member(line, owned))
            if c_opp % m != 0:
                out.append((line, m))
    return out


def winning_move(board: PrimeBoard) -> int | None:
    """3-way: I own 2/3 of a line, take the missing prime to complete it."""
    c_me, c_opp = board.me_opp()
    th = threats(c_me, c_opp)
    return th[0][1] if th else None


def blocking_move(board: PrimeBoard) -> int | None:
    """3-way on opp's chain: opp owns 2/3, take their missing prime."""
    c_me, c_opp = board.me_opp()
    th = threats(c_opp, c_me)
    return th[0][1] if th else None


def fork_move(board: PrimeBoard) -> int | None:
    """4-way meet: a prime p such that c_me*p produces 2+ 3-way threats
    AND opp has no surviving winning threat afterwards.
    """
    c_me, c_opp = board.me_opp()
    for p in board.unused_primes():
        c_me_new = c_me * p
        my_t = len(threats(c_me_new, c_opp))
        if my_t < 2:
            continue
        # opp threats are unchanged by my move except where I blocked them
        opp_t = len(threats(c_opp, c_me_new))
        if opp_t == 0:
            return p
    return None


def opp_can_win_or_fork(board: PrimeBoard) -> bool:
    """After the opp to move, can they win immediately or fork?"""
    return winning_move(board) is not None or fork_move(board) is not None


def safe_moves(board: PrimeBoard) -> list[int]:
    """Legal moves that do not let opp win or fork on the next ply."""
    out: list[int] = []
    for p in board.unused_primes():
        nxt = board.play(p)
        if nxt.winner() is not None:
            out.append(p)
            continue
        if opp_can_win_or_fork(nxt):
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Q-learning agent on top of the meet priority list
# ---------------------------------------------------------------------------

class QuadForkAgent:
    def __init__(self, alpha=0.4, gamma=0.95, eps=0.30):
        self.Q: dict[tuple, float] = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.win_chains: Counter[tuple[int, ...]] = Counter()

    def _q(self, board: PrimeBoard, p: int) -> float:
        return self.Q[(board.key(), p)]

    def _decide(self, board: PrimeBoard, training: bool) -> tuple[int, str]:
        a = winning_move(board)
        if a is not None: return a, "win"
        a = blocking_move(board)
        if a is not None: return a, "block"
        a = fork_move(board)
        if a is not None: return a, "fork"
        safe = safe_moves(board)
        if not safe:
            safe = list(board.unused_primes())
        if training and random.random() < self.eps:
            return random.choice(safe), "explore"
        # break ties with meet-degree (center > corner > side) as a structural prior
        return max(safe, key=lambda p: (self._q(board, p), PRIME_DEGREE[p])), "Q"

    def best_action(self, board: PrimeBoard) -> int:
        return self._decide(board, training=False)[0]

    def act(self, board: PrimeBoard, training: bool) -> int:
        return self._decide(board, training=training)[0]

    def update(self, s: PrimeBoard, a: int, r: float, s_next: PrimeBoard) -> None:
        old = self._q(s, a)
        if s_next.winner() is not None:
            target = r
        else:
            opp_best = max(self._q(s_next, p) for p in s_next.unused_primes())
            target = r - self.gamma * opp_best
        self.Q[(s.key(), a)] = old + self.alpha * (target - old)


# ---------------------------------------------------------------------------
# Self-play / eval
# ---------------------------------------------------------------------------

def reward(winner: str | None, mover: str) -> float:
    if winner is None or winner == "draw":
        return 0.0
    return 1.0 if winner == mover else -1.0


def self_play_episode(agent: QuadForkAgent) -> None:
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
    if winner == "X":
        agent.win_chains[tuple(sorted(x_moves))] += 1
    elif winner == "O":
        agent.win_chains[tuple(sorted(o_moves))] += 1


def eval_random(agent: QuadForkAgent, games: int, side: str) -> dict:
    w = d = l = 0
    for _ in range(games):
        b = PrimeBoard()
        while b.winner() is None:
            a = agent.best_action(b) if b.turn == side else random.choice(b.unused_primes())
            b = b.play(a)
        wr = b.winner()
        if wr == side: w += 1
        elif wr == "draw": d += 1
        else: l += 1
    return {"w": w, "d": d, "l": l}


def minimax_value(b: PrimeBoard, memo: dict) -> int:
    wr = b.winner()
    if wr == "draw": return 0
    if wr is not None: return -1
    k = b.key()
    if k in memo: return memo[k]
    best = -2
    for p in b.unused_primes():
        v = -minimax_value(b.play(p), memo)
        if v > best: best = v
    memo[k] = best
    return best


def eval_minimax(agent: QuadForkAgent, games: int, side: str) -> dict:
    memo: dict = {}
    w = d = l = 0
    for _ in range(games):
        b = PrimeBoard()
        while b.winner() is None:
            if b.turn == side:
                a = agent.best_action(b)
            else:
                legals = list(b.unused_primes())
                a = min(legals, key=lambda p: minimax_value(b.play(p), memo))
            b = b.play(a)
        wr = b.winner()
        if wr == side: w += 1
        elif wr == "draw": d += 1
        else: l += 1
    return {"w": w, "d": d, "l": l}


# ---------------------------------------------------------------------------
# Demo + main
# ---------------------------------------------------------------------------

def demo_meet_hierarchy() -> None:
    print("=" * 72)
    print("MEET HIERARCHY -> game mechanics")
    print("=" * 72)
    print(f"Square primes: {SQUARE_PRIMES}")
    print("Meet degree (= # winning lines each prime appears in):")
    for p in SQUARE_PRIMES:
        kind = "center" if p == 13 else ("corner" if p in (3, 7, 19, 29) else "side")
        print(f"  prime {p:>2}  degree={PRIME_DEGREE[p]}  ({kind})")

    print("\n2-way meet (swap): bank(3)@13 vs bank(13)@3")
    L, R = swap_meet(3, 13)
    print(f"  identical? {L.coord == R.coord}     C=3*13={3*13} regardless of order")

    print("\n3-way meet on row (3,5,7):")
    eq = triple_equalization(3, 5, 7)
    coords = {psi.coord for _, psi in eq.values()}
    print(f"  all 3 pair-witnesses collapse to: {coords}")

    print("\n4-way meet (fork) example:")
    # Build the classic corner-corner-center state: X has {3, 29}, O has {13}.
    b = PrimeBoard(cx=3 * 29, co=13, turn="O")
    print(f"  state: X=(3,29) corners  O=(13) center  O to move")
    of = []
    for p in b.unused_primes():
        c_opp_new = 3 * 29 * p
        if len(threats(c_opp_new, b.co)) >= 2:
            of.append(p)
    print(f"  X's fork-creating primes (4-way meets available next ply): {of}")
    print("  -> O must play a side (5,11,17,23) to neutralize the fork")
    sm = safe_moves(b)
    print(f"  safe_moves(O) = {sorted(sm)}")
    print(f"  (corner moves filtered out because X could fork)")


def train() -> QuadForkAgent:
    print("\n" + "=" * 72)
    print("TRAIN: 1/2/3/4-way meet algebra + Q over safe moves")
    print("=" * 72)
    random.seed(42)
    agent = QuadForkAgent(alpha=0.4, gamma=0.95, eps=0.30)

    checkpoints = [500, 1500, 5000, 15000]
    for i in range(1, max(checkpoints) + 1):
        self_play_episode(agent)
        if i % 500 == 0:
            agent.eps = max(0.05, 0.30 * (1.0 - i / max(checkpoints)))
        if i in checkpoints:
            vR_X = eval_random(agent, 400, "X")
            vR_O = eval_random(agent, 400, "O")
            print(
                f"  ep={i:>5}  |Q|={len(agent.Q):>5}  eps={agent.eps:.2f}   "
                f"vs random  X w/d/l={vR_X['w']}/{vR_X['d']}/{vR_X['l']}   "
                f"O w/d/l={vR_O['w']}/{vR_O['d']}/{vR_O['l']}"
            )
    return agent


def main() -> None:
    demo_meet_hierarchy()
    agent = train()

    print("\n--- Gate: vs perfect minimax (must never lose) ---")
    agent.eps = 0.0
    vM_X = eval_minimax(agent, 50, "X")
    vM_O = eval_minimax(agent, 50, "O")
    print(f"  as X  w/d/l = {vM_X['w']}/{vM_X['d']}/{vM_X['l']}")
    print(f"  as O  w/d/l = {vM_O['w']}/{vM_O['d']}/{vM_O['l']}")
    learned = vM_X['l'] == 0 and vM_O['l'] == 0
    print(f"  LEARNED (zero losses to perfect play): {learned}")

    print("\n--- Top promoted winning chains (next-prime ID assignment) ---")
    next_id = 31
    for rank, (chain, count) in enumerate(agent.win_chains.most_common(6), 1):
        prime_id = next_id
        # advance to the next odd prime
        x = next_id + 2
        while True:
            ok = x > 1
            d = 3
            while d * d <= x:
                if x % d == 0: ok = False; break
                d += 2
            if ok: break
            x += 2
        next_id = x
        print(f"  #{rank}  chain={chain}  wins={count:>5}  -> prime {prime_id}")


if __name__ == "__main__":
    main()
