#!/usr/bin/env python3
"""
Tic-tac-toe on the AETHOS 3D complex plane.

Board squares are labeled by the first 9 odd primes:

    pos 0  pos 1  pos 2        3   5   7
    pos 3  pos 4  pos 5   =   11  13  17
    pos 6  pos 7  pos 8       19  23  29

Each ply, the lattice address is alpha = (A, b, w, n):
    A_X = sorted primes X has played    (branch VA1)
    A_O = sorted primes O has played    (branch VA3)
    unused primes = ALL_PRIMES \\ (A_X | A_O) = the legal-move set
    n   = ply count (transgressor)

Self-play tabular Q-learning verifies the encoding is rich enough to learn:
the gate is "never loses to perfect minimax" (best case: all draws).
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_complex_plane import wing_transform
from aethos_lattice import BranchKind
from core.primes import chain_primes


# ---------------------------------------------------------------------------
# Board / primes
# ---------------------------------------------------------------------------

SQUARE_PRIMES: tuple[int, ...] = chain_primes(9)  # 3,5,7,11,13,17,19,23,29
POS_TO_PRIME: dict[int, int] = {i: p for i, p in enumerate(SQUARE_PRIMES)}
PRIME_TO_POS: dict[int, int] = {p: i for i, p in POS_TO_PRIME.items()}
ALL_PRIMES: frozenset[int] = frozenset(SQUARE_PRIMES)

# 8 winning lines as prime sets (rows, cols, diagonals)
_LINES_POS = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)
WINNING_LINES: tuple[frozenset[int], ...] = tuple(
    frozenset(POS_TO_PRIME[i] for i in line) for line in _LINES_POS
)


class Board:
    """Tic-tac-toe state addressed by prime chains."""

    __slots__ = ("x", "o", "turn")

    def __init__(
        self,
        x: frozenset[int] = frozenset(),
        o: frozenset[int] = frozenset(),
        turn: str = "X",
    ):
        self.x = x
        self.o = o
        self.turn = turn

    @property
    def n(self) -> int:
        return len(self.x) + len(self.o)

    def chain(self, side: str) -> tuple[int, ...]:
        return tuple(sorted(self.x if side == "X" else self.o))

    def unused_primes(self) -> frozenset[int]:
        return ALL_PRIMES - self.x - self.o

    def winner(self) -> str | None:
        for line in WINNING_LINES:
            if line <= self.x:
                return "X"
            if line <= self.o:
                return "O"
        if not self.unused_primes():
            return "draw"
        return None

    def play(self, prime: int) -> "Board":
        if prime not in self.unused_primes():
            raise ValueError(f"prime {prime} not legal: x={self.x} o={self.o}")
        if self.turn == "X":
            return Board(self.x | {prime}, self.o, "O")
        return Board(self.x, self.o | {prime}, "X")

    def key(self) -> tuple:
        return (self.x, self.o, self.turn)

    def psi_readout(self) -> dict:
        """Lattice readouts witnessing the encoding (X via VA1, O via VA3)."""
        out: dict = {}
        if self.x:
            out["X_VA1_w1"] = wing_transform(BranchKind.VA1, self.chain("X"), self.n, wing=1)
        if self.o:
            out["O_VA3_w1"] = wing_transform(BranchKind.VA3, self.chain("O"), self.n, wing=1)
        return out

    def render(self) -> str:
        rows = []
        for r in range(3):
            cells = []
            for c in range(3):
                pos = r * 3 + c
                p = POS_TO_PRIME[pos]
                if p in self.x:
                    cells.append(" X ")
                elif p in self.o:
                    cells.append(" O ")
                else:
                    cells.append(f"{p:>2} ")
            rows.append("|".join(cells))
        return "\n-----------\n".join(rows)


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class QAgent:
    """Tabular Q on (state, action) where action = an unused prime.

    Q(s,a) is the value of playing a in s from the perspective of the side
    to move in s. Self-play: each transition updates only the moving side.
    """

    def __init__(self, alpha: float = 0.4, gamma: float = 0.95, eps: float = 0.2):
        self.Q: dict[tuple, float] = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def value(self, board: Board, action_prime: int) -> float:
        return self.Q[(board.key(), action_prime)]

    def best_action(self, board: Board) -> int:
        legals = list(board.unused_primes())
        return max(legals, key=lambda p: self.value(board, p))

    def act(self, board: Board, training: bool) -> int:
        legals = list(board.unused_primes())
        if training and random.random() < self.eps:
            return random.choice(legals)
        return max(legals, key=lambda p: self.value(board, p))

    def update(self, s: Board, a: int, r: float, s_next: Board) -> None:
        old = self.value(s, a)
        if s_next.winner() is not None:
            target = r
        else:
            # opponent moves next; from mover's POV, opp's best is bad for us
            opp_best = max(self.value(s_next, p) for p in s_next.unused_primes())
            target = r - self.gamma * opp_best
        self.Q[(s.key(), a)] = old + self.alpha * (target - old)


# ---------------------------------------------------------------------------
# Self-play, eval, minimax oracle
# ---------------------------------------------------------------------------

def reward_for_mover(winner: str | None, mover: str) -> float:
    if winner is None or winner == "draw":
        return 0.0
    return 1.0 if winner == mover else -1.0


def self_play_episode(agent: QAgent) -> str | None:
    board = Board()
    transitions: list[tuple[str, Board, int, Board]] = []
    while board.winner() is None:
        side = board.turn
        a = agent.act(board, training=True)
        s_next = board.play(a)
        transitions.append((side, board, a, s_next))
        board = s_next
    winner = board.winner()
    for side, s, a, s_next in transitions:
        r = reward_for_mover(winner, side) if s_next.winner() is not None else 0.0
        agent.update(s, a, r, s_next)
    return winner


def evaluate_vs_random(agent: QAgent, games: int, agent_side: str) -> dict:
    w_w = w_d = w_l = 0
    for _ in range(games):
        board = Board()
        while board.winner() is None:
            if board.turn == agent_side:
                a = agent.best_action(board)
            else:
                a = random.choice(list(board.unused_primes()))
            board = board.play(a)
        w = board.winner()
        if w == agent_side:
            w_w += 1
        elif w == "draw":
            w_d += 1
        else:
            w_l += 1
    return {"w": w_w, "d": w_d, "l": w_l}


def minimax_value(board: Board, memo: dict) -> int:
    """Value to side-to-move: +1 win / 0 draw / -1 loss under perfect play."""
    w = board.winner()
    if w == "draw":
        return 0
    if w is not None:
        # someone just played and won the game => side-to-move has already lost
        return -1
    k = board.key()
    if k in memo:
        return memo[k]
    best = -2
    for p in board.unused_primes():
        v = -minimax_value(board.play(p), memo)
        if v > best:
            best = v
    memo[k] = best
    return best


def minimax_action(board: Board, memo: dict) -> int:
    legals = list(board.unused_primes())
    # pick prime that minimizes opponent's value after our move
    return min(legals, key=lambda p: minimax_value(board.play(p), memo))


def evaluate_vs_minimax(agent: QAgent, games: int, agent_side: str) -> dict:
    memo: dict = {}
    w_w = w_d = w_l = 0
    for _ in range(games):
        board = Board()
        while board.winner() is None:
            if board.turn == agent_side:
                a = agent.best_action(board)
            else:
                a = minimax_action(board, memo)
            board = board.play(a)
        w = board.winner()
        if w == agent_side:
            w_w += 1
        elif w == "draw":
            w_d += 1
        else:
            w_l += 1
    return {"w": w_w, "d": w_d, "l": w_l}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)
    print("=" * 72)
    print("TIC-TAC-TOE on AETHOS 3D complex plane")
    print("=" * 72)
    print(f"Square primes: {SQUARE_PRIMES}  (center = prime {POS_TO_PRIME[4]})")
    print("State alpha = (A, b, w, n)")
    print("  X -> A_X (VA1)   O -> A_O (VA3)   unused primes = legal moves\n")

    agent = QAgent(alpha=0.4, gamma=0.95, eps=0.30)

    print("--- Self-play training ---")
    checkpoints = [500, 2500, 10000, 30000, 60000]
    train_games = checkpoints[-1]
    for i in range(1, train_games + 1):
        self_play_episode(agent)
        # gentle epsilon decay
        if i % 1000 == 0:
            agent.eps = max(0.05, 0.30 * (1.0 - i / train_games))
        if i in checkpoints:
            vsR_X = evaluate_vs_random(agent, 400, "X")
            vsR_O = evaluate_vs_random(agent, 400, "O")
            print(
                f"  ep={i:>6}  |Q|={len(agent.Q):>6}  eps={agent.eps:.2f}   "
                f"vs random  X w/d/l={vsR_X['w']}/{vsR_X['d']}/{vsR_X['l']}   "
                f"O w/d/l={vsR_O['w']}/{vsR_O['d']}/{vsR_O['l']}"
            )

    print("\n--- Gate: vs perfect minimax (any loss = not learned) ---")
    agent.eps = 0.0
    vsM_X = evaluate_vs_minimax(agent, 50, "X")
    vsM_O = evaluate_vs_minimax(agent, 50, "O")
    print(f"  agent as X  w/d/l = {vsM_X['w']}/{vsM_X['d']}/{vsM_X['l']}")
    print(f"  agent as O  w/d/l = {vsM_O['w']}/{vsM_O['d']}/{vsM_O['l']}")
    learned = vsM_X['l'] == 0 and vsM_O['l'] == 0
    print(f"  LEARNED (no losses to perfect play): {learned}")

    print("\n--- Sample greedy self-play with lattice readout ---")
    board = Board()
    ply = 0
    while board.winner() is None:
        side = board.turn
        unused = sorted(board.unused_primes())
        a = agent.best_action(board)
        print(f"\nply n={ply}  turn={side}  unused primes = {unused}")
        print(f"  picks prime {a}  (pos {PRIME_TO_POS[a]})")
        # readout BEFORE the move shows current state encoding
        for label, psi in board.psi_readout().items():
            z = psi.z
            print(f"  {label}: z={z.real:+.0f}{z.imag:+.0f}i  zeta={psi.zeta:.0f}  |z|^2={psi.modulus_squared:.0f}")
        board = board.play(a)
        ply += 1
    print(f"\nFinal result: {board.winner()}  (ply n={ply})")
    print(board.render())


if __name__ == "__main__":
    main()
