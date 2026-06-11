#!/usr/bin/env python3
"""
checkers_meet_learner.py - checkers via prime composites + meet algebra.

State alpha = (C_xm, C_xk, C_om, C_ok, turn) using the 4 branches:
    C_X_men   = product of primes for X's men     (branch VA1)
    C_X_kings = product of primes for X's kings   (branch VA2)
    C_O_men   = product of primes for O's men     (branch VA3)
    C_O_kings = product of primes for O's kings   (branch VA4)

32 dark squares labeled by chain_primes(32).
The 4 piece types == the 4 branches VA1..VA4 of your formula, used literally.

Meet algebra in use:
    3-way   single capture: (source, over, land) with (me, opp, empty)
    4-way+  multi-jump chain: stacked 3-way meets, mandatory continuation

Forced-capture rule == "if any 3-way meet exists, you must take one"
    => the meet algebra itself is the legal-move generator.

Honest scope: tabular Q can't solve checkers (state ~ 10^20).
This shows the meet layer scales cleanly to the next tactical depth;
Q-learning fills in what it can within the positions it actually sees.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.primes import chain_primes


# =====================================================================
# Prime mapping for 32 dark squares
# =====================================================================

SQUARE_PRIMES: tuple[int, ...] = chain_primes(32)
PRIME_TO_POS: dict[int, int] = {p: i for i, p in enumerate(SQUARE_PRIMES)}


# =====================================================================
# Geometry: 8x8 board, 32 dark squares
# =====================================================================
# Positions 0..31 number dark squares row-by-row.
# Top of board = row 0. O starts at rows 0,1,2. X starts at rows 5,6,7.
# X men move up (toward row 0); O men move down. Kings move both ways.

def pos_to_rc(pos: int) -> tuple[int, int]:
    row = pos // 4
    col_in_row = pos % 4
    col = 2 * col_in_row + (1 if row % 2 == 0 else 0)
    return row, col


def rc_to_pos(row: int, col: int) -> int | None:
    if not (0 <= row < 8 and 0 <= col < 8):
        return None
    if (row + col) % 2 == 0:
        return None  # light square (not playable)
    return row * 4 + col // 2


DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))  # UL, UR, DL, DR
DIR_NAMES = ("UL", "UR", "DL", "DR")


def _build_step_tables():
    steps, jumps = [], []
    for pos in range(32):
        r, c = pos_to_rc(pos)
        s, j = [], []
        for dr, dc in DIRS:
            over = rc_to_pos(r + dr, c + dc)
            s.append(over)
            if over is None:
                j.append((None, None))
            else:
                land = rc_to_pos(r + 2 * dr, c + 2 * dc)
                j.append((over, land))
        steps.append(tuple(s))
        jumps.append(tuple(j))
    return tuple(steps), tuple(jumps)


STEPS, JUMPS = _build_step_tables()


def move_dir_indices(side: str, is_king: bool) -> tuple[int, ...]:
    if is_king:
        return (0, 1, 2, 3)
    return (0, 1) if side == "X" else (2, 3)


def is_back_rank(side: str, pos: int) -> bool:
    r, _ = pos_to_rc(pos)
    return (r == 0) if side == "X" else (r == 7)


# =====================================================================
# CheckersBoard - composite state
# =====================================================================

@dataclass(frozen=True)
class CheckersBoard:
    xm: int = 1
    xk: int = 1
    om: int = 1
    ok: int = 1
    turn: str = "X"

    def occupied(self) -> int:
        return self.xm * self.xk * self.om * self.ok

    def piece_at(self, pos: int) -> tuple[str, bool] | None:
        p = SQUARE_PRIMES[pos]
        if self.xm % p == 0: return ("X", False)
        if self.xk % p == 0: return ("X", True)
        if self.om % p == 0: return ("O", False)
        if self.ok % p == 0: return ("O", True)
        return None

    def is_empty(self, pos: int) -> bool:
        return self.occupied() % SQUARE_PRIMES[pos] != 0

    def piece_counts(self) -> tuple[int, int, int, int]:
        def c(x: int) -> int:
            return sum(1 for p in SQUARE_PRIMES if x % p == 0)
        return c(self.xm), c(self.xk), c(self.om), c(self.ok)

    def render(self) -> str:
        rows = []
        for r in range(8):
            cells = []
            for c in range(8):
                if (r + c) % 2 == 0:
                    cells.append(" . ")
                else:
                    pos = rc_to_pos(r, c)
                    piece = self.piece_at(pos)
                    if piece is None: cells.append(" - ")
                    elif piece == ("X", False): cells.append(" x ")
                    elif piece == ("X", True): cells.append(" X ")
                    elif piece == ("O", False): cells.append(" o ")
                    else: cells.append(" O ")
            rows.append("".join(cells))
        return "\n".join(rows)


def initial_board() -> CheckersBoard:
    om = 1
    for pos in range(12):
        om *= SQUARE_PRIMES[pos]
    xm = 1
    for pos in range(20, 32):
        xm *= SQUARE_PRIMES[pos]
    return CheckersBoard(xm=xm, om=om, turn="X")


# =====================================================================
# Moves - meet-algebra-driven generation
# =====================================================================

@dataclass(frozen=True)
class Move:
    source: int
    path: tuple[int, ...]        # sequence of landing positions
    captures: tuple[int, ...]    # PRIMES of captured pieces (in capture order)
    promoted: bool

    @property
    def dest(self) -> int:
        return self.path[-1]

    @property
    def is_capture(self) -> bool:
        return len(self.captures) > 0


def _find_jumps(board: CheckersBoard, source: int, side: str, is_king: bool,
                current: int, path: tuple[int, ...],
                captured: tuple[int, ...]) -> list[Move]:
    """Recursive multi-jump enumeration. Each step is a 3-way meet."""
    visited = {source} | set(path)
    captured_positions = {PRIME_TO_POS[p] for p in captured}
    results: list[Move] = []

    for di in move_dir_indices(side, is_king):
        over, land = JUMPS[current][di]
        if over is None or land is None:
            continue
        # 3-way meet check: (current, over, land) requires (me, opp, free)
        over_prime = SQUARE_PRIMES[over]
        if over_prime in captured:
            continue
        piece_over = board.piece_at(over)
        if piece_over is None or piece_over[0] == side:
            continue
        # land must be available: empty in original board, or previously visited,
        # or a captured-piece's old square (captured pieces removed mid-chain)
        land_ok = (land in visited or land in captured_positions
                   or board.is_empty(land))
        if not land_ok:
            continue

        new_path = path + (land,)
        new_captured = captured + (over_prime,)
        # promotion ends a man's multi-jump (standard rule)
        promoted = (not is_king) and is_back_rank(side, land)
        if promoted:
            results.append(Move(source, new_path, new_captured, True))
        else:
            sub = _find_jumps(board, source, side, is_king, land, new_path, new_captured)
            if sub:
                results.extend(sub)
            else:
                results.append(Move(source, new_path, new_captured, False))
    return results


def legal_moves(board: CheckersBoard) -> list[Move]:
    """Captures if any (mandatory), else quiet moves."""
    side = board.turn
    caps: list[Move] = []
    quiet: list[Move] = []
    for pos in range(32):
        piece = board.piece_at(pos)
        if piece is None or piece[0] != side:
            continue
        _, is_king = piece
        jumps = _find_jumps(board, pos, side, is_king, pos, (), ())
        if jumps:
            caps.extend(jumps)
        else:
            for di in move_dir_indices(side, is_king):
                tgt = STEPS[pos][di]
                if tgt is None or not board.is_empty(tgt):
                    continue
                promoted = (not is_king) and is_back_rank(side, tgt)
                quiet.append(Move(pos, (tgt,), (), promoted))
    return caps if caps else quiet


def legal_moves_no_force(board: CheckersBoard) -> list[Move]:
    """All legal moves ignoring forced-capture (for 'lawless' random opponent)."""
    side = board.turn
    out: list[Move] = []
    for pos in range(32):
        piece = board.piece_at(pos)
        if piece is None or piece[0] != side:
            continue
        _, is_king = piece
        out.extend(_find_jumps(board, pos, side, is_king, pos, (), ()))
        for di in move_dir_indices(side, is_king):
            tgt = STEPS[pos][di]
            if tgt is None or not board.is_empty(tgt):
                continue
            promoted = (not is_king) and is_back_rank(side, tgt)
            out.append(Move(pos, (tgt,), (), promoted))
    return out


def apply_move(board: CheckersBoard, move: Move) -> CheckersBoard:
    side = board.turn
    src_p = SQUARE_PRIMES[move.source]
    dest_p = SQUARE_PRIMES[move.dest]
    xm, xk, om, ok = board.xm, board.xk, board.om, board.ok

    # remove source piece, note king status
    if side == "X":
        if xk % src_p == 0:
            was_king = True; xk //= src_p
        else:
            was_king = False; xm //= src_p
    else:
        if ok % src_p == 0:
            was_king = True; ok //= src_p
        else:
            was_king = False; om //= src_p

    becomes_king = was_king or move.promoted
    if side == "X":
        if becomes_king: xk *= dest_p
        else: xm *= dest_p
    else:
        if becomes_king: ok *= dest_p
        else: om *= dest_p

    # remove captured opp pieces (each from whichever chain they live in)
    for cap in move.captures:
        if side == "X":
            if om % cap == 0: om //= cap
            else: ok //= cap
        else:
            if xm % cap == 0: xm //= cap
            else: xk //= cap

    return CheckersBoard(xm=xm, xk=xk, om=om, ok=ok,
                         turn="O" if side == "X" else "X")


def game_winner(board: CheckersBoard) -> str | None:
    """Returns 'X', 'O', or None (None means ongoing)."""
    if board.xm == 1 and board.xk == 1: return "O"
    if board.om == 1 and board.ok == 1: return "X"
    if not legal_moves(board):
        return "O" if board.turn == "X" else "X"
    return None


# =====================================================================
# Q-agent with meet priorities
# =====================================================================

class CheckersMeetAgent:
    def __init__(self, alpha: float = 0.3, gamma: float = 0.95, eps: float = 0.20):
        self.Q: dict[tuple, float] = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def state_key(self, board: CheckersBoard) -> tuple:
        return (board.xm, board.xk, board.om, board.ok, board.turn)

    def value(self, board: CheckersBoard, move: Move) -> float:
        return self.Q[(self.state_key(board), move)]

    def _candidates(self, board: CheckersBoard) -> list[Move]:
        moves = legal_moves(board)
        if not moves:
            return moves
        # meet-prior: when forced to capture, prefer longest chain, then promotion
        max_c = max(len(m.captures) for m in moves)
        cands = [m for m in moves if len(m.captures) == max_c]
        prom = [m for m in cands if m.promoted]
        return prom if prom else cands

    def act(self, board: CheckersBoard, training: bool) -> Move:
        cands = self._candidates(board)
        if training and random.random() < self.eps:
            return random.choice(cands)
        return max(cands, key=lambda m: self.value(board, m))

    def update(self, s: CheckersBoard, a: Move, r: float,
               s_next: CheckersBoard, terminal: bool) -> None:
        old = self.value(s, a)
        if terminal:
            target = r
        else:
            nxt = self._candidates(s_next)
            if not nxt:
                target = r
            else:
                opp_best = max(self.value(s_next, m) for m in nxt)
                target = r - self.gamma * opp_best
        self.Q[(self.state_key(s), a)] = old + self.alpha * (target - old)


# =====================================================================
# Self-play / eval
# =====================================================================

MAX_PLIES = 200


def play_episode(agent: CheckersMeetAgent, training: bool = True) -> str:
    board = initial_board()
    transitions: list[tuple[str, CheckersBoard, Move, CheckersBoard]] = []
    plies = 0
    winner: str | None = None
    while plies < MAX_PLIES:
        winner = game_winner(board)
        if winner is not None:
            break
        side = board.turn
        a = agent.act(board, training=training)
        s_next = apply_move(board, a)
        transitions.append((side, board, a, s_next))
        board = s_next
        plies += 1
    if winner is None:
        winner = "draw"
    if training:
        for side, s, a, s_next in transitions:
            w_next = game_winner(s_next)
            terminal = w_next is not None
            r = 0.0
            if terminal:
                if w_next == side: r = 1.0
                elif w_next is not None and w_next != "draw": r = -1.0
            agent.update(s, a, r, s_next, terminal)
    return winner


def rand_policy_rules(board: CheckersBoard) -> Move | None:
    moves = legal_moves(board)
    return random.choice(moves) if moves else None


def rand_policy_no_rules(board: CheckersBoard) -> Move | None:
    moves = legal_moves_no_force(board)
    return random.choice(moves) if moves else None


def eval_vs(agent: CheckersMeetAgent, games: int, agent_side: str,
            opp_policy) -> dict:
    w = d = l = 0
    for _ in range(games):
        b = initial_board()
        plies = 0
        winner: str | None = None
        while plies < MAX_PLIES:
            winner = game_winner(b)
            if winner is not None:
                break
            if b.turn == agent_side:
                m = agent.act(b, training=False)
            else:
                m = opp_policy(b)
                if m is None:
                    winner = agent_side
                    break
            b = apply_move(b, m)
            plies += 1
        if winner is None:
            winner = "draw"
        if winner == agent_side: w += 1
        elif winner == "draw":  d += 1
        else:                    l += 1
    return {"w": w, "d": d, "l": l}


# =====================================================================
# Demos
# =====================================================================

def demo_meet_detection() -> None:
    print("=" * 72)
    print("MEET ALGEBRA -> checkers tactics")
    print("=" * 72)
    print(f"First 8 square primes: {SQUARE_PRIMES[:8]}... ({len(SQUARE_PRIMES)} total)")
    print(f"Last 4 square primes:  {SQUARE_PRIMES[-4:]}")

    print("\n--- Initial board ---")
    b = initial_board()
    print(b.render())
    xm, xk, om, ok = b.piece_counts()
    print(f"\nPieces: X men={xm}, X kings={xk}, O men={om}, O kings={ok}")
    print(f"Legal opening moves for X: {len(legal_moves(b))}")

    print("\n--- 3-way meet: single capture ---")
    # X at pos 17, O at pos 13. X jumps UL: 17 -> over 13 -> land 8.
    b1 = CheckersBoard(xm=SQUARE_PRIMES[17], om=SQUARE_PRIMES[13], turn="X")
    print(b1.render())
    moves = legal_moves(b1)
    print(f"\nLegal moves: {len(moves)} (forced capture)")
    for m in moves:
        print(f"  {m.source} -> {' -> '.join(map(str, m.path))}  captures primes {m.captures}")

    print("\n--- 4-way meet: double jump (two stacked 3-ways) ---")
    # X at 21. O at 16 and 8. Empty at 12 and 5.
    b2 = CheckersBoard(xm=SQUARE_PRIMES[21],
                        om=SQUARE_PRIMES[16] * SQUARE_PRIMES[8], turn="X")
    print(b2.render())
    moves = legal_moves(b2)
    print(f"\nLegal moves: {len(moves)}")
    for m in moves:
        chain = ' -> '.join(map(str, (m.source,) + m.path))
        print(f"  {chain}  captures {len(m.captures)} pieces (primes {m.captures})")

    print("\n--- 5-way meet: triple jump with promotion ---")
    # X at pos 26 = (6, 5).
    # Jump UL over pos 22 = (5, 4)  ->  land pos 17 = (4, 3).
    # Jump UL over pos 13 = (3, 2)  ->  land pos  8 = (2, 1).
    # Jump UR over pos  5 = (1, 2)  ->  land pos  1 = (0, 3)  back rank -> PROMOTE.
    b3 = CheckersBoard(xm=SQUARE_PRIMES[26],
                        om=(SQUARE_PRIMES[22] * SQUARE_PRIMES[13] * SQUARE_PRIMES[5]),
                        turn="X")
    print(b3.render())
    moves = legal_moves(b3)
    print(f"\nLegal moves: {len(moves)}")
    for m in moves:
        chain = ' -> '.join(map(str, (m.source,) + m.path))
        print(f"  {chain}  captures={len(m.captures)}  promoted={m.promoted}")


def train_and_eval() -> None:
    print("\n" + "=" * 72)
    print("SELF-PLAY TRAINING (tabular Q + meet priorities)")
    print("=" * 72)
    random.seed(42)
    agent = CheckersMeetAgent(alpha=0.3, gamma=0.95, eps=0.25)
    checkpoints = [25, 100, 300, 700]
    for i in range(1, max(checkpoints) + 1):
        play_episode(agent)
        if i % 100 == 0:
            agent.eps = max(0.05, 0.25 * (1.0 - i / max(checkpoints)))
        if i in checkpoints:
            vR = eval_vs(agent, 40, "X", rand_policy_rules)
            vU = eval_vs(agent, 40, "X", rand_policy_no_rules)
            print(f"  ep={i:>4}  |Q|={len(agent.Q):>6}  eps={agent.eps:.2f}   "
                  f"vs random(forced caps)  w/d/l={vR['w']}/{vR['d']}/{vR['l']}   "
                  f"vs random(no rules)  w/d/l={vU['w']}/{vU['d']}/{vU['l']}")
    return agent


def sample_game(agent: CheckersMeetAgent) -> None:
    print("\n--- Sample game: agent (X) vs random+rules (O) ---")
    random.seed(7)
    b = initial_board()
    plies = 0
    cap_log: list[tuple[int, str, int, bool]] = []
    while plies < MAX_PLIES:
        w = game_winner(b)
        if w is not None:
            break
        if b.turn == "X":
            m = agent.act(b, training=False)
        else:
            m = rand_policy_rules(b)
            if m is None:
                break
        if m.is_capture or m.promoted:
            cap_log.append((plies, b.turn, len(m.captures), m.promoted))
        b = apply_move(b, m)
        plies += 1
    winner = game_winner(b) or "draw"
    print(f"Final: {winner}  ({plies} plies)")
    xm, xk, om, ok = b.piece_counts()
    print(f"Final pieces: X men={xm} kings={xk}  O men={om} kings={ok}")
    print(b.render())
    if cap_log:
        print(f"\nTactical events ({len(cap_log)}):")
        for ply, side, ncap, prom in cap_log[:25]:
            tag = []
            if ncap > 0: tag.append(f"capture x{ncap}")
            if prom:    tag.append("PROMOTION")
            print(f"  ply {ply:>3}  {side}: {', '.join(tag)}")


def main() -> None:
    demo_meet_detection()
    agent = train_and_eval()
    sample_game(agent)


if __name__ == "__main__":
    main()
