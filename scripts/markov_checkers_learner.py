#!/usr/bin/env python3
"""
markov_checkers_learner.py - Markov state abstraction + 32-chamber actions.

User's insight: raw-state Q is redundant because the game is Markovian.
Future depends only on present state, not on history. Many positions are
correlated -> they lead to the same outcomes via the same transitions.
We don't need every (position, action). We need the right abstraction.

Two compressions, both from your formula's structure:

1. State abstraction (Markov - keep only what matters for the future):
   s = (material_bucket, kings_bucket, phase_bucket)   from side-to-move POV
   max |state| = 7 * 5 * 3 = 105

2. Action signature using the 32-chamber lattice:
   chamber = (branch - 1) * 8 + wing
     branch in {VA1=our-man, VA2=our-king, VA3=opp-man, VA4=opp-king}
     wing 1..4 = (UL, UR, DL, DR) non-capture
     wing 5..8 = same directions, capture variants
   a = (chamber, capture_bucket, promoted)
   max |action| = 32 * 5 * 2 = 320

|Q| max ~ 33,600.   v1 (raw state) grew to 253,712.

Test: do we learn faster, slower, or about the same?
"""

from __future__ import annotations
import random
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parent))

from checkers_meet_learner import (
    CheckersBoard, Move, initial_board,
    legal_moves, legal_moves_no_force, apply_move, game_winner,
    SQUARE_PRIMES, pos_to_rc, MAX_PLIES,
)


# =====================================================================
# State abstraction (perspective-flipped to side-to-move)
# =====================================================================

def state_abs(board: CheckersBoard) -> tuple[int, int, int]:
    xm, xk, om, ok = board.piece_counts()
    me_str = (xm + 2 * xk) if board.turn == "X" else (om + 2 * ok)
    opp_str = (om + 2 * ok) if board.turn == "X" else (xm + 2 * xk)
    me_k = xk if board.turn == "X" else ok
    opp_k = ok if board.turn == "X" else xk

    mat = me_str - opp_str
    if   mat <= -10: mb = 0
    elif mat <= -5:  mb = 1
    elif mat <= -1:  mb = 2
    elif mat == 0:   mb = 3
    elif mat <= 4:   mb = 4
    elif mat <= 9:   mb = 5
    else:            mb = 6

    kd = me_k - opp_k
    if   kd <= -2: kb = 0
    elif kd == -1: kb = 1
    elif kd == 0:  kb = 2
    elif kd == 1:  kb = 3
    else:          kb = 4

    total = xm + xk + om + ok
    if   total >= 20: pb = 0
    elif total >= 10: pb = 1
    else:             pb = 2

    return (mb, kb, pb)


# =====================================================================
# Action chamber (32 = 4 branches * 8 wings)
# =====================================================================

def _dir_idx(src_pos: int, dst_pos: int) -> int:
    sr, sc = pos_to_rc(src_pos)
    fr, fc = pos_to_rc(dst_pos)
    if fr < sr and fc < sc: return 0  # UL
    if fr < sr and fc > sc: return 1  # UR
    if fr > sr and fc < sc: return 2  # DL
    return 3                           # DR


def move_chamber(board: CheckersBoard, move: Move) -> int:
    """Lattice chamber for the move: (branch-1)*8 + wing in 1..32."""
    src_p = SQUARE_PRIMES[move.source]
    side = board.turn
    if side == "X":
        is_king = (board.xk % src_p == 0)
        branch = 2 if is_king else 1
    else:
        is_king = (board.ok % src_p == 0)
        branch = 4 if is_king else 3
    di = _dir_idx(move.source, move.path[0])
    wing = (di + 1) + (4 if move.is_capture else 0)  # 1..8
    return (branch - 1) * 8 + wing  # 1..32


def action_sig(board: CheckersBoard, move: Move) -> tuple[int, int, int]:
    nc = len(move.captures)
    cap_bucket = nc if nc < 4 else 4
    return (move_chamber(board, move), cap_bucket, int(move.promoted))


# =====================================================================
# Agent
# =====================================================================

class MarkovChamberAgent:
    def __init__(self, alpha: float = 0.3, gamma: float = 0.95, eps: float = 0.25):
        self.Q: dict[tuple, float] = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def value(self, board, move):
        return self.Q[(state_abs(board), action_sig(board, move))]

    def _candidates(self, board):
        moves = legal_moves(board)
        if not moves: return moves
        max_c = max(len(m.captures) for m in moves)
        cands = [m for m in moves if len(m.captures) == max_c]
        prom = [m for m in cands if m.promoted]
        return prom if prom else cands

    def act(self, board, training):
        cands = self._candidates(board)
        if not cands: return None
        if training and random.random() < self.eps:
            return random.choice(cands)
        return max(cands, key=lambda m: self.value(board, m))

    def update(self, s, a, r, s_next, terminal):
        key = (state_abs(s), action_sig(s, a))
        old = self.Q[key]
        if terminal:
            target = r
        else:
            nxt = self._candidates(s_next)
            if not nxt:
                target = r
            else:
                opp_best = max(self.value(s_next, m) for m in nxt)
                target = r - self.gamma * opp_best
        self.Q[key] = old + self.alpha * (target - old)


def play_episode(agent: MarkovChamberAgent) -> str:
    board = initial_board()
    transitions = []
    plies = 0
    winner = None
    while plies < MAX_PLIES:
        winner = game_winner(board)
        if winner is not None: break
        side = board.turn
        a = agent.act(board, training=True)
        if a is None: break
        s_next = apply_move(board, a)
        transitions.append((side, board, a, s_next))
        board = s_next
        plies += 1
    if winner is None: winner = "draw"
    for side, s, a, s_next in transitions:
        w_next = game_winner(s_next)
        terminal = w_next is not None
        r = 0.0
        if terminal:
            if w_next == side: r = 1.0
            elif w_next is not None and w_next != "draw": r = -1.0
        agent.update(s, a, r, s_next, terminal)
    return winner


def rand_rules(board):
    moves = legal_moves(board)
    return random.choice(moves) if moves else None


def rand_no_rules(board):
    moves = legal_moves_no_force(board)
    return random.choice(moves) if moves else None


def eval_vs(agent, games, side, opp):
    w = d = l = 0
    for _ in range(games):
        b = initial_board()
        plies = 0
        winner = None
        while plies < MAX_PLIES:
            winner = game_winner(b)
            if winner is not None: break
            if b.turn == side:
                m = agent.act(b, training=False)
            else:
                m = opp(b)
            if m is None:
                winner = side
                break
            b = apply_move(b, m)
            plies += 1
        if winner is None: winner = "draw"
        if winner == side: w += 1
        elif winner == "draw": d += 1
        else: l += 1
    return {"w": w, "d": d, "l": l}


def main():
    print("=" * 72)
    print("MARKOV ABSTRACTION  vs  RAW-STATE Q")
    print("=" * 72)
    print()
    print("State (from side-to-move POV):")
    print("  s = (material_bucket [7], kings_bucket [5], phase_bucket [3])")
    print("      max |state| = 105")
    print()
    print("Action (32-chamber lattice + tactics):")
    print("  a = (chamber [32], capture_bucket [5], promoted [2])")
    print("      max |action| = 320")
    print()
    print("Max |Q| ~ 33,600     v1 raw-state |Q| grew to 253,712")
    print()
    print("=" * 72)
    print()

    random.seed(42)
    agent = MarkovChamberAgent(alpha=0.3, gamma=0.95, eps=0.25)
    checkpoints = [25, 100, 300, 700]
    last_vR = None
    for i in range(1, max(checkpoints) + 1):
        play_episode(agent)
        if i % 100 == 0:
            agent.eps = max(0.05, 0.25 * (1.0 - i / max(checkpoints)))
        if i in checkpoints:
            vR = eval_vs(agent, 40, "X", rand_rules)
            vU = eval_vs(agent, 40, "X", rand_no_rules)
            last_vR = vR
            print(f"  ep={i:>4}  |Q|={len(agent.Q):>5}  eps={agent.eps:.2f}   "
                  f"vs random+rules  w/d/l={vR['w']}/{vR['d']}/{vR['l']}   "
                  f"vs random no-rules  w/d/l={vU['w']}/{vU['d']}/{vU['l']}")

    print()
    print("=" * 72)
    print("HEAD-TO-HEAD AT 700 EPISODES")
    print("=" * 72)
    wins = last_vR["w"]
    win_pct = 100 * wins // 40
    v2_summary = f"{wins} ({win_pct}%)"
    compression = 253712 // max(len(agent.Q), 1)
    print(f"{'metric':<32} {'v1 raw':<14} {'v2 Markov':<14}")
    print(f"{'-' * 32} {'-' * 14} {'-' * 14}")
    print(f"{'|Q| entries':<32} {'253,712':<14} {len(agent.Q):<14}")
    print(f"{'vs random+rules wins/40':<32} {'27 (67%)':<14} {v2_summary:<14}")
    print(f"{'Q-table compression':<32} {'1x':<14} {f'{compression}x':<14}")
    print()

    print("--- Learned policy (top-5 highest, bottom-5 lowest Q) ---")
    sorted_Q = sorted(agent.Q.items(), key=lambda x: x[1], reverse=True)
    print("Highest Q values:")
    for (s, a), v in sorted_Q[:5]:
        print(f"  s(mat={s[0]}, k={s[1]}, ph={s[2]})  "
              f"a(chamber={a[0]:>2}, caps={a[1]}, prom={a[2]})  Q={v:+.3f}")
    print("Lowest Q values:")
    for (s, a), v in sorted_Q[-5:]:
        print(f"  s(mat={s[0]}, k={s[1]}, ph={s[2]})  "
              f"a(chamber={a[0]:>2}, caps={a[1]}, prom={a[2]})  Q={v:+.3f}")


if __name__ == "__main__":
    main()
