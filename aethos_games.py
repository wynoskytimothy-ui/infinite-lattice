"""
aethos_games.py - Three game adapters for the universal AETHOS learner.

Each adapter exposes the SAME interface so one UniversalAgent works for all:

  Tic-tac-toe  - 9 squares, 2 piece types, full state key feasible
  Checkers     - 32 squares, 4 piece types (men/kings x colors), abstracted
  Chess        - 64 squares, 12 piece types, python-chess + abstraction

Common state encoding via prime composites (chain_primes per square).
Common action signature via the lattice chamber idea ((branch, wing) -> id).

Each adapter defines:
  NAME, CHAMBERS
  initial_state(), legal_moves(state), apply_move(state, move),
  winner(state), turn(state),
  abstract_state(state) -> small hashable tuple for Q,
  move_signature(state, move) -> small hashable tuple for Q,
  render(state) -> str

Optional (meet algebra, if applicable):
  winning_move(state), blocking_move(state), fork_move(state)
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess

from aethos_complex_plane import missing_member
from core.primes import chain_primes


# =====================================================================
# TIC-TAC-TOE adapter (uses full state — it's small enough)
# =====================================================================

_TTT_SQUARE_PRIMES = chain_primes(9)
_TTT_POS_TO_PRIME = {i: _TTT_SQUARE_PRIMES[i] for i in range(9)}
_TTT_PRIME_TO_POS = {p: i for i, p in _TTT_POS_TO_PRIME.items()}
_TTT_ALL_PRODUCT = 1
for _p in _TTT_SQUARE_PRIMES:
    _TTT_ALL_PRODUCT *= _p
_TTT_WIN_LINES = (
    (3, 5, 7), (11, 13, 17), (19, 23, 29),
    (3, 11, 19), (5, 13, 23), (7, 17, 29),
    (3, 13, 29), (7, 13, 19),
)
_TTT_WIN_PRODUCTS = tuple(a * p * q for (a, p, q) in _TTT_WIN_LINES)

# meet-degree: how many winning lines each prime appears in
# center prime 13 -> 4, corner primes -> 3, side primes -> 2
_TTT_PRIME_DEGREE = Counter()
for _line in _TTT_WIN_LINES:
    for _p in _line:
        _TTT_PRIME_DEGREE[_p] += 1


class TicTacToeGame:
    NAME = "tic-tac-toe"
    CHAMBERS = 9
    SQUARE_PRIMES = _TTT_SQUARE_PRIMES
    POS_TO_PRIME = _TTT_POS_TO_PRIME
    PRIME_TO_POS = _TTT_PRIME_TO_POS
    ALL_PRIMES = frozenset(_TTT_SQUARE_PRIMES)
    ALL_PRODUCT = _TTT_ALL_PRODUCT
    WIN_LINES = _TTT_WIN_LINES
    WIN_PRODUCTS = _TTT_WIN_PRODUCTS

    @dataclass(frozen=True)
    class State:
        cx: int = 1
        co: int = 1
        turn: str = "X"

    def initial_state(self):
        return self.State()

    def turn(self, state):
        return state.turn

    def winner(self, state):
        for prod in self.WIN_PRODUCTS:
            if state.cx % prod == 0: return "X"
            if state.co % prod == 0: return "O"
        if state.cx * state.co == self.ALL_PRODUCT:
            return "draw"
        return None

    def unused_primes(self, state):
        c = state.cx * state.co
        return tuple(p for p in self.SQUARE_PRIMES if c % p != 0)

    def legal_moves(self, state):
        return list(self.unused_primes(state))

    def apply_move(self, state, move):
        if state.turn == "X":
            return self.State(state.cx * move, state.co, "O")
        return self.State(state.cx, state.co * move, "X")

    def abstract_state(self, state):
        return (state.cx, state.co, state.turn)  # full state

    def move_signature(self, state, move):
        return self.PRIME_TO_POS[move]  # square index 0..8

    def move_prior(self, state, move):
        # structural prior from meet-degree: center 4 > corner 3 > side 2
        return _TTT_PRIME_DEGREE[move]

    # ---- Meet algebra ----
    def _threats(self, c_me, c_opp):
        out = []
        for line in self.WIN_LINES:
            owned = tuple(p for p in line if c_me % p == 0)
            if len(owned) == 2:
                m = int(missing_member(line, owned))
                if c_opp % m != 0:
                    out.append(m)
        return out

    def _me_opp(self, state):
        return (state.cx, state.co) if state.turn == "X" else (state.co, state.cx)

    def winning_move(self, state):
        c_me, c_opp = self._me_opp(state)
        th = self._threats(c_me, c_opp)
        return th[0] if th else None

    def blocking_move(self, state):
        c_me, c_opp = self._me_opp(state)
        th = self._threats(c_opp, c_me)
        return th[0] if th else None

    def fork_move(self, state):
        c_me, c_opp = self._me_opp(state)
        for p in self.unused_primes(state):
            c_new = c_me * p
            if len(self._threats(c_new, c_opp)) >= 2 and len(self._threats(c_opp, c_new)) == 0:
                return p
        return None

    def safe_moves(self, state):
        out = []
        for p in self.unused_primes(state):
            nxt = self.apply_move(state, p)
            if self.winner(nxt) is not None:
                out.append(p); continue
            if self.winning_move(nxt) is not None: continue
            if self.fork_move(nxt) is not None: continue
            out.append(p)
        return out

    def render(self, state):
        rows = []
        for r in range(3):
            cells = []
            for c in range(3):
                p = self.POS_TO_PRIME[r * 3 + c]
                if state.cx % p == 0: cells.append(" X ")
                elif state.co % p == 0: cells.append(" O ")
                else: cells.append(f"{p:>2} ")
            rows.append("|".join(cells))
        return "\n-----------\n".join(rows)


# =====================================================================
# CHECKERS adapter
# =====================================================================

_CHK_SQUARE_PRIMES = chain_primes(32)
_CHK_PRIME_TO_POS = {p: i for i, p in enumerate(_CHK_SQUARE_PRIMES)}
_CHK_ALL_PRODUCT = 1
for _p in _CHK_SQUARE_PRIMES:
    _CHK_ALL_PRODUCT *= _p


class CheckersGame:
    NAME = "checkers"
    CHAMBERS = 32  # 4 branches * 8 wings
    SQUARE_PRIMES = _CHK_SQUARE_PRIMES
    PRIME_TO_POS = _CHK_PRIME_TO_POS
    ALL_PRODUCT = _CHK_ALL_PRODUCT

    @staticmethod
    def pos_to_rc(pos):
        row = pos // 4
        col_in_row = pos % 4
        col = 2 * col_in_row + (1 if row % 2 == 0 else 0)
        return row, col

    @staticmethod
    def rc_to_pos(row, col):
        if not (0 <= row < 8 and 0 <= col < 8): return None
        if (row + col) % 2 == 0: return None
        return row * 4 + col // 2

    DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    @classmethod
    def _build_tables(cls):
        steps, jumps = [], []
        for pos in range(32):
            r, c = cls.pos_to_rc(pos)
            s, j = [], []
            for dr, dc in cls.DIRS:
                over = cls.rc_to_pos(r + dr, c + dc)
                s.append(over)
                if over is None:
                    j.append((None, None))
                else:
                    land = cls.rc_to_pos(r + 2 * dr, c + 2 * dc)
                    j.append((over, land))
            steps.append(tuple(s))
            jumps.append(tuple(j))
        return tuple(steps), tuple(jumps)

    STEPS = None
    JUMPS = None

    @classmethod
    def _init_tables(cls):
        if cls.STEPS is None:
            cls.STEPS, cls.JUMPS = cls._build_tables()

    def __init__(self):
        self._init_tables()

    @dataclass(frozen=True)
    class State:
        xm: int = 1
        xk: int = 1
        om: int = 1
        ok: int = 1
        turn: str = "X"

    @dataclass(frozen=True)
    class Move:
        source: int
        path: tuple
        captures: tuple
        promoted: bool

        @property
        def dest(self): return self.path[-1]
        @property
        def is_capture(self): return len(self.captures) > 0

    @staticmethod
    def _move_dirs(side, is_king):
        if is_king: return (0, 1, 2, 3)
        return (0, 1) if side == "X" else (2, 3)

    @classmethod
    def _back_rank(cls, side, pos):
        r, _ = cls.pos_to_rc(pos)
        return (r == 0) if side == "X" else (r == 7)

    def initial_state(self):
        om = 1
        for pos in range(12): om *= self.SQUARE_PRIMES[pos]
        xm = 1
        for pos in range(20, 32): xm *= self.SQUARE_PRIMES[pos]
        return self.State(xm=xm, xk=1, om=om, ok=1, turn="X")

    def turn(self, state):
        return state.turn

    def _occupied(self, state):
        return state.xm * state.xk * state.om * state.ok

    def _piece_at(self, state, pos):
        p = self.SQUARE_PRIMES[pos]
        if state.xm % p == 0: return ("X", False)
        if state.xk % p == 0: return ("X", True)
        if state.om % p == 0: return ("O", False)
        if state.ok % p == 0: return ("O", True)
        return None

    def _is_empty(self, state, pos):
        return self._occupied(state) % self.SQUARE_PRIMES[pos] != 0

    def _piece_counts(self, state):
        def c(x): return sum(1 for p in self.SQUARE_PRIMES if x % p == 0)
        return c(state.xm), c(state.xk), c(state.om), c(state.ok)

    def _find_jumps(self, state, source, side, is_king, current, path, captured):
        visited = {source} | set(path)
        cap_pos = {self.PRIME_TO_POS[p] for p in captured}
        results = []
        for di in self._move_dirs(side, is_king):
            over, land = self.JUMPS[current][di]
            if over is None or land is None: continue
            op = self.SQUARE_PRIMES[over]
            if op in captured: continue
            pover = self._piece_at(state, over)
            if pover is None or pover[0] == side: continue
            if land not in visited and land not in cap_pos and not self._is_empty(state, land):
                continue
            new_path = path + (land,)
            new_caps = captured + (op,)
            promoted = (not is_king) and self._back_rank(side, land)
            if promoted:
                results.append(self.Move(source, new_path, new_caps, True))
            else:
                sub = self._find_jumps(state, source, side, is_king, land, new_path, new_caps)
                if sub: results.extend(sub)
                else: results.append(self.Move(source, new_path, new_caps, False))
        return results

    def legal_moves(self, state):
        side = state.turn
        caps, quiet = [], []
        for pos in range(32):
            piece = self._piece_at(state, pos)
            if piece is None or piece[0] != side: continue
            _, is_king = piece
            j = self._find_jumps(state, pos, side, is_king, pos, (), ())
            if j: caps.extend(j)
            else:
                for di in self._move_dirs(side, is_king):
                    tgt = self.STEPS[pos][di]
                    if tgt is None or not self._is_empty(state, tgt): continue
                    promoted = (not is_king) and self._back_rank(side, tgt)
                    quiet.append(self.Move(pos, (tgt,), (), promoted))
        return caps if caps else quiet

    def apply_move(self, state, move):
        side = state.turn
        sp = self.SQUARE_PRIMES[move.source]
        dp = self.SQUARE_PRIMES[move.dest]
        xm, xk, om, ok = state.xm, state.xk, state.om, state.ok
        if side == "X":
            if xk % sp == 0: wk = True; xk //= sp
            else: wk = False; xm //= sp
        else:
            if ok % sp == 0: wk = True; ok //= sp
            else: wk = False; om //= sp
        bk = wk or move.promoted
        if side == "X":
            if bk: xk *= dp
            else: xm *= dp
        else:
            if bk: ok *= dp
            else: om *= dp
        for cap in move.captures:
            if side == "X":
                if om % cap == 0: om //= cap
                else: ok //= cap
            else:
                if xm % cap == 0: xm //= cap
                else: xk //= cap
        return self.State(xm=xm, xk=xk, om=om, ok=ok, turn="O" if side == "X" else "X")

    def winner(self, state):
        if state.xm == 1 and state.xk == 1: return "O"
        if state.om == 1 and state.ok == 1: return "X"
        if not self.legal_moves(state):
            return "O" if state.turn == "X" else "X"
        return None

    def abstract_state(self, state):
        xm, xk, om, ok = self._piece_counts(state)
        me = (xm + 2 * xk) if state.turn == "X" else (om + 2 * ok)
        opp = (om + 2 * ok) if state.turn == "X" else (xm + 2 * xk)
        me_k = xk if state.turn == "X" else ok
        opp_k = ok if state.turn == "X" else xk
        mat = me - opp
        if   mat <= -10: mb = 0
        elif mat <= -5:  mb = 1
        elif mat <= -1:  mb = 2
        elif mat == 0:   mb = 3
        elif mat <= 4:   mb = 4
        elif mat <= 9:   mb = 5
        else:            mb = 6
        kd = me_k - opp_k
        if   kd <= -2: kb = 0
        elif kd == 0:  kb = 1
        else:          kb = 2
        total = xm + xk + om + ok
        if total >= 20: pb = 0
        elif total >= 10: pb = 1
        else: pb = 2
        return (mb, kb, pb)

    def move_signature(self, state, move):
        sp = self.SQUARE_PRIMES[move.source]
        side = state.turn
        if side == "X":
            is_king = (state.xk % sp == 0)
            branch = 2 if is_king else 1
        else:
            is_king = (state.ok % sp == 0)
            branch = 4 if is_king else 3
        sr, sc = self.pos_to_rc(move.source)
        fr, fc = self.pos_to_rc(move.path[0])
        if fr < sr and fc < sc: di = 0
        elif fr < sr and fc > sc: di = 1
        elif fr > sr and fc < sc: di = 2
        else: di = 3
        wing = (di + 1) + (4 if move.is_capture else 0)
        chamber = (branch - 1) * 8 + wing
        nc = len(move.captures)
        return (chamber, nc if nc < 4 else 4, int(move.promoted))

    def move_prior(self, state, move):
        # prefer longer capture chains; promotion as secondary
        return (len(move.captures), int(move.promoted))

    def render(self, state):
        rows = []
        for r in range(8):
            cells = []
            for c in range(8):
                if (r + c) % 2 == 0:
                    cells.append(" . ")
                else:
                    pos = self.rc_to_pos(r, c)
                    p = self._piece_at(state, pos)
                    if p is None: cells.append(" - ")
                    elif p == ("X", False): cells.append(" x ")
                    elif p == ("X", True): cells.append(" X ")
                    elif p == ("O", False): cells.append(" o ")
                    else: cells.append(" O ")
            rows.append("".join(cells))
        return "\n".join(rows)


# =====================================================================
# CHESS adapter (python-chess wrapper)
# =====================================================================

class ChessGame:
    NAME = "chess"
    CHAMBERS = 48  # 6 piece types * 8 direction/capture variants
    PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                   chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

    @dataclass(frozen=True)
    class State:
        fen: str  # python-chess board state as FEN

    def initial_state(self):
        return self.State(fen=chess.Board().fen())

    def _board(self, state):
        return chess.Board(state.fen)

    def turn(self, state):
        b = self._board(state)
        return "X" if b.turn == chess.WHITE else "O"  # X = white

    def legal_moves(self, state):
        return list(self._board(state).legal_moves)

    def apply_move(self, state, move):
        b = self._board(state)
        b.push(move)
        return self.State(fen=b.fen())

    def winner(self, state):
        b = self._board(state)
        if b.is_checkmate():
            return "X" if b.turn == chess.BLACK else "O"  # the side to move is mated
        if b.is_stalemate() or b.is_insufficient_material() or b.is_fivefold_repetition():
            return "draw"
        if b.halfmove_clock >= 100:  # 50-move rule
            return "draw"
        return None

    def _material(self, b, color):
        s = 0
        for pt, v in self.PIECE_VALUE.items():
            s += v * len(b.pieces(pt, color))
        return s

    def abstract_state(self, state):
        b = self._board(state)
        me = chess.WHITE if b.turn == chess.WHITE else chess.BLACK
        opp = not me
        mat = self._material(b, me) - self._material(b, opp)
        if   mat <= -8: mb = 0
        elif mat <= -3: mb = 1
        elif mat == 0:  mb = 2
        elif mat <= 3:  mb = 3
        else:           mb = 4
        # piece-count bucket
        total = sum(len(b.pieces(pt, c)) for pt in self.PIECE_VALUE for c in (chess.WHITE, chess.BLACK))
        if total >= 24: pb = 0
        elif total >= 12: pb = 1
        else: pb = 2
        in_check = int(b.is_check())
        return (mb, pb, in_check)

    def move_signature(self, state, move):
        b = self._board(state)
        piece = b.piece_at(move.from_square)
        if piece is None:
            piece_type = 1
        else:
            piece_type = piece.piece_type  # 1..6
        is_cap = int(b.is_capture(move))
        is_check = 0
        b2 = b.copy()
        b2.push(move)
        is_check = int(b2.is_check())
        is_prom = int(move.promotion is not None)
        # chamber = piece_type-1 of 6 * 8 wings encoding (cap, check, prom)
        wing = is_cap * 4 + is_check * 2 + is_prom
        chamber = (piece_type - 1) * 8 + wing + 1  # 1..48
        return (chamber, is_cap, is_prom)

    def move_prior(self, state, move):
        # prefer capturing high-value pieces > giving check > promotion
        b = self._board(state)
        cap_value = 0
        if b.is_capture(move):
            victim = b.piece_at(move.to_square)
            if victim is not None:
                cap_value = self.PIECE_VALUE.get(victim.piece_type, 0)
            elif b.is_en_passant(move):
                cap_value = self.PIECE_VALUE[chess.PAWN]
        b2 = b.copy()
        b2.push(move)
        gives_check = int(b2.is_check())
        is_prom = int(move.promotion is not None)
        return (cap_value, gives_check, is_prom)

    def render(self, state):
        return str(self._board(state))


# =====================================================================
# helper: side-perspective reward
# =====================================================================

def reward(winner: str | None, mover: str) -> float:
    if winner is None or winner == "draw":
        return 0.0
    return 1.0 if winner == mover else -1.0
