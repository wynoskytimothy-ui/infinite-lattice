"""
aethos_universal_learner.py - one Q-learner, any game.

Pattern: state and action are encoded into the universal Q-table key as
    (game_name, abstract_state, action_signature)

The agent picks moves by:
  1. Meet-algebra priority (if game provides it):
       winning_move -> blocking_move -> fork_move
     (these are derived from the formula, not heuristics)
  2. Among remaining moves, eps-greedy with Q-table

Same agent instance is reused across games. The Q-table grows additively;
game_name in the key prevents key collisions, but features like
"material balance" naturally re-occur in similar buckets across games,
so structural transfer is at least possible.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import Protocol


class GameAdapter(Protocol):
    NAME: str
    CHAMBERS: int

    def initial_state(self): ...
    def turn(self, state) -> str: ...
    def winner(self, state) -> str | None: ...
    def legal_moves(self, state) -> list: ...
    def apply_move(self, state, move): ...
    def abstract_state(self, state) -> tuple: ...
    def move_signature(self, state, move) -> tuple: ...
    def render(self, state) -> str: ...


class UniversalAgent:
    """One Q-learner working across any GameAdapter."""

    def __init__(self, alpha: float = 0.3, gamma: float = 0.95, eps: float = 0.2):
        self.Q: dict = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def _key(self, game: GameAdapter, state, move):
        return (game.NAME, game.abstract_state(state), game.move_signature(state, move))

    def value(self, game: GameAdapter, state, move):
        return self.Q[self._key(game, state, move)]

    def _meet_action(self, game: GameAdapter, state):
        """Try meet-algebra forced moves (3-way win/block, 4-way fork)."""
        if hasattr(game, "winning_move"):
            m = game.winning_move(state)
            if m is not None: return m
        if hasattr(game, "blocking_move"):
            m = game.blocking_move(state)
            if m is not None: return m
        if hasattr(game, "fork_move"):
            m = game.fork_move(state)
            if m is not None: return m
        return None

    def _candidates(self, game: GameAdapter, state):
        """Candidate moves after meet-algebra filtering."""
        if hasattr(game, "safe_moves"):
            sm = game.safe_moves(state)
            if sm: return sm
        return game.legal_moves(state)

    def _score(self, game: GameAdapter, state, move):
        v = self.value(game, state, move)
        if hasattr(game, "move_prior"):
            return (v, game.move_prior(state, move))
        return (v, 0)

    def best_action(self, game: GameAdapter, state):
        forced = self._meet_action(game, state)
        if forced is not None: return forced
        cands = self._candidates(game, state)
        if not cands: return None
        return max(cands, key=lambda m: self._score(game, state, m))

    def act(self, game: GameAdapter, state, training: bool):
        forced = self._meet_action(game, state)
        if forced is not None: return forced
        cands = self._candidates(game, state)
        if not cands: return None
        if training and random.random() < self.eps:
            return random.choice(cands)
        return max(cands, key=lambda m: self._score(game, state, m))

    def update(self, game: GameAdapter, s, a, r, s_next, terminal: bool):
        key = self._key(game, s, a)
        old = self.Q[key]
        if terminal:
            target = r
        else:
            nxt = game.legal_moves(s_next)
            if not nxt:
                target = r
            else:
                opp_best = max(self.value(game, s_next, m) for m in nxt)
                target = r - self.gamma * opp_best
        self.Q[key] = old + self.alpha * (target - old)


# ---------------------------------------------------------------------
# Training and evaluation harness
# ---------------------------------------------------------------------

def reward_for(winner: str | None, mover: str) -> float:
    if winner is None or winner == "draw":
        return 0.0
    return 1.0 if winner == mover else -1.0


def play_episode(agent: UniversalAgent, game: GameAdapter, max_plies: int = 300) -> str:
    state = game.initial_state()
    transitions = []
    plies = 0
    winner = None
    while plies < max_plies:
        winner = game.winner(state)
        if winner is not None: break
        side = game.turn(state)
        a = agent.act(game, state, training=True)
        if a is None:
            winner = "O" if side == "X" else "X"
            break
        s_next = game.apply_move(state, a)
        transitions.append((side, state, a, s_next))
        state = s_next
        plies += 1
    if winner is None:
        winner = "draw"
    for side, s, a, s_next in transitions:
        w = game.winner(s_next)
        terminal = w is not None
        r = reward_for(w, side) if terminal else 0.0
        agent.update(game, s, a, r, s_next, terminal)
    return winner


def random_policy(game: GameAdapter, state):
    moves = game.legal_moves(state)
    return random.choice(moves) if moves else None


def evaluate_vs_random(agent: UniversalAgent, game: GameAdapter, games: int,
                       agent_side: str = "X", max_plies: int = 300) -> dict:
    w = d = l = 0
    for _ in range(games):
        state = game.initial_state()
        plies = 0
        winner = None
        while plies < max_plies:
            winner = game.winner(state)
            if winner is not None: break
            if game.turn(state) == agent_side:
                m = agent.best_action(game, state)
            else:
                m = random_policy(game, state)
            if m is None:
                winner = agent_side if game.turn(state) != agent_side else (
                    "O" if agent_side == "X" else "X")
                break
            state = game.apply_move(state, m)
            plies += 1
        if winner is None: winner = "draw"
        if winner == agent_side: w += 1
        elif winner == "draw":   d += 1
        else:                    l += 1
    return {"w": w, "d": d, "l": l}


def train_on_game(agent: UniversalAgent, game: GameAdapter, episodes: int,
                  checkpoints: list[int], eval_games: int = 50,
                  max_plies: int = 300, eps_schedule: bool = True):
    """Train agent on a single game with periodic eval-vs-random checkpoints."""
    print(f"\n--- training on {game.NAME} ({episodes} episodes) ---")
    qs_before = len(agent.Q)
    t0 = time.time()
    last_eval = None
    for i in range(1, episodes + 1):
        play_episode(agent, game, max_plies=max_plies)
        if eps_schedule and i % max(episodes // 30, 1) == 0:
            agent.eps = max(0.05, 0.25 * (1.0 - i / episodes))
        if i in checkpoints:
            vR = evaluate_vs_random(agent, game, eval_games, "X", max_plies=max_plies)
            vRO = evaluate_vs_random(agent, game, eval_games, "O", max_plies=max_plies)
            last_eval = (vR, vRO)
            dt = time.time() - t0
            new_q = len(agent.Q) - qs_before
            print(f"  ep={i:>5}  |Q|+={new_q:>5}  eps={agent.eps:.2f}  ({dt:.1f}s)  "
                  f"vs random  X w/d/l={vR['w']}/{vR['d']}/{vR['l']}   "
                  f"O w/d/l={vRO['w']}/{vRO['d']}/{vRO['l']}")
    return last_eval
