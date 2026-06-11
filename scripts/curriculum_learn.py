#!/usr/bin/env python3
"""
curriculum_learn.py - one UniversalAgent learns three games in sequence.

Pipeline:
    1. TIC-TAC-TOE  - small, can be solved (gate: never lose to perfect minimax)
    2. CHECKERS     - medium, beat random + improve over self-play
    3. CHESS        - large, beat random via material-grabbing (python-chess)

Same agent instance, same architecture, same prime composite + meet algebra
+ chamber action machinery. Only the game adapter changes.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_games import CheckersGame, ChessGame, TicTacToeGame
from aethos_universal_learner import (
    UniversalAgent,
    evaluate_vs_random,
    train_on_game,
)


def gate_tic_tac_toe(agent: UniversalAgent, game: TicTacToeGame, games: int = 50) -> dict:
    """Minimax gate for tic-tac-toe — must never lose."""
    def minimax_value(state, memo):
        w = game.winner(state)
        if w == "draw": return 0
        if w is not None: return -1
        k = game.abstract_state(state)
        if k in memo: return memo[k]
        best = -2
        for p in game.legal_moves(state):
            v = -minimax_value(game.apply_move(state, p), memo)
            if v > best: best = v
        memo[k] = best
        return best

    def minimax_action(state, memo):
        legals = game.legal_moves(state)
        return min(legals, key=lambda p: minimax_value(game.apply_move(state, p), memo))

    memo: dict = {}
    results = {"X": {"w": 0, "d": 0, "l": 0}, "O": {"w": 0, "d": 0, "l": 0}}
    for side in ("X", "O"):
        for _ in range(games):
            state = game.initial_state()
            winner = None
            while True:
                w = game.winner(state)
                if w is not None: winner = w; break
                if game.turn(state) == side:
                    m = agent.best_action(game, state)
                else:
                    m = minimax_action(state, memo)
                state = game.apply_move(state, m)
            if winner == side: results[side]["w"] += 1
            elif winner == "draw": results[side]["d"] += 1
            else: results[side]["l"] += 1
    return results


def main():
    random.seed(42)
    print("=" * 72)
    print("AETHOS CURRICULUM: one learner, three games of increasing complexity")
    print("=" * 72)

    agent = UniversalAgent(alpha=0.3, gamma=0.95, eps=0.25)

    # ----- 1. TIC-TAC-TOE -----
    print("\n" + "#" * 72)
    print("# STAGE 1: TIC-TAC-TOE")
    print("#" * 72)
    ttt = TicTacToeGame()
    train_on_game(
        agent, ttt,
        episodes=15_000,
        checkpoints=[500, 2500, 7500, 15000],
        eval_games=200,
        max_plies=50,
    )

    print("\n--- TIC-TAC-TOE gate: vs perfect minimax (must never lose) ---")
    agent.eps = 0.0
    gate = gate_tic_tac_toe(agent, ttt, games=50)
    print(f"  as X  w/d/l = {gate['X']['w']}/{gate['X']['d']}/{gate['X']['l']}")
    print(f"  as O  w/d/l = {gate['O']['w']}/{gate['O']['d']}/{gate['O']['l']}")
    ttt_passed = gate["X"]["l"] == 0 and gate["O"]["l"] == 0
    print(f"  LEARNED (zero losses to perfect play): {ttt_passed}")

    # ----- 2. CHECKERS -----
    print("\n" + "#" * 72)
    print("# STAGE 2: CHECKERS")
    print("#" * 72)
    agent.eps = 0.25  # reset exploration for new game
    chk = CheckersGame()
    train_on_game(
        agent, chk,
        episodes=600,
        checkpoints=[50, 200, 400, 600],
        eval_games=30,
        max_plies=200,
    )

    # ----- 3. CHESS -----
    print("\n" + "#" * 72)
    print("# STAGE 3: CHESS")
    print("#" * 72)
    agent.eps = 0.30  # higher exploration for huge state space
    chs = ChessGame()
    train_on_game(
        agent, chs,
        episodes=40,
        checkpoints=[5, 15, 30, 40],
        eval_games=15,
        max_plies=160,
    )

    # ----- summary -----
    print("\n" + "=" * 72)
    print("CURRICULUM SUMMARY")
    print("=" * 72)
    print(f"Final |Q| = {len(agent.Q):,} entries")
    by_game = {"tic-tac-toe": 0, "checkers": 0, "chess": 0}
    for (game_name, _, _) in agent.Q:
        by_game[game_name] = by_game.get(game_name, 0) + 1
    for g, n in by_game.items():
        print(f"  {g:<14}  {n:>6} entries")


if __name__ == "__main__":
    main()
