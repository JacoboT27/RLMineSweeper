"""
Play Minesweeper yourself in the terminal, against the same environment the
agent trains on. Optionally RECORD your games as transitions for imitation /
offline learning (see --record).

Usage:
    python play.py                     # default 9x9, 10 mines
    python play.py 16 16 40            # custom: height width mines
    python play.py --record           # save your games to demos/ as .npz
    python play.py --record --demo-dir mydemos

Commands each turn:
    r c          reveal cell at row r, col c   (0-indexed)   e.g.  3 4
    f r c        toggle a flag on cell (r, c)                e.g.  f 3 4
    q            quit

Recording notes:
    Each finished (or quit) game is written atomically as one demos/game_XXXXX.npz
    holding (obs, act, rew, next_obs, done, next_mask) -- exactly the tuple the
    DDQN ReplayBuffer stores -- plus a `won` flag. Quitting only loses the current
    in-progress game; everything already saved is safe, so you can resume later
    and it continues numbering. Flags are a human aid and are NOT transitions
    (they don't call env.step), so only reveals are recorded.
"""
import os
import re
import glob
import argparse

import numpy as np
from minesweeper_env import MinesweeperEnv


class DemoRecorder:
    """Appends each game as its own .npz in demo_dir, resuming numbering."""

    def __init__(self, demo_dir: str):
        self.dir = demo_dir
        os.makedirs(demo_dir, exist_ok=True)
        existing = glob.glob(os.path.join(demo_dir, "game_*.npz"))
        nums = [int(re.search(r"game_(\d+)", os.path.basename(f)).group(1)) for f in existing]
        self.next_idx = (max(nums) + 1) if nums else 0
        self.total = 0
        self.games = 0

    def save_game(self, transitions, won: bool):
        if not transitions:
            return None, 0
        obs       = np.array([t[0] for t in transitions], dtype=np.float32)   # (T,11,H,W)
        act       = np.array([t[1] for t in transitions], dtype=np.int64)     # (T,)
        rew       = np.array([t[2] for t in transitions], dtype=np.float32)   # (T,)
        next_obs  = np.array([t[3] for t in transitions], dtype=np.float32)   # (T,11,H,W)
        done      = np.array([t[4] for t in transitions], dtype=np.float32)   # (T,)
        next_mask = np.array([t[5] for t in transitions], dtype=bool)         # (T,H*W)
        path = os.path.join(self.dir, f"game_{self.next_idx:05d}.npz")
        np.savez_compressed(path, obs=obs, act=act, rew=rew, next_obs=next_obs,
                            done=done, next_mask=next_mask, won=np.array(int(won)))
        self.next_idx += 1
        self.total += len(transitions)
        self.games += 1
        return path, len(transitions)


def render_with_coords(env, flags, reveal_all: bool = False) -> str:
    g = env.game
    H, W = g.height, g.width
    pad = max(2, len(str(max(H, W) - 1)))
    header = " " * (pad + 1) + " ".join(f"{c:>{pad}}" for c in range(W))
    sep = " " * (pad + 1) + "-" * (len(header) - (pad + 1))
    lines = [header, sep]
    for r in range(H):
        cells = []
        for c in range(W):
            if reveal_all and g.mines[r, c]:
                glyph = "X" if g.exploded == (r, c) else "*"
            elif reveal_all and (r, c) in flags:
                glyph = "!"
            elif g.exploded == (r, c):
                glyph = "X"
            elif (r, c) in flags and not g.revealed[r, c]:
                glyph = "F"
            elif not g.revealed[r, c]:
                glyph = "."
            elif g.counts[r, c] == 0:
                glyph = " "
            else:
                glyph = str(int(g.counts[r, c]))
            cells.append(f"{glyph:>{pad}}")
        lines.append(f"{r:>{pad}}|" + " ".join(cells))
    return "\n".join(lines)


def parse(raw: str, H: int, W: int):
    raw = raw.strip().lower()
    if raw in ("q", "quit", "exit"):
        return "quit", None
    parts = raw.replace(",", " ").split()
    is_flag = parts and parts[0] in ("f", "flag")
    if is_flag:
        parts = parts[1:]
    if len(parts) != 2:
        return "bad", None
    try:
        r, c = int(parts[0]), int(parts[1])
    except ValueError:
        return "bad", None
    if not (0 <= r < H and 0 <= c < W):
        return "oob", None
    return ("flag", (r, c)) if is_flag else ("reveal", r * W + c)


def play_one(env, recorder=None) -> None:
    obs, info = env.reset()
    H, W = env.height, env.width
    flags: set[tuple[int, int]] = set()
    transitions = []

    def finalize():
        if recorder is not None and transitions:
            path, n = recorder.save_game(transitions, bool(env.game.won))
            print(f"  [recorded {n} transitions -> {path}]")

    print(f"\n{H}x{W} board, {env.n_mines} mines.")
    print("  Commands:  'r c' reveal   |   'f r c' toggle flag   |   'q' quit\n")
    while True:
        remaining = env.n_mines - len(flags)
        print(render_with_coords(env, flags))
        print(f"  mines remaining: {remaining:>2} / {env.n_mines}   "
              f"(flags placed: {len(flags)})   |   safe cells left to clear: "
              f"{H*W - env.n_mines - int(env.game.revealed.sum())}")
        try:
            raw = input("  your move > ")
        except (EOFError, KeyboardInterrupt):
            finalize(); print("\n  bye."); return

        kind, payload = parse(raw, H, W)
        if kind == "quit":
            finalize(); print("  bye."); return
        if kind == "bad":
            print("  ? use 'r c' to reveal or 'f r c' to flag, e.g. '3 4' or 'f 3 4'.\n")
            continue
        if kind == "oob":
            print(f"  out of bounds. rows 0-{H-1}, cols 0-{W-1}.\n")
            continue
        if kind == "flag":
            cell = payload
            if env.game.revealed[cell]:
                print("  can't flag a revealed cell.\n")
                continue
            flags.discard(cell) if cell in flags else flags.add(cell)
            continue

        # reveal
        action = payload
        r, c = divmod(action, W)
        if (r, c) in flags:
            print("  that cell is flagged -- unflag it first (f r c).\n")
            continue

        prev_obs = obs
        prev_mask = info["action_mask"]
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("illegal"):
            print("  that cell is already revealed -- pick another.\n")
            continue

        # record the transition: same tuple the DDQN ReplayBuffer stores
        if recorder is not None:
            transitions.append((prev_obs.copy(), int(action), float(reward),
                                obs.copy(), bool(terminated), info["action_mask"].copy()))

        print()
        if terminated and info["is_success"]:
            print(render_with_coords(env, flags, reveal_all=True))
            print("\n  *** cleared it -- you win! ***\n"); finalize(); return
        if terminated:
            print(render_with_coords(env, flags, reveal_all=True))
            print("\n  *** BOOM -- you hit a mine. ***\n"); finalize(); return
        if truncated:
            print("\n  (episode truncated at max_steps)\n"); finalize(); return


def main():
    p = argparse.ArgumentParser(description="Play Minesweeper; optionally record games.")
    p.add_argument("board", nargs="*", type=int, help="optional: height width mines")
    p.add_argument("--record", action="store_true", help="save games as transitions to --demo-dir")
    p.add_argument("--demo-dir", default="demos", help="folder for recorded .npz games")
    args = p.parse_args()

    H, W, mines = (args.board if len(args.board) == 3 else (9, 9, 10))
    env = MinesweeperEnv(H, W, mines, render_mode=None)

    recorder = DemoRecorder(args.demo_dir) if args.record else None
    if recorder is not None:
        print(f"  recording ON -> {args.demo_dir}/  (next file: game_{recorder.next_idx:05d}.npz)")

    while True:
        play_one(env, recorder)
        try:
            again = input("  play again? (y/n) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if again not in ("y", "yes"):
            break

    if recorder is not None:
        print(f"\n  session: recorded {recorder.total} transitions across {recorder.games} games "
              f"into {args.demo_dir}/  (now {recorder.next_idx} games total)")


if __name__ == "__main__":
    main()