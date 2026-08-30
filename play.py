"""
Play Minesweeper yourself in the terminal, against the same environment the
agent will train on. Useful for getting a feel for the game and for sanity-
checking that the env behaves like real Minesweeper.

Usage:
    python play.py                # default 9x9, 10 mines
    python play.py 16 16 40       # custom: height width mines

Commands each turn:
    r c          reveal cell at row r, col c   (0-indexed)   e.g.  3 4
    f r c        toggle a flag on cell (r, c)                e.g.  f 3 4
    q            quit

Flagging is a human bookkeeping aid ONLY -- it drives the mine counter but is
not part of the RL environment (flags never change the game's outcome, so the
agent doesn't model them). Reveal-only, first click always opens a region.
"""
import sys
from minesweeper_env import MinesweeperEnv


def render_with_coords(env, flags, reveal_all: bool = False) -> str:
    """Board with row/column headers.

    Glyphs:  '.' hidden   'F' flagged (human aid)   ' ' revealed-0   '1'-'8' count
             '*' mine (reveal_all)   'X' the mine you hit   '!' wrong flag (reveal_all)
    """
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
            elif reveal_all and (r, c) in flags:      # flag on a non-mine = wrong
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
    """Return (kind, payload):
       ('quit', None) | ('reveal', action) | ('flag', (r,c)) | ('bad'|'oob', None)"""
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


def play_one(env) -> None:
    obs, info = env.reset()
    H, W = env.height, env.width
    flags: set[tuple[int, int]] = set()
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
            print("\n  bye.")
            return

        kind, payload = parse(raw, H, W)
        if kind == "quit":
            print("  bye.")
            return
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

        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("illegal"):
            print("  that cell is already revealed -- pick another.\n")
            continue
        print()
        if terminated and info["is_success"]:
            print(render_with_coords(env, flags, reveal_all=True))
            print("\n  *** cleared it -- you win! ***\n")
            return
        if terminated:
            print(render_with_coords(env, flags, reveal_all=True))
            print("\n  *** BOOM -- you hit a mine. ***\n")
            return
        if truncated:
            print("\n  (episode truncated at max_steps)\n")
            return


def main():
    H, W, mines = 9, 9, 10
    if len(sys.argv) == 4:
        H, W, mines = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    env = MinesweeperEnv(H, W, mines, render_mode=None)
    while True:
        play_one(env)
        try:
            again = input("  play again? (y/n) ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if again not in ("y", "yes"):
            break


if __name__ == "__main__":
    main()