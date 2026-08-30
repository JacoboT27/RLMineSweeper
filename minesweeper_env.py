"""
Minesweeper environment for reinforcement learning.

Two layers, deliberately separated:

  1. MinesweeperGame  -- pure game logic, no RL concepts. numpy only.
                         Board generation, first-click-safe mine placement,
                         reveal + flood-fill cascade, win/loss detection.
                         This is the bug-prone part; it is tested in isolation.

  2. MinesweeperEnv   -- a gymnasium.Env wrapping the game. Defines the
                         observation/action spaces, the reward function, the
                         legal-action mask, and the broadcast mine-density
                         channel. Algorithm-agnostic: exposes the standard
                         Gymnasium API plus info["action_mask"], so the same
                         env serves DDQN now and PPO later.

Design conventions
------------------
  Board shape:        (H, W)              = (height, width)
  Action:             a in {0, ..., H*W-1}, decoded as row = a // W, col = a % W
  Observation tensor: (C, H, W) with C = 11 channels, dtype float32:
        channel 0        hidden mask      1.0 where a cell is unrevealed
        channels 1..9    count one-hot    channel (1+k) = 1.0 where a revealed
                                          cell shows adjacent-mine count k in 0..8
        channel 10       mine density     broadcast scalar = mines_remaining / n_hidden

  Note on the action mask: in reveal-only Minesweeper the set of legal actions
  is *exactly* the set of unrevealed cells, so the legal-action mask equals
  channel 0. It would be redundant to also carry it as its own input channel,
  so it is NOT in the observation; instead it is handed to the agent via
  info["action_mask"] (shape (H*W,), bool) for masking Q-values / logits.
  That is the correction to the earlier "12-channel" sketch: 11 in, mask in info.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# --------------------------------------------------------------------------- #
#  Layer 1: pure game logic                                                   #
# --------------------------------------------------------------------------- #
class MinesweeperGame:
    """Headless Minesweeper. Holds the true (hidden) state and the reveal logic.

    Attributes
    ----------
    height, width : int
    n_mines       : int
    first_click_safe : bool
        If True, mines are placed lazily on the first reveal, excluding the
        clicked cell AND its 8 neighbours. That forces the clicked cell's
        adjacent count to 0, so the first click always opens a region --
        the standard modern-Minesweeper behaviour. (Falls back to excluding
        only the clicked cell on boards too dense to spare the neighbourhood.)
    mines    : (H, W) bool   -- True where a mine sits (valid once placed)
    counts   : (H, W) int    -- adjacent-mine count per cell (valid once placed)
    revealed : (H, W) bool   -- True where a cell has been opened
    """

    def __init__(self, height: int, width: int, n_mines: int,
                 first_click_safe: bool = True):
        if not (1 <= n_mines < height * width):
            raise ValueError(
                f"n_mines must be in [1, H*W-1] = [1, {height*width-1}], got {n_mines}"
            )
        if first_click_safe and n_mines > height * width - 1:
            raise ValueError("first_click_safe needs at least one non-mine cell.")
        self.height = height
        self.width = width
        self.n_mines = n_mines
        self.first_click_safe = first_click_safe
        self.rng: np.random.Generator = np.random.default_rng()
        self.reset()

    # ---- lifecycle -------------------------------------------------------- #
    def reset(self) -> None:
        H, W = self.height, self.width
        self.mines = np.zeros((H, W), dtype=bool)
        self.counts = np.zeros((H, W), dtype=np.int16)
        self.revealed = np.zeros((H, W), dtype=bool)
        self._mines_placed = False
        self.done = False
        self.won = False
        self.lost = False
        self.exploded: tuple[int, int] | None = None  # cell that ended the game

    # ---- mine placement --------------------------------------------------- #
    def _place_mines(self, first_click: tuple[int, int] | None) -> None:
        """Place n_mines uniformly at random. When first_click_safe and a first
        click is given, exclude the clicked cell and its 8 neighbours so the
        clicked cell has count 0 and the first reveal opens a region. On boards
        too dense to spare the whole neighbourhood, fall back to excluding just
        the clicked cell (opening not guaranteed, but the click stays safe)."""
        H, W = self.height, self.width
        exclude: set[int] = set()
        if first_click is not None and self.first_click_safe:
            r, c = first_click
            nbhd = {(r + dr) * W + (c + dc)
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if 0 <= r + dr < H and 0 <= c + dc < W}
            exclude = nbhd if (H * W - len(nbhd) >= self.n_mines) else {r * W + c}
        candidates = np.array([i for i in range(H * W) if i not in exclude])
        chosen = self.rng.choice(candidates, size=self.n_mines, replace=False)
        flat_mines = np.zeros(H * W, dtype=bool)
        flat_mines[chosen] = True
        self.mines = flat_mines.reshape(H, W)
        self._compute_counts()
        self._mines_placed = True

    def _compute_counts(self) -> None:
        """Adjacent-mine count via 8 shifted additions of a padded mine grid."""
        H, W = self.height, self.width
        m = self.mines.astype(np.int16)
        padded = np.pad(m, 1, mode="constant")
        counts = np.zeros((H, W), dtype=np.int16)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                counts += padded[1 + dr: 1 + dr + H, 1 + dc: 1 + dc + W]
        self.counts = counts

    # ---- the one action: reveal a cell ------------------------------------ #
    def reveal(self, row: int, col: int) -> int:
        """Open cell (row, col). Returns the number of newly revealed cells.

        Mutates game state: sets done/won/lost. Assumes the cell is a legal
        (currently-unrevealed, in-bounds) target; the env layer guards that.
        """
        if self.done:
            raise RuntimeError("reveal() called on a finished game; reset first.")

        if not self._mines_placed:
            self._place_mines(first_click=(row, col))

        if self.mines[row, col]:
            self.lost = True
            self.done = True
            self.exploded = (row, col)
            return 0

        newly = self._flood(row, col)

        # win iff every non-mine cell is now revealed
        if int(self.revealed.sum()) == self.height * self.width - self.n_mines:
            self.won = True
            self.done = True
        return newly

    def _flood(self, row: int, col: int) -> int:
        """Iterative flood fill. A revealed 0-cell expands to its 8 neighbours;
        a revealed non-zero cell stops. Never touches mines, because a 0-cell
        has no adjacent mines by definition."""
        H, W = self.height, self.width
        stack = [(row, col)]
        newly = 0
        while stack:
            r, c = stack.pop()
            if self.revealed[r, c]:
                continue
            self.revealed[r, c] = True
            newly += 1
            if self.counts[r, c] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and not self.revealed[nr, nc]:
                            stack.append((nr, nc))
        return newly

    # ---- convenience ------------------------------------------------------ #
    def legal_mask(self) -> np.ndarray:
        """(H*W,) bool: legal actions == unrevealed cells."""
        return (~self.revealed).reshape(-1)


# --------------------------------------------------------------------------- #
#  Layer 2: Gymnasium wrapper                                                  #
# --------------------------------------------------------------------------- #
class MinesweeperEnv(gym.Env):
    """Gymnasium Minesweeper. Observation (11, H, W) float32, action Discrete(H*W).

    Reward (dense, per click):
        safe click (non-terminal) : +step_reward   (default +0.1)
        winning click             : +win_reward    (default +1.0)
        detonating a mine         : +mine_penalty  (default -1.0), episode ends
        illegal click (masked out): +illegal_penalty (default -1.0), no-op
    Per-click means a click that cascades many cells earns the same as one that
    opens a single cell -- we reward surviving, not cascade luck. The cascade
    size is still reported in info["newly_revealed"] for logging.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(self, height: int = 9, width: int = 9, n_mines: int = 10,
                 first_click_safe: bool = True, render_mode: str | None = None,
                 step_reward: float = 0.1, win_reward: float = 1.0,
                 mine_penalty: float = -1.0, illegal_penalty: float = -1.0,
                 max_steps: int | None = None):
        super().__init__()
        self.game = MinesweeperGame(height, width, n_mines, first_click_safe)
        self.height, self.width, self.n_mines = height, width, n_mines
        self.render_mode = render_mode
        self.step_reward = step_reward
        self.win_reward = win_reward
        self.mine_penalty = mine_penalty
        self.illegal_penalty = illegal_penalty
        # safety net so a non-masking agent can't loop forever on no-op clicks
        self.max_steps = max_steps if max_steps is not None else height * width
        self._elapsed = 0

        self.action_space = spaces.Discrete(height * width)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11, height, width), dtype=np.float32
        )

    # ---- observation / info builders -------------------------------------- #
    def _get_obs(self) -> np.ndarray:
        H, W = self.height, self.width
        obs = np.zeros((11, H, W), dtype=np.float32)
        revealed = self.game.revealed
        obs[0] = (~revealed).astype(np.float32)                 # hidden mask
        counts = self.game.counts
        for k in range(9):                                      # count one-hot
            obs[1 + k] = ((counts == k) & revealed).astype(np.float32)
        n_hidden = int((~revealed).sum())
        density = (self.n_mines / n_hidden) if n_hidden > 0 else 0.0
        obs[10] = np.float32(min(density, 1.0))                 # broadcast scalar
        return obs

    def _get_info(self, newly: int = 0, illegal: bool = False) -> dict:
        n_hidden = int((~self.game.revealed).sum())
        return {
            "action_mask": self.game.legal_mask(),   # (H*W,) bool
            "newly_revealed": newly,
            "mines_remaining": self.n_mines,
            "n_hidden": n_hidden,
            "illegal": illegal,
            "is_success": self.game.won,
        }

    # ---- gym API ---------------------------------------------------------- #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.game.rng = self.np_random          # gymnasium-seeded Generator
        self.game.reset()
        self._elapsed = 0
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, self._get_info()

    def step(self, action: int):
        if self.game.done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        self._elapsed += 1
        row, col = divmod(int(action), self.width)

        # illegal (already-revealed) target: no-op with penalty. Masking should
        # prevent this ever firing; it exists so a masking bug is visible, not silent.
        if self.game.revealed[row, col]:
            truncated = self._elapsed >= self.max_steps
            obs = self._get_obs()
            return obs, self.illegal_penalty, False, truncated, self._get_info(0, illegal=True)

        newly = self.game.reveal(row, col)

        if self.game.lost:
            reward, terminated = self.mine_penalty, True
        elif self.game.won:
            reward, terminated = self.win_reward, True
        else:
            reward, terminated = self.step_reward, False

        truncated = (not terminated) and (self._elapsed >= self.max_steps)
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, self._get_info(newly)

    # ---- rendering -------------------------------------------------------- #
    def render(self):
        if self.render_mode not in ("ansi", "human"):
            return
        H, W = self.height, self.width
        g = self.game
        glyphs = []
        for r in range(H):
            row_glyphs = []
            for c in range(W):
                if g.exploded == (r, c):
                    row_glyphs.append("X")            # detonated mine
                elif not g.revealed[r, c]:
                    row_glyphs.append(".")            # hidden
                elif g.counts[r, c] == 0:
                    row_glyphs.append(" ")            # revealed blank
                else:
                    row_glyphs.append(str(int(g.counts[r, c])))
            glyphs.append(" ".join(row_glyphs))
        out = "\n".join(glyphs)
        if self.render_mode == "human":
            print(out + "\n")
            return
        return out