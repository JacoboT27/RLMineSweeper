"""Correctness tests for MinesweeperGame / MinesweeperEnv.

Run: python test_minesweeper.py
Verifies the bug-prone bits (first-click-safe, flood fill, counts, win/loss)
and Gymnasium API compliance, so the RL layer can trust this foundation.
"""
import numpy as np
from minesweeper_env import MinesweeperGame, MinesweeperEnv


def brute_force_counts(mines: np.ndarray) -> np.ndarray:
    """Independent O(H*W*9) reference implementation of adjacency counts."""
    H, W = mines.shape
    ref = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            s = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and mines[nr, nc]:
                        s += 1
            ref[r, c] = s
    return ref


def test_determinism():
    e1 = MinesweeperEnv(9, 9, 10)
    e2 = MinesweeperEnv(9, 9, 10)
    e1.reset(seed=123); e2.reset(seed=123)
    e1.step(40); e2.step(40)   # same first click -> same layout
    assert np.array_equal(e1.game.mines, e2.game.mines)
    print("PASS  determinism: same seed + same action -> identical layout")


def test_first_click_safe():
    misses = 0
    for seed in range(500):
        env = MinesweeperEnv(9, 9, 10, first_click_safe=True)
        env.reset(seed=seed)
        action = int(env.np_random.integers(0, 81))
        _, reward, terminated, _, _ = env.step(action)
        if terminated and reward < 0:
            misses += 1
    assert misses == 0, f"first click detonated {misses}/500 times"
    print("PASS  first-click-safe: 0/500 first clicks detonated")


def test_first_click_opens_region():
    """An interior first click must reveal at least its 3x3 neighbourhood (>=9)."""
    for seed in range(300):
        env = MinesweeperEnv(16, 16, 40, first_click_safe=True)
        env.reset(seed=seed)
        _, _, _, _, info = env.step(8 * 16 + 8)   # interior cell (8, 8)
        assert info["newly_revealed"] >= 9, \
            f"first click opened only {info['newly_revealed']} cells (seed {seed})"
    print("PASS  first click opens a region: interior click reveals >=9 cells (300 boards)")


def test_counts_match_reference():
    for seed in range(200):
        env = MinesweeperEnv(12, 12, 20)
        env.reset(seed=seed)
        env.step(0)  # force placement
        assert np.array_equal(np.asarray(env.game.counts),
                              brute_force_counts(env.game.mines))
    print("PASS  adjacency counts match independent reference (200 boards)")


def test_flood_reveals_connected_region():
    """A revealed 0-cell must have all 8 neighbours revealed too."""
    for seed in range(200):
        env = MinesweeperEnv(12, 12, 15)
        env.reset(seed=seed)
        env.step(int(env.np_random.integers(0, 144)))
        g = env.game
        H, W = g.height, g.width
        for r in range(H):
            for c in range(W):
                if g.revealed[r, c] and g.counts[r, c] == 0:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                assert g.revealed[nr, nc], "0-cell neighbour not revealed"
    print("PASS  flood fill: every revealed 0-cell has all neighbours revealed")


def test_never_reveal_mine_when_safe():
    for seed in range(200):
        env = MinesweeperEnv(10, 10, 12)
        env.reset(seed=seed)
        done = False
        while not done:
            mask = env.game.legal_mask()
            legal = np.flatnonzero(mask)
            # pick a legal cell that is NOT a mine to simulate perfect play
            safe = [a for a in legal if not env.game.mines[divmod(a, 10)]]
            if not safe:
                break
            _, reward, terminated, truncated, _ = env.step(int(safe[0]))
            done = terminated or truncated
            # revealed set must never include a mine
            assert not (env.game.revealed & env.game.mines).any()
        if env.game.won:
            # winning means exactly all safe cells revealed
            assert env.game.revealed.sum() == 100 - 12
    print("PASS  safe-only play never reveals a mine; win reveals all safe cells")


def test_win_and_loss_rewards():
    # loss
    env = MinesweeperEnv(5, 5, 3)
    env.reset(seed=7)
    env.step(12)  # place mines (12 safe)
    mine_action = int(np.flatnonzero(env.game.mines.reshape(-1))[0])
    _, reward, terminated, _, info = env.step(mine_action)
    assert terminated and reward == env.mine_penalty and not info["is_success"]
    # win: reveal every safe cell
    env2 = MinesweeperEnv(5, 5, 3)
    env2.reset(seed=7)
    env2.step(12)
    safe_cells = np.flatnonzero(~env2.game.mines.reshape(-1))
    last_reward, won = None, False
    for a in safe_cells:
        if env2.game.revealed[divmod(int(a), 5)]:
            continue
        _, last_reward, terminated, _, info = env2.step(int(a))
        if terminated:
            won = info["is_success"]
            break
    assert won and last_reward == env2.win_reward
    print("PASS  reward: mine -> mine_penalty+terminate; clearing safe cells -> win_reward")


def test_observation_shape_and_onehot():
    env = MinesweeperEnv(8, 8, 10)
    obs, info = env.reset(seed=1)
    assert obs.shape == (11, 8, 8) and obs.dtype == np.float32
    obs, *_ , info = env.step(20)
    # channels 0..9 are a partition: exactly one is 1.0 per cell
    stacked = obs[0:10].sum(axis=0)
    assert np.allclose(stacked, 1.0), "channels 0-9 are not a clean one-hot"
    # density channel constant across the grid, within [0,1]
    assert np.allclose(obs[10], obs[10].flat[0]) and 0.0 <= obs[10].flat[0] <= 1.0
    print("PASS  observation: shape (11,H,W), channels 0-9 one-hot, density in [0,1]")


def test_action_mask_matches_hidden():
    env = MinesweeperEnv(8, 8, 10)
    obs, info = env.reset(seed=2)
    obs, *_ , info = env.step(30)
    hidden_from_obs = obs[0].reshape(-1).astype(bool)
    assert np.array_equal(hidden_from_obs, info["action_mask"])
    print("PASS  action_mask in info equals the hidden channel (legal == unrevealed)")


def test_illegal_action_is_noop():
    env = MinesweeperEnv(6, 6, 5)
    env.reset(seed=3)
    _, _, _, _, _ = env.step(0)
    revealed_before = env.game.revealed.copy()
    # cell 0 is now revealed -> re-clicking it is illegal
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["illegal"] and reward == env.illegal_penalty and not terminated
    assert np.array_equal(revealed_before, env.game.revealed), "illegal move changed state"
    print("PASS  illegal (re-click) is a penalised no-op, state unchanged")


def test_gymnasium_api_compliance():
    from gymnasium.utils.env_checker import check_env
    env = MinesweeperEnv(9, 9, 10)
    check_env(env, skip_render_check=False)
    print("PASS  gymnasium env_checker: full API compliance")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")