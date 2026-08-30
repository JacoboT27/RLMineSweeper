"""
Watch a trained DDQN agent play Minesweeper.

Loads a checkpoint, plays greedy games, and renders each step as two panels:
  left  -- the board (revealed numbers, hidden cells, mines shown at game end)
  right -- the agent's Q-map as a heatmap (green = high Q = "safe/valuable",
           red = low Q = "risky"), revealed cells greyed out, and the cell the
           agent is about to click outlined.

Outputs a GIF (watch it play) and a filmstrip PNG (all steps at a glance).

Usage:
    python visualize.py --checkpoint best.pt --height 6 --width 6 --mines 5
    python visualize.py --checkpoint best.pt --seed 42 --out mygame

IMPORTANT: --hidden and --layers must match the architecture the checkpoint was
trained with (the checkpoint stores weights, not the network shape). The shipped
train.py uses hidden=64, layers=4, which are the defaults here.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from minesweeper_env import MinesweeperEnv
from DDQN import DDQNAgent

# classic-ish digit colours for readability
_NUM_COLORS = {1: "#1976d2", 2: "#388e3c", 3: "#d32f2f", 4: "#7b1fa2",
               5: "#795548", 6: "#0097a7", 7: "#212121", 8: "#616161"}


class DQNPlayer:
    """Wraps a trained DDQNAgent. Heatmap = per-cell Q-values."""
    heatmap_title = "Q-map (green = safer)"

    def __init__(self, checkpoint, board_hw, hidden, n_layers, dueling):
        self.agent = DDQNAgent(board_hw=board_hw, hidden=hidden, n_layers=n_layers,
                               dueling=dueling, learning_starts=0, device="cpu")
        self.agent.load(checkpoint)
        self.agent.online.eval()

    def scores(self, obs, mask, H, W):
        with torch.no_grad():
            x = torch.as_tensor(obs, device=self.agent.device).unsqueeze(0)
            q = self.agent.online(x).squeeze(0).cpu().numpy().reshape(H, W)
        return np.where(mask.reshape(H, W), q, np.nan)

    def greedy_action(self, obs, mask):
        return self.agent.select_action(obs, mask, greedy=True)


class PPOPlayer:
    """Wraps a trained PPO ActorCritic. Heatmap = policy probabilities pi(a|s):
    the softmax over the legal cells' logits -- literally 'which cell does the
    policy want to click', which is the PPO analogue of the Q-map."""
    heatmap_title = "policy pi(a|s)  (green = preferred)"

    def __init__(self, checkpoint, board_hw, hidden, n_layers):
        from PPO_agent import ActorCritic            # imported here so DQN use needs no PPO file
        ckpt = torch.load(checkpoint, map_location="cpu")
        cfg = ckpt.get("config", {})                 # architecture saved in the checkpoint
        h = cfg.get("hidden", hidden)
        nl = cfg.get("n_layers", n_layers)
        self.net = ActorCritic(in_channels=11, hidden=h, n_layers=nl)
        self.net.load_state_dict(ckpt["net"])
        self.net.eval()
        self.device = "cpu"

    def _logits(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            logits, _ = self.net(x)
        return logits.squeeze(0)                      # (H*W,)

    def scores(self, obs, mask, H, W):
        logits = self._logits(obs)
        illegal = ~torch.as_tensor(mask, dtype=torch.bool)
        probs = torch.softmax(logits.masked_fill(illegal, float("-inf")), dim=0)
        return np.where(mask.reshape(H, W), probs.cpu().numpy().reshape(H, W), np.nan)

    def greedy_action(self, obs, mask):
        logits = self._logits(obs)
        illegal = ~torch.as_tensor(mask, dtype=torch.bool)
        return int(torch.argmax(logits.masked_fill(illegal, float("-inf"))).item())


# mode -> loader. base/dueling/per all load a DDQNAgent; PER changes the buffer,
# not the network, so a "per" checkpoint is a dueling net (dueling=True). This must
# match how DDQN_train.py's --mode was configured.
MODES = {
    "base":    dict(kind="dqn", dueling=False),
    "dueling": dict(kind="dqn", dueling=True),
    "per":     dict(kind="dqn", dueling=True),
    "ppo":     dict(kind="ppo"),
}


def load_player(args, board_hw):
    m = MODES[args.mode]
    if m["kind"] == "ppo":
        return PPOPlayer(args.checkpoint, board_hw, args.hidden, args.layers)
    return DQNPlayer(args.checkpoint, board_hw, args.hidden, args.layers, m["dueling"])


def run_greedy_episode(env: MinesweeperEnv, player, seed: int):
    """Play one greedy game. Returns (frames, won). Each frame is the pre-click
    state plus the score map and the chosen cell; a final frame shows the outcome."""
    H, W = env.height, env.width
    frames = []
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        q_masked = player.scores(obs, info["action_mask"], H, W)
        a = player.greedy_action(obs, info["action_mask"])
        frames.append({
            "revealed": env.game.revealed.copy(),
            "counts": env.game.counts.copy(),
            "q": q_masked,
            "qtitle": player.heatmap_title,
            "chosen": divmod(a, W),
            "mines": None, "exploded": None, "final": False,
        })
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    frames.append({                                   # outcome frame
        "revealed": env.game.revealed.copy(),
        "counts": env.game.counts.copy(),
        "q": np.full((H, W), np.nan),
        "qtitle": player.heatmap_title,
        "chosen": None,
        "mines": env.game.mines.copy(),
        "exploded": env.game.exploded,
        "final": True,
    })
    return frames, env.game.won


def _draw_board(ax, f):
    H, W = f["revealed"].shape
    rgb = np.zeros((H, W, 3))
    for r in range(H):
        for c in range(W):
            if f["final"] and f["mines"] is not None and f["mines"][r, c]:
                rgb[r, c] = (0.85, 0.16, 0.16) if f["exploded"] == (r, c) else (0.20, 0.20, 0.20)
            elif not f["revealed"][r, c]:
                rgb[r, c] = (0.75, 0.75, 0.78)
            elif f["counts"][r, c] == 0:
                rgb[r, c] = (0.96, 0.96, 0.96)
            else:
                rgb[r, c] = (0.88, 0.90, 0.94)
    ax.imshow(rgb, interpolation="nearest")
    for r in range(H):
        for c in range(W):
            if f["final"] and f["mines"] is not None and f["mines"][r, c]:
                ax.text(c, r, "X" if f["exploded"] == (r, c) else "*",
                        ha="center", va="center", color="white", fontsize=11, fontweight="bold")
            elif f["revealed"][r, c] and f["counts"][r, c] > 0:
                n = int(f["counts"][r, c])
                ax.text(c, r, str(n), ha="center", va="center",
                        color=_NUM_COLORS.get(n, "black"), fontsize=11, fontweight="bold")
    if f["chosen"] is not None:
        r, c = f["chosen"]
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                               edgecolor="#ffb300", lw=3))
    _grid(ax, H, W)
    ax.set_title("board")


def _draw_qmap(ax, f):
    H, W = f["q"].shape
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d9d9d9")                       # revealed/illegal cells
    im = ax.imshow(np.ma.masked_invalid(f["q"]), cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if f["chosen"] is not None:
        r, c = f["chosen"]
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                               edgecolor="#1a1a1a", lw=3))
    _grid(ax, H, W)
    ax.set_title(f.get("qtitle", "Q-map (green = safer)"))


def _grid(ax, H, W):
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.set_xticks([]); ax.set_yticks([])


def _frame_to_image(f, step: int, n_steps: int, won: bool) -> Image.Image:
    fig, (axb, axq) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    _draw_board(axb, f)
    _draw_qmap(axq, f)
    if f["final"]:
        fig.suptitle("WIN" if won else "LOSS", fontweight="bold",
                     color=("#2e7d32" if won else "#c62828"))
    else:
        fig.suptitle(f"step {step + 1}/{n_steps - 1}")
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    arr = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = Image.fromarray(arr[..., :3].copy())
    plt.close(fig)
    return img


def save_gif(frames, won, path, ms=1200):
    imgs = [_frame_to_image(f, i, len(frames), won) for i, f in enumerate(frames)]
    durations = [ms] * (len(imgs) - 1) + [ms * 2]      # linger on the outcome
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durations, loop=0)
    return imgs


def save_filmstrip(imgs, path, cols=4):
    n = len(imgs)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    w, h = imgs[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(path)


def quick_stats(env, player, n_games: int) -> tuple[float, float]:
    wins, lengths = 0, []
    for s in range(n_games):
        obs, info = env.reset(seed=10_000 + s)
        done, steps = False, 0
        while not done:
            a = player.greedy_action(obs, info["action_mask"])
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            done = term or trunc
        wins += int(info["is_success"]); lengths.append(steps)
    return wins / n_games, float(np.mean(lengths))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(MODES), default="base",
                   help="base | dueling | per | ppo  (selects loader + default paths)")
    p.add_argument("--height", type=int, default=9)
    p.add_argument("--width", type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--checkpoint", default=None, help="default: policies/best_<mode>.pt")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="output basename; default: visualization/game_<mode>")
    p.add_argument("--stats", type=int, default=100, help="greedy games for a win-rate readout (0 to skip)")
    args = p.parse_args()

    if args.checkpoint is None:
        args.checkpoint = f"policies/best_{args.mode}.pt"
    out_base = args.out or f"visualization/game_{args.mode}"
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

    env = MinesweeperEnv(args.height, args.width, args.mines)
    player = load_player(args, (args.height, args.width))

    seed = args.seed if args.seed is not None else int(np.random.randint(1_000_000))
    frames, won = run_greedy_episode(env, player, seed)
    imgs = save_gif(frames, won, f"{out_base}.gif")
    save_filmstrip(imgs, f"{out_base}_filmstrip.png")
    print(f"[{args.mode}] seed {seed}: {'WIN' if won else 'LOSS'} in {len(frames)-1} clicks "
          f"-> {out_base}.gif, {out_base}_filmstrip.png")

    if args.stats > 0:
        wr, mlen = quick_stats(env, player, args.stats)
        print(f"[{args.mode}] greedy win rate over {args.stats} games: {wr:.3f} (avg {mlen:.1f} clicks)")


if __name__ == "__main__":
    main()