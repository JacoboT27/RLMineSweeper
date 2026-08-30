"""
Training loop for the DDQN Minesweeper agent.

Runs episodes, stores transitions, calls agent.learn() each step (epsilon decays
inside the agent as it acts), periodically evaluates greedily on a *fixed* set of
boards, saves the best checkpoint, and early-stops via a patience parameter.

Early stopping
--------------
Every `eval_every` episodes we run a greedy evaluation and read a monitored
metric (default: mean return; alternatively win rate). If it improves on the
best-so-far by more than `min_delta`, we save the checkpoint and reset the
clock. If `patience` episodes pass with no improvement, training stops and the
best checkpoint is the one on disk. Because improvement is only checked at eval
points, the effective patience granularity is `eval_every` episodes.

Evaluation uses a fixed list of seeds so every eval scores the *same* boards --
that keeps the improvement signal low-variance, which is what makes patience
meaningful rather than noise-driven.
"""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

from minesweeper_env import MinesweeperEnv
from DDQN import DDQNAgent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_history(history: dict, out_path: str = "training.png") -> str:
    """Save a 2x2 diagnostic figure (win rate, return, epsilon, TD loss) to PNG.
    matplotlib is imported lazily so the module doesn't hard-require it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = history["episode"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.plot(ep, history["eval_win_rate"], marker="o", label="eval (greedy)")
    if any(np.isfinite(history.get("train_win_rate", [np.nan]))):
        ax.plot(ep, history["train_win_rate"], alpha=0.5, label="train (rolling)")
    ax.set_title("win rate"); ax.set_ylim(0, 1); ax.legend()

    ax = axes[0, 1]
    ax.plot(ep, history["eval_return"], marker="o", color="tab:green", label="eval (greedy)")
    if any(np.isfinite(history.get("train_return", [np.nan]))):
        ax.plot(ep, history["train_return"], alpha=0.5, color="tab:olive", label="train (rolling)")
    ax.set_title("mean return"); ax.legend()

    ax = axes[1, 0]
    ax.plot(ep, history["epsilon"], marker="o", color="tab:orange")
    ax.set_title("epsilon (exploration rate)"); ax.set_ylim(0, 1)

    ax = axes[1, 1]
    loss = history.get("loss", [])
    if any(np.isfinite(loss)):
        ax.plot(ep, loss, marker="o", color="tab:red")
        ax.set_yscale("log")
    ax.set_title("mean TD loss (log)")

    for a in axes.flat:
        a.set_xlabel("episode"); a.grid(alpha=0.3)
    if "best_episode" in history:
        for a in axes.flat:
            a.axvline(history["best_episode"], ls="--", color="grey", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def evaluate(env: MinesweeperEnv, agent: DDQNAgent, eval_seeds) -> tuple[float, float, float]:
    """Greedy rollout over fixed seeds. Returns (win_rate, mean_return, mean_len)."""
    wins, returns, lengths = 0, [], []
    for s in eval_seeds:
        obs, info = env.reset(seed=int(s))
        done, ret, steps = False, 0.0, 0
        while not done:
            a = agent.select_action(obs, info["action_mask"], greedy=True)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            steps += 1
            done = term or trunc
        wins += int(info["is_success"])
        returns.append(ret)
        lengths.append(steps)
    return wins / len(eval_seeds), float(np.mean(returns)), float(np.mean(lengths))


def train(env: MinesweeperEnv, agent: DDQNAgent, *,
          total_episodes: int = 50_000,
          eval_every: int = 500,
          eval_episodes: int = 200,
          patience: int = 5_000,
          patience_start: int = 0,
          min_delta: float = 1e-3,
          monitor: str = "return",           # "return" or "win_rate"
          learn_every: int = 1,
          checkpoint_path: str = "policies/best.pt",
          plot_path: str | None = "training_plots/training.png",
          log_every: int = 500,
          verbose: bool = True) -> dict:
    """Train in place. Returns a history dict; best weights are at checkpoint_path."""
    assert monitor in ("return", "win_rate")
    last_path = (checkpoint_path[:-3] + "_last.pt") if checkpoint_path.endswith(".pt") \
        else checkpoint_path + ".last"
    # fixed eval boards, offset far from any training seeds
    eval_seeds = np.arange(eval_episodes) + 1_000_000

    ep_return = deque(maxlen=100)
    ep_length = deque(maxlen=100)
    ep_win = deque(maxlen=100)
    recent_loss = deque(maxlen=1000)

    history: dict[str, list] = {"episode": [], "eval_win_rate": [],
                                "eval_return": [], "eval_len": [], "epsilon": [],
                                "train_win_rate": [], "train_return": [], "loss": []}

    best_metric = -float("inf")
    best_episode = 0
    global_step = 0

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()
        done, ret, steps = False, 0.0, 0
        while not done:
            a = agent.select_action(obs, info["action_mask"], greedy=False)
            next_obs, r, term, trunc, next_info = env.step(a)
            # store `term` (true terminal) as done -- a truncation should still
            # bootstrap from next_obs, so it is NOT flagged done here.
            agent.store(obs, a, r, next_obs, term, next_info["action_mask"])
            obs, info = next_obs, next_info
            if global_step % learn_every == 0:
                loss = agent.learn()
                if loss is not None:
                    recent_loss.append(loss)
            ret += r
            steps += 1
            global_step += 1
            done = term or trunc

        ep_return.append(ret); ep_length.append(steps); ep_win.append(int(info["is_success"]))

        if verbose and episode % log_every == 0:
            loss_str = f"{np.mean(recent_loss):.4f}" if recent_loss else "  n/a"
            print(f"ep {episode:>6} | eps {agent.epsilon():.3f} | "
                  f"train winrate {np.mean(ep_win):.3f} | train ret {np.mean(ep_return):+.2f} | "
                  f"len {np.mean(ep_length):.1f} | loss {loss_str}")

        if episode % eval_every == 0:
            win_rate, mean_ret, mean_len = evaluate(env, agent, eval_seeds)
            metric = mean_ret if monitor == "return" else win_rate
            history["episode"].append(episode)
            history["eval_win_rate"].append(win_rate)
            history["eval_return"].append(mean_ret)
            history["eval_len"].append(mean_len)
            history["epsilon"].append(agent.epsilon())
            history["train_win_rate"].append(float(np.mean(ep_win)) if ep_win else float("nan"))
            history["train_return"].append(float(np.mean(ep_return)) if ep_return else float("nan"))
            history["loss"].append(float(np.mean(recent_loss)) if recent_loss else float("nan"))
            agent.save(last_path)   # rolling "last" checkpoint, always saved

            improved = False
            if episode >= patience_start:
                if best_episode < patience_start:
                    # first evaluation inside the active window sets the baseline,
                    # so an early high-epsilon fluke can never become "best"
                    best_metric, best_episode, improved = metric, episode, True
                elif metric > best_metric + min_delta:
                    best_metric, best_episode, improved = metric, episode, True
                if improved:
                    agent.save(checkpoint_path)

            if verbose:
                if episode < patience_start:
                    flag = "  (warmup; not tracked for early stop)"
                elif improved:
                    flag = "  <-- best (saved)"
                else:
                    flag = f"  (no improve for {episode - best_episode} eps)"
                print(f"  >> EVAL ep {episode}: win_rate {win_rate:.3f} | "
                      f"return {mean_ret:+.2f} | len {mean_len:.1f}{flag}")

            if episode >= patience_start and (episode - best_episode) >= patience:
                if verbose:
                    print(f"\nEarly stop: no improvement in {monitor} for "
                          f"{episode - best_episode} episodes "
                          f"(best {best_metric:+.3f} at ep {best_episode}).")
                break

    history["best_episode"] = best_episode
    history["best_metric"] = best_metric
    history["checkpoint_path"] = checkpoint_path
    if plot_path is not None and history["episode"]:
        try:
            plot_history(history, plot_path)
            if verbose:
                print(f"saved training plots to {plot_path}")
        except Exception as e:
            if verbose:
                print(f"(plotting skipped: {e})")
    return history


if __name__ == "__main__":
    import argparse

    # Each mode maps to the DDQNAgent flags. Cumulative, matching the project's
    # development order: base -> add dueling head -> add prioritized replay on top.
    # (For an isolated PER ablation instead, set "per" to dueling=False.)
    MODES = {
        "base":    dict(dueling=False, prioritized=False),
        "dueling": dict(dueling=True,  prioritized=False),
        "per":     dict(dueling=True,  prioritized=True),
    }

    p = argparse.ArgumentParser(description="Train a DDQN variant to convergence.")
    p.add_argument("--mode", choices=list(MODES), default="base",
                   help="base | dueling | per  (per = dueling + prioritized replay)")
    p.add_argument("--height", type=int, default=9)
    p.add_argument("--width", type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--eps-decay-steps", type=int, default=40_000)
    p.add_argument("--total-episodes", type=int, default=40_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--patience", type=int, default=5_000)
    p.add_argument("--patience-start", type=int, default=6_000)
    p.add_argument("--init-ckpt", default=None,
                   help="optional warm-start (e.g. a 6x6 run, to fine-tune on 9x9)")
    p.add_argument("--checkpoint", default=None, help="default: best_<mode>.pt")
    p.add_argument("--plot", default=None, help="default: training_<mode>.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    flags = MODES[args.mode]
    ckpt = args.checkpoint or f"policies/best_{args.mode}.pt"
    plot = args.plot or f"training_plots/training_{args.mode}.png"

    set_seed(args.seed)
    env = MinesweeperEnv(args.height, args.width, args.mines)
    agent = DDQNAgent(
        board_hw=(args.height, args.width),
        hidden=args.hidden, n_layers=args.n_layers,
        dueling=flags["dueling"], prioritized=flags["prioritized"],
        gamma=0.99, lr=args.lr, batch_size=64, buffer_capacity=100_000,
        eps_start=1.0, eps_end=0.05, eps_decay_steps=args.eps_decay_steps,
        target_sync=1_000, learning_starts=2_000,
    )
    if args.init_ckpt:
        # warm-start (e.g. 6x6 -> 9x9). strict=False lets a plain checkpoint seed a
        # dueling net (body + advantage head load; value stream stays fresh).
        agent.load(args.init_ckpt, weights_only=True, strict=not flags["dueling"])
        agent.act_step = 0     # restart the epsilon schedule for the new stage
        print(f"warm-started from {args.init_ckpt}")

    print(f"mode={args.mode} | dueling={flags['dueling']} prioritized={flags['prioritized']} "
          f"| board {args.height}x{args.width}/{args.mines} -> {ckpt}")

    hist = train(
        env, agent,
        total_episodes=args.total_episodes,
        eval_every=args.eval_every, eval_episodes=args.eval_episodes,
        patience=args.patience, patience_start=args.patience_start,
        min_delta=1e-3, monitor="return",
        checkpoint_path=ckpt, plot_path=plot,
    )
    print(f"\n[{args.mode}] best {hist['best_metric']:.3f} at episode {hist['best_episode']} -> {ckpt}")