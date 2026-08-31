"""
Train a DDQN variant from human demonstrations + self-play.

Pipeline (everything we discussed):
  1. Load recorded games from demos/ (play.py --record). Optionally wins-only.
  2. Seed a DEMO-AWARE buffer: the demos live in a PROTECTED partition that is
     never evicted, and every minibatch OVERSAMPLES them (a fixed demo fraction),
     so a small human set stays influential next to millions of self-play steps.
  3. Train online: the agent plays, stores its own transitions in the self-play
     partition, and learns on a demo+self-play mixture -- never pure offline, so
     it doesn't overfit the demos or extrapolate wildly off-distribution.
  4. Imitation is enforced with the DQfD large-margin loss on demo transitions:
     it pushes the demonstrator's action's Q at least `margin` above the others,
     so the demos teach *which cell to click*, not just supply extra data.

--mode selects the network (base=plain, dueling/per=dueling head). The demo-aware
buffer replaces the replay strategy, so PER prioritization is not stacked here --
demonstrations get the sampling priority instead.

Usage:
  python transfer_learning.py --mode base                       # from scratch + demos
  python transfer_learning.py --mode dueling --init-ckpt policies/best_dueling.pt
  python transfer_learning.py --mode base --all-games --margin-lambda 0   # demos as data only
"""
from __future__ import annotations

import os
import glob
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from minesweeper_env import MinesweeperEnv
from DDQN import DDQNAgent
from DDQN_train import evaluate, plot_history, set_seed

MODES = {"base": dict(dueling=False),
         "dueling": dict(dueling=True),
         "per": dict(dueling=True)}   # per uses the dueling net; demo buffer replaces PER

_KEYS = ["obs", "act", "rew", "next_obs", "done", "next_mask"]


# --------------------------------------------------------------------------- #
#  Load demonstrations                                                        #
# --------------------------------------------------------------------------- #
def load_demos(demo_dir: str, wins_only: bool = True):
    files = sorted(glob.glob(os.path.join(demo_dir, "game_*.npz")))
    if not files:
        raise FileNotFoundError(f"No demo games found in {demo_dir}/ (record with play.py --record)")
    acc = {k: [] for k in _KEYS}
    n_games = n_wins = kept = 0
    for f in files:
        d = np.load(f)
        n_games += 1
        won = int(d["won"]); n_wins += won
        if wins_only and not won:
            continue
        for k in _KEYS:
            acc[k].append(d[k])
        kept += 1
    if not acc["act"]:
        raise ValueError(f"{demo_dir}/ has {n_games} games but 0 wins; use --all-games to include losses")
    demos = {k: np.concatenate(acc[k], axis=0) for k in _KEYS}
    n_trans = len(demos["act"])
    print(f"  demos: {n_games} games ({n_wins} wins) | loaded {kept} games -> "
          f"{n_trans} transitions ({'wins only' if wins_only else 'all games'})")
    return demos


# --------------------------------------------------------------------------- #
#  Demo-aware replay buffer                                                    #
# --------------------------------------------------------------------------- #
class DemoReplayBuffer:
    """Protected demo partition (never evicted) + a self-play ring buffer.
    sample() draws a fixed fraction of each minibatch from the demos."""

    def __init__(self, capacity: int, obs_shape, n_actions: int,
                 device: torch.device, demos: dict):
        self.device = device
        self.demos = {k: demos[k] for k in _KEYS}
        self.n_demo = len(demos["act"])
        # self-play ring
        self.capacity = capacity
        self.s = {
            "obs": np.zeros((capacity, *obs_shape), np.float32),
            "next_obs": np.zeros((capacity, *obs_shape), np.float32),
            "act": np.zeros(capacity, np.int64),
            "rew": np.zeros(capacity, np.float32),
            "done": np.zeros(capacity, np.float32),
            "next_mask": np.zeros((capacity, n_actions), bool),
        }
        self.ptr = 0
        self.size = 0

    def push(self, obs, act, rew, next_obs, done, next_mask):
        i = self.ptr
        self.s["obs"][i] = obs
        self.s["next_obs"][i] = next_obs
        self.s["act"][i] = act
        self.s["rew"][i] = rew
        self.s["done"][i] = float(done)
        self.s["next_mask"][i] = next_mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, demo_frac: float):
        if self.size == 0:                       # no self-play yet -> all demos (warmup)
            n_demo, n_self = batch_size, 0
        else:
            n_demo = int(round(batch_size * demo_frac))
            n_self = batch_size - n_demo
        di = np.random.randint(0, self.n_demo, n_demo)
        si = np.random.randint(0, self.size, n_self) if n_self > 0 else np.zeros(0, int)

        def cat(key):
            return np.concatenate([self.demos[key][di], self.s[key][si]], axis=0)

        d = self.device
        batch = (torch.as_tensor(cat("obs"), device=d),
                 torch.as_tensor(cat("act"), device=d),
                 torch.as_tensor(cat("rew"), device=d),
                 torch.as_tensor(cat("next_obs"), device=d),
                 torch.as_tensor(cat("done"), device=d),
                 torch.as_tensor(cat("next_mask"), device=d))
        is_demo = torch.zeros(batch_size, dtype=torch.bool, device=d)
        is_demo[:n_demo] = True
        return batch + (is_demo,)


# --------------------------------------------------------------------------- #
#  One learning step: Double-DQN TD loss (+ DQfD margin loss on demos)         #
# --------------------------------------------------------------------------- #
def learn_step(agent, buffer, *, batch_size, demo_frac, margin, margin_lambda):
    obs, act, rew, next_obs, done, next_mask, is_demo = buffer.sample(batch_size, demo_frac)

    q_all = agent.online(obs)                                   # (B, A), grads on
    q_sa = q_all.gather(1, act.view(-1, 1)).squeeze(1)          # Q(s,a)

    with torch.no_grad():                                       # Double-DQN target
        q_next = agent.online(next_obs).masked_fill(~next_mask, float("-inf"))
        a_star = q_next.argmax(1, keepdim=True)
        q_next_t = agent.target(next_obs).gather(1, a_star).squeeze(1)
        target = rew + agent.gamma * (1.0 - done) * q_next_t
    td_loss = F.smooth_l1_loss(q_sa, target)

    # DQfD large-margin loss on the demo subset: push Q(s, a_demo) at least
    # `margin` above every other legal action. Legal mask at s = obs channel 0.
    margin_loss = torch.zeros((), device=obs.device)
    if margin_lambda > 0 and bool(is_demo.any()):
        qd = q_all[is_demo]                                     # (Bd, A)
        ad = act[is_demo]                                       # (Bd,)
        legal = obs[is_demo][:, 0].reshape(qd.shape[0], -1) > 0.5   # hidden == legal
        bonus = margin * (~F.one_hot(ad, qd.shape[1]).bool()).float()
        max_q = (qd + bonus).masked_fill(~legal, float("-inf")).max(1).values
        margin_loss = (max_q - qd.gather(1, ad.view(-1, 1)).squeeze(1)).mean()

    loss = td_loss + margin_lambda * margin_loss
    agent.optimizer.zero_grad()
    loss.backward()
    if agent.grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(agent.online.parameters(), agent.grad_clip)
    agent.optimizer.step()

    agent.learn_step += 1
    if agent.learn_step % agent.target_sync == 0:
        agent.target.load_state_dict(agent.online.state_dict())
    return float(loss.item()), float(td_loss.item()), float(margin_loss.detach())


# --------------------------------------------------------------------------- #
#  Training loop                                                               #
# --------------------------------------------------------------------------- #
def train_transfer(env, agent, buffer, *, total_episodes, demo_frac, margin,
                   margin_lambda, eval_every, eval_episodes, patience,
                   patience_start, checkpoint_path, plot_path, batch_size,
                   learn_every=1, log_every=500):
    eval_seeds = np.arange(eval_episodes) + 1_000_000
    ep_ret, ep_len, ep_win = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    recent = deque(maxlen=1000)
    history = {"episode": [], "eval_win_rate": [], "eval_return": [], "eval_len": [],
               "epsilon": [], "train_win_rate": [], "train_return": [], "loss": []}
    best_metric, best_episode, gstep = -float("inf"), 0, 0

    for episode in range(1, total_episodes + 1):
        obs, info = env.reset()
        done, ret, steps = False, 0.0, 0
        while not done:
            a = agent.select_action(obs, info["action_mask"], greedy=False)
            next_obs, r, term, trunc, next_info = env.step(a)
            buffer.push(obs, a, r, next_obs, term, next_info["action_mask"])
            obs, info = next_obs, next_info
            if gstep % learn_every == 0:
                loss, _, _ = learn_step(agent, buffer, batch_size=batch_size,
                                        demo_frac=demo_frac, margin=margin,
                                        margin_lambda=margin_lambda)
                recent.append(loss)
            ret += r; steps += 1; gstep += 1
            done = term or trunc
        ep_ret.append(ret); ep_len.append(steps); ep_win.append(int(info["is_success"]))

        if episode % log_every == 0:
            print(f"ep {episode:>6} | eps {agent.epsilon():.3f} | train winrate {np.mean(ep_win):.3f} "
                  f"| train ret {np.mean(ep_ret):+.2f} | loss {np.mean(recent):.4f}")

        if episode % eval_every == 0:
            wr, ret_m, len_m = evaluate(env, agent, eval_seeds)
            metric = ret_m
            for k, v in zip(["episode", "eval_win_rate", "eval_return", "eval_len", "epsilon",
                             "train_win_rate", "train_return", "loss"],
                            [episode, wr, ret_m, len_m, agent.epsilon(),
                             float(np.mean(ep_win)), float(np.mean(ep_ret)),
                             float(np.mean(recent)) if recent else float("nan")]):
                history[k].append(v)
            agent.save(checkpoint_path[:-3] + "_last.pt" if checkpoint_path.endswith(".pt")
                       else checkpoint_path + ".last")
            improved = False
            if episode >= patience_start:
                if best_episode < patience_start or metric > best_metric + 1e-3:
                    best_metric, best_episode, improved = metric, episode, True
                    agent.save(checkpoint_path)
            flag = "  <-- best" if improved else (
                "  (warmup)" if episode < patience_start else f"  (no improve {episode - best_episode})")
            print(f"  >> EVAL ep {episode}: win {wr:.3f} | return {ret_m:+.2f} | len {len_m:.1f}{flag}")
            if episode >= patience_start and (episode - best_episode) >= patience:
                print(f"\nEarly stop: no improvement for {episode - best_episode} eps "
                      f"(best {best_metric:+.3f} at ep {best_episode})")
                break

    history["best_episode"] = best_episode
    if plot_path and history["episode"]:
        try:
            plot_history(history, plot_path)
            print(f"saved plots to {plot_path}")
        except Exception as e:
            print(f"(plotting skipped: {e})")
    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a DDQN variant from human demos + self-play.")
    p.add_argument("--mode", choices=list(MODES), default="base")
    p.add_argument("--demo-dir", default="demos")
    p.add_argument("--all-games", action="store_true", help="use losses too (default: wins only)")
    p.add_argument("--height", type=int, default=9)
    p.add_argument("--width", type=int, default=9)
    p.add_argument("--mines", type=int, default=10)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--demo-frac", type=float, default=0.25, help="fraction of each minibatch from demos")
    p.add_argument("--margin", type=float, default=0.8, help="DQfD margin l(a_E, a)")
    p.add_argument("--margin-lambda", type=float, default=1.0, help="weight of the imitation loss (0 disables)")
    p.add_argument("--init-ckpt", default=None, help="warm-start weights (Variant A); omit for from-scratch")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-decay-steps", type=int, default=40_000)
    p.add_argument("--total-episodes", type=int, default=40_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--patience", type=int, default=5_000)
    p.add_argument("--patience-start", type=int, default=6_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-capacity", type=int, default=100_000)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--plot", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ckpt = args.checkpoint or f"policies/best_{args.mode}_demo.pt"
    plot = args.plot or f"training_plots/training_{args.mode}_demo.png"
    os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(plot) or ".", exist_ok=True)

    set_seed(args.seed)
    demos = load_demos(args.demo_dir, wins_only=not args.all_games)

    env = MinesweeperEnv(args.height, args.width, args.mines)
    dueling = MODES[args.mode]["dueling"]
    agent = DDQNAgent(
        board_hw=(args.height, args.width), hidden=args.hidden, n_layers=args.n_layers,
        dueling=dueling, prioritized=False, gamma=0.99, lr=args.lr,
        batch_size=args.batch_size, buffer_capacity=1,          # own buffer unused
        eps_start=args.eps_start, eps_end=0.05, eps_decay_steps=args.eps_decay_steps,
        target_sync=1_000, learning_starts=0,
    )
    if args.init_ckpt:
        agent.load(args.init_ckpt, weights_only=True, strict=not dueling)
        agent.act_step = 0
        print(f"  warm-started from {args.init_ckpt}")

    buffer = DemoReplayBuffer(args.buffer_capacity, (11, args.height, args.width),
                              args.height * args.width, agent.device, demos)

    print(f"  mode={args.mode} dueling={dueling} | demo_frac={args.demo_frac} "
          f"margin={args.margin} lambda={args.margin_lambda} | {'from scratch' if not args.init_ckpt else 'warm-start'} -> {ckpt}")

    hist = train_transfer(
        env, agent, buffer,
        total_episodes=args.total_episodes, demo_frac=args.demo_frac,
        margin=args.margin, margin_lambda=args.margin_lambda,
        eval_every=args.eval_every, eval_episodes=args.eval_episodes,
        patience=args.patience, patience_start=args.patience_start,
        checkpoint_path=ckpt, plot_path=plot, batch_size=args.batch_size,
    )
    print(f"\n[{args.mode}+demos] best {hist.get('best_episode', 0)} -> {ckpt}")