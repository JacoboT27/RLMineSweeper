import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from minesweeper_env import MinesweeperEnv
from DDQN_train import evaluate, plot_history, set_seed   # reuse your DQN harness

class MaskedCategorical:
    """A categorical distribution over cells, with illegal cells masked out.
    logits: (B, A) raw scores from the policy head, A = H*W
    mask:   (B, A) bool, True = legal (unrevealed) cell
    """
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        # push illegal logits to -inf so their probability is exactly 0
        masked = logits.masked_fill(~mask, float("-inf"))
        self.dist = Categorical(logits=masked)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()              # (B,) chosen cell index

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(actions)     # (B,) log pi(a|s)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()             # (B,) exploration signal


class ActorCritic(nn.Module):
    """Shared conv body -> (policy logits per cell, scalar state value).
    Input  x: (B, 11, H, W)
    Returns   logits (B, H*W)  and  value (B,)
    """
    def __init__(self, in_channels: int = 11, hidden: int = 64, n_layers: int = 4,
                 dilations=None):
        super().__init__()
        # (3) dilations let a few layers "see" farther WITHOUT adding depth: a 3x3
        # conv with dilation d and padding=d keeps the spatial size but widens the
        # receptive field. e.g. dilations=[1,2,4,8] reaches radius 15 in 4 layers,
        # covering a 9x9 board corner-to-corner. Weight shapes are unchanged, so a
        # non-dilated checkpoint still loads into it (shape-compatible warm-start).
        if dilations is None:
            dilations = [1] * n_layers
        blocks, c = [], in_channels
        for d in dilations:
            blocks += [nn.Conv2d(c, hidden, kernel_size=3, padding=d, dilation=d),
                       nn.ReLU(inplace=True)]
            c = hidden
        self.body = nn.Sequential(*blocks)

        self.actor = nn.Conv2d(hidden, 1, kernel_size=1)      # per-cell logit (like the DQN head)
        self.critic = nn.Sequential(                          # scalar V(s) (like the dueling value stream)
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        z = self.body(x)                          # (B, hidden, H, W)
        logits = self.actor(z).flatten(1)         # (B, H*W)
        value = self.critic(z.mean(dim=(2, 3))).squeeze(-1)   # GAP -> (B,)
        return logits, value

class VecEnv:
    """N independent Minesweeper envs stepped together. Auto-resets any env that
    finishes, so a rollout is a continuous stream of steps across many games."""
    def __init__(self, n_envs, height=9, width=9, mines=10, seed=0):
        self.envs = [MinesweeperEnv(height, width, mines) for _ in range(n_envs)]
        self.n = n_envs
        self.obs = np.zeros((n_envs, 11, height, width), np.float32)
        self.masks = np.zeros((n_envs, height * width), bool)
        for i, e in enumerate(self.envs):
            o, info = e.reset(seed=seed + i)
            self.obs[i], self.masks[i] = o, info["action_mask"]

    def step(self, actions):
        rewards = np.zeros(self.n, np.float32)
        dones = np.zeros(self.n, np.float32)
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, info = e.step(int(a))
            rewards[i], dones[i] = r, float(term or trunc)
            if term or trunc:                       # game over -> start a fresh one
                o, info = e.reset()
            self.obs[i], self.masks[i] = o, info["action_mask"]
        return self.obs, self.masks, rewards, dones

@torch.no_grad()
def collect_rollout(vec, net, T, device):
    """Run the policy for T steps across all envs. Returns a dict of rollout
    tensors shaped (T, N, ...), plus the bootstrap value V(s_T) of shape (N,)."""
    N = vec.n
    obs_buf   = torch.zeros(T, N, 11, 9, 9, device=device)
    mask_buf  = torch.zeros(T, N, 81, dtype=torch.bool, device=device)
    act_buf   = torch.zeros(T, N, dtype=torch.long, device=device)
    logp_buf  = torch.zeros(T, N, device=device)
    val_buf   = torch.zeros(T, N, device=device)
    rew_buf   = torch.zeros(T, N, device=device)
    done_buf  = torch.zeros(T, N, device=device)

    obs  = torch.as_tensor(vec.obs, device=device)          # (N, 11, 9, 9)
    mask = torch.as_tensor(vec.masks, device=device)        # (N, 81)

    for t in range(T):
        logits, value = net(obs)                            # (N, 81), (N,)
        dist = MaskedCategorical(logits, mask)
        action = dist.sample()                              # (N,)

        obs_buf[t], mask_buf[t] = obs, mask                 # state we acted FROM
        act_buf[t]  = action
        logp_buf[t] = dist.log_prob(action)                 # pi_old(a|s), snapshot NOW
        val_buf[t]  = value                                 # V(s_t) baseline

        next_obs, next_mask, reward, done = vec.step(action.cpu().numpy())
        rew_buf[t]  = torch.as_tensor(reward, device=device)
        done_buf[t] = torch.as_tensor(done, device=device)

        obs  = torch.as_tensor(next_obs, device=device)
        mask = torch.as_tensor(next_mask, device=device)

    # bootstrap: value of the state AFTER the last collected step
    _, last_value = net(obs)                                # (N,)

    return {
        "obs": obs_buf, "mask": mask_buf, "act": act_buf,
        "logp": logp_buf, "val": val_buf, "rew": rew_buf, "done": done_buf,
    }, last_value

def compute_gae(roll, last_value, gamma=0.99, lam=0.95):
    """Backward GAE over a (T, N) rollout. Returns advantages and returns, both (T, N)."""
    rew, val, done = roll["rew"], roll["val"], roll["done"]      # each (T, N)
    T, N = rew.shape
    adv = torch.zeros_like(rew)
    gae = torch.zeros(N, device=rew.device)                     # running A_{t+1}, starts at 0
    next_value = last_value                                     # V(s_T), the bootstrap

    for t in reversed(range(T)):                               # walk backward through time
        nonterminal = 1.0 - done[t]                            # 0 at a game seam
        delta = rew[t] + gamma * next_value * nonterminal - val[t]
        gae = delta + gamma * lam * nonterminal * gae          # (1-d) cuts the carry
        adv[t] = gae
        next_value = val[t]                                    # shift the "future value" back one step

    returns = adv + val                                        # R_t = A_t + V(s_t)
    return adv, returns

def ppo_update(net, optimizer, roll, adv, returns, *,
               clip=0.2, epochs=4, minibatch=256, vf_coef=0.5, ent_coef=0.01,
               max_grad_norm=0.5):
    # flatten (T, N, ...) -> (B, ...)
    B = adv.numel()
    obs   = roll["obs"].reshape(B, *roll["obs"].shape[2:])
    mask  = roll["mask"].reshape(B, -1)
    act   = roll["act"].reshape(B)
    logp_old = roll["logp"].reshape(B)
    adv   = adv.reshape(B)
    returns = returns.reshape(B)

    # normalize advantages across the batch (variance control -- big stabilizer)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    idx = np.arange(B)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, B, minibatch):
            mb = idx[start:start + minibatch]

            logits, value = net(obs[mb])                 # FRESH forward, grads on
            dist = MaskedCategorical(logits, mask[mb])
            logp = dist.log_prob(act[mb])                # log pi_new(a|s)

            ratio = torch.exp(logp - logp_old[mb])       # r_t = pi_new / pi_old
            a = adv[mb]
            unclipped = ratio * a
            clipped   = torch.clamp(ratio, 1 - clip, 1 + clip) * a
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = (value - returns[mb]).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)   # grad clip != ratio clip
            optimizer.step()

    return {"policy_loss": policy_loss.detach().item(),
            "value_loss":  value_loss.detach().item(),
            "entropy":     entropy.detach().item()}

def train_ppo(n_envs=32, rollout=128, total_iterations=4000,
              lr=2.5e-4, eval_every=50, eval_episodes=200,
              n_layers=4, dilations=None, minibatch=256,
              ent_coef_start=0.01, ent_coef_end=0.0,
              init_ckpt=None,
              checkpoint_path="policies/best_ppo_v2.pt", plot_path="training_plots/training_ppo_v2.png"):
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec = VecEnv(n_envs)
    net = ActorCritic(hidden=64, n_layers=n_layers, dilations=dilations).to(device)

    # (1) warm-start: continue from an existing checkpoint. strict=False so this also
    #     works when the new net is DEEPER (extra layers stay fresh) or DILATED (same
    #     shapes). Same architecture -> full transfer, so eval resumes at its win rate.
    if init_ckpt is not None:
        ckpt = torch.load(init_ckpt, map_location=device)
        missing, unexpected = net.load_state_dict(ckpt["net"], strict=False)
        fresh = [k for k in missing if k.endswith(".weight")]
        print(f"warm-started from {init_ckpt} | freshly-initialized layers: {fresh or 'none (full transfer)'}")

    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    eval_env = MinesweeperEnv(9, 9, 10)
    history = {"iter": [], "eval_win_rate": [], "eval_return": [], "entropy": [], "ent_coef": []}
    best = -float("inf")

    for it in range(1, total_iterations + 1):
        roll, last_v = collect_rollout(vec, net, rollout, device)
        adv, returns = compute_gae(roll, last_v)

        # (2) entropy-coefficient decay: strong exploration pressure early, ~0 late
        #     so the policy can fully commit to safe cells once it knows the game.
        frac = it / total_iterations
        ent_coef = ent_coef_start + (ent_coef_end - ent_coef_start) * frac

        stats = ppo_update(net, opt, roll, adv, returns,
                           ent_coef=ent_coef, minibatch=minibatch)

        if it % eval_every == 0:
            wr, ret = evaluate_ppo(eval_env, net, eval_episodes, device)
            history["iter"].append(it)
            history["eval_win_rate"].append(wr)
            history["eval_return"].append(ret)
            history["entropy"].append(stats["entropy"])
            history["ent_coef"].append(ent_coef)
            improved = ret > best
            if improved:
                best = ret
                torch.save({"net": net.state_dict(),
                            "config": {"hidden": 64, "n_layers": n_layers,
                                       "dilations": dilations}}, checkpoint_path)
            print(f"iter {it:4d} | win {wr:.3f} | ret {ret:+.2f} | "
                  f"entropy {stats['entropy']:.2f} | ent_coef {ent_coef:.4f} | "
                  f"ploss {stats['policy_loss']:+.3f}{'  <-- best' if improved else ''}")

    return history

@torch.no_grad()
def evaluate_ppo(env, net, n_games, device):
    wins, returns = 0, []
    for s in range(n_games):
        obs, info = env.reset(seed=1_000_000 + s)          # same fixed eval seeds as DQN
        done, ret = False, 0.0
        while not done:
            logits, _ = net(torch.as_tensor(obs, device=device).unsqueeze(0))
            logits = logits.masked_fill(
                ~torch.as_tensor(info["action_mask"], device=device).unsqueeze(0), float("-inf"))
            a = int(logits.argmax(1))                       # greedy = most likely cell
            obs, r, term, trunc, info = env.step(a)
            ret += r; done = term or trunc
        wins += int(info["is_success"]); returns.append(ret)
    return wins / n_games, float(np.mean(returns))



# --------------------------------------------------------------------------- #
#  Plotting (PPO-specific: the DQN plot_history expects different keys)        #
# --------------------------------------------------------------------------- #
def plot_ppo(history, out_path="training_ppo.png", dqn_baseline=0.38):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = history["iter"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(it, history["eval_win_rate"], marker="o")
    axes[0].axhline(dqn_baseline, ls="--", color="grey", alpha=0.7, label=f"DQN ~{dqn_baseline}")
    axes[0].set_title("eval win rate"); axes[0].set_ylim(0, 1); axes[0].legend()

    axes[1].plot(it, history["eval_return"], marker="o", color="tab:green")
    axes[1].set_title("eval mean return")

    axes[2].plot(it, history["entropy"], marker="o", color="tab:orange")
    axes[2].set_title("policy entropy (should decay)")

    for a in axes:
        a.set_xlabel("iteration"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


if __name__ == "__main__":
    hist = train_ppo(
        n_envs=32, rollout=128,            # (1) wider batch: 32*128 = 4096 per iteration
        total_iterations=4000,             # (1) train longer
        lr=2.5e-4, minibatch=256,
        n_layers=4, dilations=None,        
        ent_coef_start=0.01, ent_coef_end=0.0,   # (2) anneal exploration pressure to ~0
        init_ckpt=None,          
        checkpoint_path="policies/best_ppo.pt", plot_path="training_plots/training_ppo.png",
    )

    # (3) RECEPTIVE-FIELD variant -- run this INSTEAD (uncomment) if the visualizer
    # shows LONG-RANGE deaths (constraints spanning more than a 9-cell window).
    # Dilations widen the field to cover the whole board in 4 layers. Weights are
    # shape-compatible so it warm-starts, but each conv's meaning shifts, so expect a
    # dip before it climbs. NOTE: to visualize a dilated checkpoint, update the
    # visualizer's PPOPlayer to read cfg["dilations"] and pass it to ActorCritic.
    # hist = train_ppo(
    #     n_envs=32, rollout=128, total_iterations=4000, lr=2.5e-4, minibatch=256,
    #     n_layers=4, dilations=[1, 2, 4, 8],
    #     ent_coef_start=0.01, ent_coef_end=0.0, init_ckpt="best_ppo.pt",
    #     checkpoint_path="best_ppo_dilated.pt", plot_path="training_ppo_dilated.png",
    # )

    if hist["iter"]:
        plot_ppo(hist, "training_plots/training_ppo.png")
        print("\nbest eval return:", max(hist["eval_return"]),
              "| best eval win rate:", max(hist["eval_win_rate"]))