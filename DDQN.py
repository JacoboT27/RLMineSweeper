"""
DDQN agent for Minesweeper.

Contents (the "policy" script):
  QNetwork     -- fully-convolutional Q-net. Input (B, 11, H, W) -> output
                  (B, H*W): one Q-value per cell/action. No dense layer, so the
                  parameter count is independent of board size and weights
                  transfer across the curriculum (6x6 -> 16x16).
  ReplayBuffer -- uniform experience replay. Crucially stores the legal-action
                  mask at s' (next_mask), which the DDQN target needs.
  DDQNAgent    -- ties them together. Exposes the surface the training loop uses:
                  select_action(obs, mask, greedy), store(...), learn(), save/load.

Double-DQN target (decouples action *selection* from *evaluation* to curb the
overestimation bias of the plain-DQN max):
    a*  = argmax_{a' in legal(s')}  Q(s', a'; theta)          # online selects
    y   = r + gamma * (1 - done) * Q(s', a*; theta^-)         # target evaluates
    L   = SmoothL1( Q(s, a; theta),  y )
Only theta is optimised; theta^- is a periodically-synced frozen copy.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Fully-convolutional Q-network                                              #
# --------------------------------------------------------------------------- #
class QNetwork(nn.Module):
    """(B, in_channels, H, W) -> (B, H*W). Spatial dims preserved throughout
    (3x3 convs, padding 1, stride 1); a final 1x1 conv produces one value per
    cell. No flatten-to-dense, so params don't depend on H, W.

    Receptive field: each 3x3 layer adds 1 to the radius, so n_layers layers see
    a (2*n_layers + 1) window. Bump n_layers (or add dilation) for large boards.
    """

    def __init__(self, in_channels: int = 11, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        blocks: list[nn.Module] = []
        c = in_channels
        for _ in range(n_layers):
            blocks += [nn.Conv2d(c, hidden, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            c = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)   # per-cell Q, no activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)                 # (B, hidden, H, W)
        q = self.head(z)                 # (B, 1, H, W)
        return q.flatten(start_dim=1)    # (B, H*W)


class DuelingQNetwork(nn.Module):
    """Dueling head: shared conv body, then a per-cell advantage stream A(s,a)
    and a scalar state-value stream V(s), recombined as
        Q(s, a) = V(s) + A(s, a) - mean_a A(s, a).
    Factoring the big shared "how good is this position" term into V lets the
    advantage stream spend its capacity on *ranking* cells against each other --
    the midgame skill plain DQN tends to get wrong when many cells look similar.

    Stays size-agnostic: the value stream global-average-pools over H, W before
    its linear layers, so its parameters depend only on the channel count, not
    the board size -- the curriculum transfer property is preserved.

    The advantage conv is named `head` to match QNetwork, so a plain-QNetwork
    checkpoint loaded with strict=False warm-starts the body AND the advantage
    stream, leaving only V random. Since V is added equally to every action, the
    initial greedy argmax is identical to the source network's policy.
    """

    def __init__(self, in_channels: int = 11, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        blocks: list[nn.Module] = []
        c = in_channels
        for _ in range(n_layers):
            blocks += [nn.Conv2d(c, hidden, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            c = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)      # advantage A(s,a), per cell
        self.value = nn.Sequential(                          # scalar V(s) per board
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)                       # (B, hidden, H, W)
        adv = self.head(z).flatten(start_dim=1)               # (B, H*W)
        v = self.value(z.mean(dim=(2, 3)))                    # GAP -> (B, hidden) -> (B, 1)
        return v + adv - adv.mean(dim=1, keepdim=True)        # (B, H*W)


# --------------------------------------------------------------------------- #
#  Replay buffer                                                              #
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Preallocated ring buffer. One buffer is tied to a fixed board size; the
    curriculum uses a fresh buffer per stage (transitions have that stage's
    shape). Memory ~ capacity * 2 * 11 * H * W * 4 bytes -- keep capacity modest
    on small boards."""

    def __init__(self, capacity: int, obs_shape: tuple[int, int, int],
                 n_actions: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=bool)
        self.ptr = 0
        self.size = 0

    def push(self, obs, act, rew, next_obs, done, next_mask) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.act[i] = act
        self.rew[i] = rew
        self.done[i] = float(done)
        self.next_mask[i] = next_mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        d = self.device
        return (
            torch.as_tensor(self.obs[idx], device=d),        # (B,11,H,W) float32
            torch.as_tensor(self.act[idx], device=d),        # (B,) int64
            torch.as_tensor(self.rew[idx], device=d),        # (B,) float32
            torch.as_tensor(self.next_obs[idx], device=d),   # (B,11,H,W) float32
            torch.as_tensor(self.done[idx], device=d),       # (B,) float32
            torch.as_tensor(self.next_mask[idx], device=d),  # (B,H*W) bool
        )


# --------------------------------------------------------------------------- #
#  Prioritized replay (sum-tree)                                              #
# --------------------------------------------------------------------------- #
class SumTree:
    """Fixed-capacity binary tree where each leaf holds a priority and each
    internal node holds the sum of its children. total() is the root. get(s)
    walks down in O(log N) to the leaf whose cumulative priority range contains
    s -- so sampling proportional to priority is O(log N), not O(N)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        idx = tree_idx
        while idx != 0:                       # propagate the delta up to the root
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float):
        """Return (tree_idx, priority, data_idx) for cumulative value s."""
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):        # reached a leaf
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx, float(self.tree[idx]), idx - (self.capacity - 1)


class PrioritizedReplayBuffer:
    """Experience replay that samples transitions in proportion to |TD error|^alpha,
    so rare, high-error transitions (the mine hits and misvalued cells) get
    replayed far more often than a uniform buffer would allow. Importance-sampling
    weights (annealed via beta) correct the bias this introduces.

    Same data layout and push() signature as ReplayBuffer; sample() additionally
    returns the IS weights and the tree indices to update after the gradient step."""

    def __init__(self, capacity: int, obs_shape, n_actions: int,
                 device: torch.device, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.eps = eps
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=bool)
        self.tree = SumTree(capacity)
        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0               # new transitions get max priority

    def push(self, obs, act, rew, next_obs, done, next_mask) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.act[i] = act
        self.rew[i] = rew
        self.done[i] = float(done)
        self.next_mask[i] = next_mask
        self.tree.update(i + (self.capacity - 1), self.max_priority ** self.alpha)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        total = self.tree.total()
        segment = total / batch_size
        tree_idxs = np.empty(batch_size, dtype=np.int64)
        data_idxs = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            s = max(np.random.uniform(segment * i, segment * (i + 1)), 1e-8)
            t_idx, p, d_idx = self.tree.get(s)
            tree_idxs[i] = t_idx
            data_idxs[i] = min(max(d_idx, 0), self.size - 1)   # guard the s==0 edge
            priorities[i] = p
        probs = np.maximum(priorities, 1e-12) / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()                               # normalize to (0, 1]
        d, di = self.device, data_idxs
        return (
            torch.as_tensor(self.obs[di], device=d),
            torch.as_tensor(self.act[di], device=d),
            torch.as_tensor(self.rew[di], device=d),
            torch.as_tensor(self.next_obs[di], device=d),
            torch.as_tensor(self.done[di], device=d),
            torch.as_tensor(self.next_mask[di], device=d),
            torch.as_tensor(weights, dtype=torch.float32, device=d),
            tree_idxs,
        )

    def update_priorities(self, tree_idxs, td_errors) -> None:
        for idx, td in zip(tree_idxs, td_errors):
            p = abs(float(td)) + self.eps
            self.max_priority = max(self.max_priority, p)
            self.tree.update(int(idx), p ** self.alpha)


# --------------------------------------------------------------------------- #
#  DDQN agent                                                                 #
# --------------------------------------------------------------------------- #
class DDQNAgent:
    def __init__(self, board_hw: tuple[int, int], in_channels: int = 11,
                 hidden: int = 64, n_layers: int = 4, dueling: bool = False,
                 gamma: float = 0.99,
                 lr: float = 1e-3, batch_size: int = 64,
                 buffer_capacity: int = 100_000, eps_start: float = 1.0,
                 eps_end: float = 0.05, eps_decay_steps: int = 50_000,
                 target_sync: int = 1000, learning_starts: int = 1000,
                 grad_clip: float = 10.0, prioritized: bool = False,
                 per_alpha: float = 0.6, per_beta_start: float = 0.4,
                 per_beta_steps: int = 100_000, per_eps: float = 1e-6,
                 device: str | None = None):
        H, W = board_hw
        self.H, self.W = H, W
        self.n_actions = H * W
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.target_sync = target_sync
        self.learning_starts = learning_starts
        self.grad_clip = grad_clip
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        Net = DuelingQNetwork if dueling else QNetwork
        self.online = Net(in_channels, hidden, n_layers).to(self.device)
        self.target = Net(in_channels, hidden, n_layers).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()                       # target is never trained directly
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.prioritized = prioritized
        self.per_beta_start = per_beta_start
        self.per_beta_steps = per_beta_steps
        if prioritized:
            self.buffer = PrioritizedReplayBuffer(
                buffer_capacity, (in_channels, H, W), self.n_actions, self.device,
                alpha=per_alpha, eps=per_eps)
        else:
            self.buffer = ReplayBuffer(buffer_capacity, (in_channels, H, W),
                                       self.n_actions, self.device)

        self.act_step = 0     # counts training action selections (drives epsilon)
        self.learn_step = 0   # counts gradient steps (drives target sync)

    # ---- exploration schedule -------------------------------------------- #
    def epsilon(self) -> float:
        frac = min(1.0, self.act_step / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ---- action selection (masked) --------------------------------------- #
    def select_action(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        """obs (11,H,W) float32, mask (H*W,) bool (True = legal). Returns a
        legal action index. Illegal actions are impossible to select."""
        if not greedy:
            self.act_step += 1
            if np.random.random() < self.epsilon():
                legal = np.flatnonzero(mask)
                return int(np.random.choice(legal))
        with torch.no_grad():
            x = torch.as_tensor(obs, device=self.device).unsqueeze(0)   # (1,11,H,W)
            q = self.online(x).squeeze(0)                               # (H*W,)
            illegal = ~torch.as_tensor(mask, device=self.device)
            q = q.masked_fill(illegal, float("-inf"))
            return int(torch.argmax(q).item())

    # ---- storage --------------------------------------------------------- #
    def store(self, obs, act, rew, next_obs, done, next_mask) -> None:
        self.buffer.push(obs, act, rew, next_obs, done, next_mask)

    # ---- learning (one DDQN gradient step) ------------------------------- #
    def learn(self) -> float | None:
        if self.buffer.size < max(self.batch_size, self.learning_starts):
            return None
        if self.prioritized:
            beta = min(1.0, self.per_beta_start + (1.0 - self.per_beta_start)
                       * self.learn_step / max(1, self.per_beta_steps))
            obs, act, rew, next_obs, done, next_mask, weights, tree_idxs = \
                self.buffer.sample(self.batch_size, beta)
        else:
            obs, act, rew, next_obs, done, next_mask = self.buffer.sample(self.batch_size)
            weights, tree_idxs = None, None

        q_sa = self.online(obs).gather(1, act.view(-1, 1)).squeeze(1)   # Q(s,a;theta)

        with torch.no_grad():
            q_next_online = self.online(next_obs)                       # select with theta
            q_next_online = q_next_online.masked_fill(~next_mask, float("-inf"))
            a_star = q_next_online.argmax(dim=1, keepdim=True)          # (B,1)
            q_next_target = self.target(next_obs).gather(1, a_star).squeeze(1)  # eval with theta^-
            target = rew + self.gamma * (1.0 - done) * q_next_target    # no bootstrap if done

        td_error = q_sa - target                                        # (B,) per-sample
        huber = F.smooth_l1_loss(q_sa, target, reduction="none")        # (B,)
        loss = (weights * huber).mean() if weights is not None else huber.mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.prioritized:                                            # feed |TD| back as priority
            self.buffer.update_priorities(tree_idxs, td_error.detach().abs().cpu().numpy())

        self.learn_step += 1
        if self.learn_step % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    # ---- checkpointing --------------------------------------------------- #
    def save(self, path: str) -> None:
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "act_step": self.act_step,
            "learn_step": self.learn_step,
        }, path)

    def load(self, path: str, weights_only: bool = False, strict: bool = True) -> None:
        """Load a checkpoint. weights_only=True loads just the network weights
        (skips optimizer/step counters) -- use it for a clean fine-tune or when
        the optimizer shape differs. strict=False allows partial loads, e.g.
        warm-starting a DuelingQNetwork from a plain QNetwork checkpoint: the
        body and advantage `head` match and load; the value stream stays fresh."""
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"], strict=strict)
        self.target.load_state_dict(ckpt["target"], strict=strict)
        if not weights_only:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.act_step = ckpt.get("act_step", 0)
            self.learn_step = ckpt.get("learn_step", 0)