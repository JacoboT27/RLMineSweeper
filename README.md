# Reinforcement Learning Agent to Solve Minesweeper Boards with Transfer Learning

This repository trains agents to play Minesweeper and visualizes their gameplay.
It compares a value-based approach (**Double DQN**, in three variants) against a
policy-gradient approach (**PPO**), all on a 9×9 board with 10 mines.

**Headline result:** the DQN variants plateau around a **0.32–0.42** greedy win
rate; PPO reaches **~0.87**, close to the practical ceiling for pure RL on this
board (~0.90).

---

## Contents

- [Game Environment](#game-environment)
- [MDP Formulation](#mdp-formulation)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Reward Function](#reward-function)
- [DDQN](#ddqn)
  - [The three modes](#the-three-modes)
- [PPO](#ppo)
- [Theory](#theory)
- [Usage](#usage)
- [Results](#results)
  - [Metrics](#metrics)
  - [Training results](#training-results)
  - [Evaluation results](#evaluation-results)
  - [Gameplay](#gameplay)
  - [Discussion](#discussion)
- [Transfer Learning](#transfer-learning)

---

## Game Environment

- **`minesweeper_env.py`** defines the game logic — board generation, mine
  placement, reveal + flood-fill cascade, and win/loss detection — wrapped in a
  Gymnasium `Env` that returns the observation, reward, termination flags, and a
  legal-action mask at each step. Mines are placed lazily on the first click,
  excluding the clicked cell *and its 8 neighbours*, so the first reveal always
  opens a region (standard modern-Minesweeper behaviour). Flagging is not
  modelled: flags never change the outcome, so the agent only ever reveals.

- **`test_minesweeper.py`** runs correctness tests over the environment
  (first-click-safe, flood fill, adjacency counts, win/loss, Gymnasium API
  compliance). A debug tool, not part of training.

- **`play.py`** lets a human play Minesweeper in the terminal, against the same
  environment the agents train on.

---

## MDP Formulation

### Action Space

`Discrete(H·W)` — one action per cell (81 actions on a 9×9 board). Action `a`
reveals the cell at `row = a // W`, `col = a % W`. Only unrevealed cells are
legal; the environment supplies a boolean legal-action mask in
`info["action_mask"]`, and every agent masks illegal cells before selecting an
action (set to `-∞` before an argmax or a softmax).

### Observation Space

`Box(0, 1, shape=(11, H, W), float32)` — an 11-channel spatial tensor:

| Channel | Meaning |
|---|---|
| 0 | **Hidden mask** — 1 where a cell is unrevealed |
| 1–9 | **Count one-hot** — channel `1+k` is 1 where a revealed cell shows adjacent-mine count `k` (0–8) |
| 10 | **Mine density** — a broadcast scalar `mines_remaining / hidden_cells`, the prior probability that an unknown cell is a mine |

Channels 0–9 form a clean one-hot per cell (exactly one is 1). The legal-action
mask equals channel 0 (a cell is clickable iff it is hidden), so it is *not*
duplicated as a channel — it is passed in `info` for the agent to apply. The
fully-convolutional networks make this encoding board-size-agnostic.

### Reward Function

Dense, per click:

| Event | Reward |
|---|---|
| Safe reveal (non-terminal) | **+0.1** |
| Winning click (all safe cells revealed) | **+1.0** |
| Hitting a mine (terminal) | **−1.0** |
| Illegal click (masked out in practice) | **−1.0** |

"Per click" means a click is rewarded the same whether it cascades 1 cell or 40
— the agent is rewarded for *surviving*, not for cascade luck. An episode ends on
a mine (loss) or when every non-mine cell is revealed (win).

---

## DDQN

Double Deep Q-Network is an **off-policy, value-based** algorithm. It learns a
Q-function `Q(s, a)` — the expected return of clicking cell `a` in state `s` —
and acts greedily by taking the highest-Q legal cell. Transitions are stored in a
replay buffer and reused; a slowly-updated **target network** stabilizes the
bootstrap; and the **double** trick decouples action *selection* from *evaluation*
to curb the overestimation bias of the plain-DQN max (see [Theory](#theory)).

### The three modes

Selected with `--mode`. The modes are **cumulative**, matching the project's
development order:

- **`base`** — standard Double DQN with a fully-convolutional Q-network
  (`(B, 11, H, W) → (B, H·W)`, one Q-value per cell, no dense layers).
- **`dueling`** — adds a **dueling head** that factors
  `Q(s, a) = V(s) + A(s, a) − meanₐ A(s, a)`, separating *how good the position
  is* (`V`) from *how much better each cell is than average* (`A`). This is meant
  to sharpen the *relative ranking* of similar-valued cells in the midgame.
- **`per`** — adds **Prioritized Experience Replay** on top of dueling. Instead
  of sampling the buffer uniformly, it samples transitions with probability
  `∝ |TD-error|^α`, so rare, high-error transitions (mine hits, misvalued cells)
  are replayed far more often; importance-sampling weights correct the bias this
  introduces.

### Files

- **`DDQN.py`** defines the Q-network (fully-convolutional: input
  `(B, 11, H, W)` → output `(B, H·W)`, no dense layer, so parameter count is
  independent of board size and weights transfer across a curriculum, 6×6 → 9×9),
  the dueling head, the uniform and prioritized replay buffers, and the agent
  (masked ε-greedy action selection, the Double-DQN update, target syncing, and
  checkpoint save/load).
- **`DDQN_train.py`** holds the training loop: it runs episodes, stores
  transitions, learns each step (ε decays as the agent acts), evaluates greedily
  on a fixed set of seeds, early-stops on a patience criterion, and writes
  training plots plus `best` and `last` checkpoints.

### Usage

```bash
python DDQN_train.py --mode base
python DDQN_train.py --mode dueling
python DDQN_train.py --mode per
```

Each writes `policies/best_<mode>.pt` (+ `_last.pt`) and
`training_plots/training_<mode>.png`. 

---

## PPO

Proximal Policy Optimization is an **on-policy, actor-critic** algorithm. A shared
convolutional body feeds two heads: an **actor** (per-cell logits → a probability
distribution over which cell to click) and a **critic** (a scalar value `V(s)`).
It collects fresh experience from **many parallel environments**, computes
advantages with **GAE**, and updates the policy with a **clipped surrogate
objective** that limits how far the policy can move per update (see
[Theory](#theory)). Because it optimizes the policy directly rather than taking an
argmax over value estimates, it avoids the failure mode where a value network
misranks a risky cell above an obviously safe one.

### Files

- **`PPO_agent.py`** defines the masked categorical policy, the actor-critic
  network, the vectorized environment, rollout collection, GAE, the clipped PPO
  update, and the training loop. Saves the best checkpoint and training plots.

### Usage

```bash
python PPO_agent.py
```

The training loop runs for a fixed iteration budget (no early stopping) and keeps
the best-return checkpoint; read the training plot to judge convergence.

---

## Theory

**Double DQN target.** The online network `θ` *selects* the next action; the
frozen target network `θ⁻` *evaluates* it, which removes the max-operator's
upward bias:

```
a* = argmax_{a' ∈ legal(s')} Q(s', a'; θ)
y  = r + γ (1 − done) · Q(s', a*; θ⁻)
L  = Huber( Q(s, a; θ),  y )
```

The `(1 − done)` term zeroes the bootstrap at terminal steps — important here,
where every mine hit and every win ends an episode.

**Dueling head.** `Q(s, a) = V(s) + A(s, a) − meanₐ A(s, a)`. Factoring the large
shared "position value" into `V` lets the advantage stream spend capacity on
ranking cells against each other. The mean-subtraction keeps the decomposition
identifiable.

**Prioritized replay.** Sample transition `i` with probability
`P(i) ∝ pᵢ^α`, where `pᵢ = |TD-errorᵢ| + ε`. Correct the resulting bias with
importance-sampling weights `wᵢ = (N · P(i))^{−β}`, with `β` annealed toward 1. A
sum-tree gives O(log N) sampling and priority updates.

**PPO — GAE and the clipped objective.** After a rollout, advantages are an
exponentially-weighted sum of TD residuals `δ_t = r_t + γ(1−d_t)V(s_{t+1}) − V(s_t)`:

```
A_t = δ_t + γλ(1 − d_t) A_{t+1}      (computed backward; (1−d) cuts game seams)
```

The policy is updated for a few epochs over the fresh batch with the clipped
surrogate, where `r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)`:

```
L_clip = −E[ min( r_t · A_t,  clip(r_t, 1−ε, 1+ε) · A_t ) ]
L      = L_clip + c₁ · (V − Rₜ)²  −  c₂ · entropy
```

The clip is what lets PPO safely take several gradient steps on one on-policy
batch: it caps how far the policy may drift from the one that generated the data.
The entropy bonus keeps exploration alive; annealing its coefficient toward 0 late
in training lets the policy fully commit to safe cells.

---

## Results

All training and evaluation was done on a **9×9 grid with 10 mines**. Evaluation
uses a **fixed set of held-out seeds** (`1_000_000 + i`), identical for the DQN and
PPO agents, so all win rates below are directly comparable.

### Metrics

- **Win rate** — fraction of games won (all non-mine cells revealed). The primary
  metric.
- **Return** — mean total reward per game. Since safe clicks pay +0.1, wins +1,
  and mines −1, higher return means more safe cells revealed and/or more wins.
  Smoother and less noisy than win rate.
- **Len** — mean number of clicks per game. A click can cascade many cells, so
  longer generally means the agent survived (revealed) longer.
- **Entropy** (PPO only) — mean entropy of the policy's action distribution; high
  early (exploring), decaying as the policy commits.
- **Ploss** (PPO only) — the clipped policy loss. It hovers near zero *by design*
  (clipped objective) and is not a progress signal — read win rate/return instead.

### Training results

| Model | Checkpoint | Training | Win rate | Return | Len |
|---|---|---|---|---|---|
| base | `best_base.pt` | 16,000 ep | 0.350 | +0.92 | 13.2 |
| dueling | `best_dueling.pt` | 22,000 ep | 0.335 | +0.82 | 12.5 |
| per | `best_per.pt` | 16,500 ep | 0.375 | +0.82 | 11.7 |
| ppo | `best_ppo.pt` | 3,250 iters | 0.850 | +2.73 | — |

*(PPO also reported entropy 0.48 and policy loss +0.038 at its best checkpoint.
Note DQN "training" is counted in episodes and PPO in iterations — they are not
directly comparable units.)*

### Evaluation results

Greedy play (argmax) over 100 held-out games:

| Model | Greedy win rate | Avg clicks |
|---|---|---|
| base | 0.42 | 12.7 |
| dueling | 0.32 | 13.6 |
| per | 0.39 | 10.9 |
| **ppo** | **0.87** | 20.6 |

### Gameplay

Generated with `visualize.py` (board on the left, the agent's decision heatmap on
the right — Q-values for DQN, policy probabilities for PPO):

| base | dueling |
|---|---|
| ![base](visualization/game_base.gif) | ![dueling](visualization/game_dueling.gif) |

| per | ppo |
|---|---|
| ![per](visualization/game_per.gif) | ![ppo](visualization/game_ppo.gif) |

**`visualize.py`** runs a policy over 100 games to measure win rate and renders a
full game as a GIF + filmstrip:

```bash
python visualize.py --mode base
python visualize.py --mode dueling
python visualize.py --mode per
python visualize.py --mode ppo
```

Each loads `policies/best_<mode>.pt` and writes
`visualization/game_<mode>.gif` and `..._filmstrip.png`.

### Discussion

- **PPO roughly doubles the value-based agents** (0.87 vs 0.32–0.42 greedy). The
  DQN variants tend to misrank cells — assigning high value to an uncertain cell
  over an obviously safe one — a known weakness of taking an argmax over a
  bootstrapped value surface. PPO optimizes the action *probabilities* directly
  and sidesteps this.
- **Dueling and PER did not clearly beat base** on this board — `base` even edges
  `dueling` at evaluation. That is a legitimate finding: their benefit is
  problem-dependent, and 9×9/10 apparently isn't complex enough (or the runs not
  long enough) for the value-ranking machinery to pay off. It is a useful reminder
  that a fancier component is not monotonically better.
- **Caveat: single seed.** These are single-run numbers, and RL is high-variance
  run-to-run. Small gaps (e.g. base 0.42 vs per 0.39) should not be over-read;
  multi-seed runs with reported spread would be needed to make firm ordering
  claims among the DQN variants. The PPO vs DQN gap is large enough to be robust.
- **Ceiling.** ~0.90 is roughly the practical maximum for pure RL on beginner
  boards, since some positions force a genuine guess no agent can win. PPO at 0.87
  is near that ceiling; closing the last gap would require a hybrid (a constraint
  solver for deducible cells + RL for the true guesses).
 
---
## Transfer Learning

```bash
python transfer_learning.py --mode base --init-ckpt policies/best_base.pt --margin 0.1 --margin-lambda 0.3
```

