# Reinforcement Learning Agent to Solve Minesweeper Boards

This repository trains a policy for playing minesweeper, and visualizes the gameplay. A DQN model is compared with a PPO policy.

#TODO Add Index for all subsection
#TODO Add Gifs of results
#TODO Write report of theory and include it in the repo

---

## Game Environment
- `minesweeper_env.py` defines the game logic, creates the board, handles the inputs, and the reset function. It also contains a gymnasium API wrapper that return the action, observation, and reward at each step.

- `test_minesweeper.py` runs some tests over the environment to ensure it is working as intended. It is a debug tool, not relevant for training of the policies

- `play.py` allows the user to play minesweeper on the terminal.

---
## MDP
### Action Space
#TODO Write action space

### Decision Space
#TODO Describe observation space

### Reward Function
#TODO Describe reward function

---
## DDQN

Double Deep Q-Network is an off-policy algorithm that splits action selection and action evaluation into two separate networks.

#TODO explain the differce of all 3 modes

### Files
 - `DDQN.py` defines the Double Deep Q-Network structure. It contains a fully-convolutional Q-net with Input (B, 11, H, W) -> output (B, H*W): one Q-value per cell/action. No dense layer, so the parameter count is independent of board size and weights transfer across the curriculum (6x6 -> 16x16). It also defines the Replay Buffer. 
 - `DDQN_train.pt` holds the Training loop for the DDQN Minesweeper agent. Produces training Plots and best and last checkpoints. 

 ### Usage

```bash
python DDQN_train.py --mode base
python DDQN_train.py --mode dueling
python DDQN_train.py --mode per
```
---
## PPO

Proximal Policy Optimization is an On-Policy Actor-Critic RL algorithm that uses multiple parallel environments to gather data. 

### Files

- `PPO_agent.py` defines the policy and the training loop. Saves the best checkpoint and training plots

### Usage

```bash
python PPO_agent.py
```

---
## Results

All training and evaluations were done on a 9x9 grid with 10 mines. 

### Visualization

- `visualize.py` runs the policy over 100 games to measure win rate, and produces a visualization of a full gameplay

```bash
python visualize.py --mode base
python visualize.py --mode dueling
python visualize.py --mode per
python visualize.py --mode ppo

```

### Train Results

#TODO explain what is return, win rate, len, etc. maybe change layout to a table

- base: `best_base.pt` trained for 16000 episodes, obtained a 0.35 win rate, +0.92 return, and 13.2 len
- dueling: `best_dueling.pt` trained for 22000  episodes, obtained a 0.335 win rate, +0.82 return and 12.5 len 
- per: `best_per.pt` trained for 16500 episodes, obtained a 0.375 win rate, +0.82 return and 11.7 len
- ppo: `best_ppo.pt` trained for 3250 iterations, obtained a 0.85 win rate, +2.73 return, 0.48 entropy and +0.038 ploss

### Eval Results
- base: greedy win rate of 0.42 with avg clicks of 12.7
- dueling: greedy win rate of 0.32 with avg clicks of 13.6
- per: greedy win rate of 0.39 with avg clicks of 10.9
- ppo: greedy win rate of 0.87 with avg clicks of 20.6
---