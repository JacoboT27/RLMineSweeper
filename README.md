# Reinforcement Learning Agent to Solve Minesweeper Boards
---
This repository trains a policy for playing minesweeper, and visualizes the gameplay. A DQN model is compared with a PPO policy.

---

## Game Environment
- `minesweeper_env.py` defines the game logic, creates the board, handles the inputs, and the reset function. It also contains a gymnasium API wrapper that return the action, observation, and reward at each step.

- `test_minesweeper.py` runs some tests over the environment to ensure it is working as intended. It is a debug tool, not relevant for training of the policies

- `play.py` allows the user to play minesweeper on the terminal.

---