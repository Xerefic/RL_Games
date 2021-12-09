# Shaastra 2022
# RL Games: Vasuki

## Description: 

Are you fed up with friends snakes betraying you? This is your chance to redeem yourself and rise above the rest to crown yourself as the King of the Snakes Vasuki!

We present to you RL Games, an arena where you will find only blood and venom. Create your bot using Reinforcement Learning techniques to fight against the best of the best bots. With a cash prize pool of INR 25K, what more do you need?

As a part of Shaastra 2022, RL games is a team competition where participants compete to find the best policy to a given environment using reinforcement learning methods. Do check out our [website](www.shaastra.org) for other exciting opportunities!

## Scripts

- Link to notebook: [![Open All Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B-Ohlwsehrg_eJwAeVQZM8ZmzO5GBrNn?usp=sharing)
- [Environment](environment.py)

## Game Description

The environment consists of two snakes (agents) and `4` food locations at any instant. The snakes (agents) can move in three directions; namely **left**, **right** or **straight** ahead. The objective of the game is to possess a greater score than the opponent either by consuming the food or by colliding with the opponent.

A breief description on the environment is given below:

### State Space:
- The state space is characterised by a `8 x 8` Grid (Continious Space).
- At any instant of time, `4` random coordinates out of `8` fixed coordinates possess food.

### Action Space:
- The agent may choose one of the three possible moves; left, right, forward at any instant.
- Depending on the position of the agent, the move may or may not be executed.
	- For instance, if the agent lies on the first row and is facing North, and decides to move left, the move will be determined illegal and the agent will not be displaced. Although the move does not take place, the agent will be turned to face West.
	- That is, the agent will first turn to left and then try to move. Since the move is illegal, the agent stays put.


### Rules:
1. The agent must eat the food to grow.
2. If the agent collides with the opponent:
	- Let `s1` and `s2` be the scores of the two agents. 
	- If `s1 > s2`, `r1 = 5 s2/(s1-s2)` and `r2 = -3 s2/(s1-s2)`
	- If `s1 < s2`, `r1 = -3 s1/(s2-s1)` and `r2 = 5 s1/(s2-s1)`
3. After collison, the agent with the lesser score is randomly respawned.

### Reward System:
- `-1` for legal moves
- `-2` for illegal moves
- `+4` for consuming food
- Collision
	- If `s1 > s2`, `r1 = 5 s2/(s1-s2)` and `r2 = -3 s2/(s1-s2)`
	- If `s1 <s 2`, `r1 = -3 s1/(s2-s2)` and `r2 = 5 s1/(s2-s1)`

### Evaluation Metric:
- Every game lasts for a maximum of `game_length = 100` iterations.
- The agent with the greater score wins the game.

- Play `runs = 1000` games against the opponent.
- The agent with higher number of victories wins the bracket.

### Visualisation:

![VIDEO](https://user-images.githubusercontent.com/80670240/126877788-2e90d653-4c93-4ee8-bacf-8ff299d8bbb9.mp4)

