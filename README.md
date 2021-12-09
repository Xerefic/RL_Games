# Shaastra 2022
# RL Games: Vasuki

## Description: 

Are you fed up with friends snakes betraying you? This is your chance to redeem yourself and rise above the rest to crown yourself as the King of the Snakes Vasuki!

We present to you RL Games, an arena where you will find only blood and venom. Create your own bot to fight against the best of the best bots. With a cash prize pool of INR 25K, what more do you need?

As a part of Shaastra 2022, RL games is a team competition where participants compete to find the best policy to a given environment using reinforcement learning methods. Do check out our [website](www.shaastra.org) for other exciting opportunities!

## Scripts

- Link to notebook: [![Open All Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B-Ohlwsehrg_eJwAeVQZM8ZmzO5GBrNn?usp=sharing)
- [Envorionment](environment.py)

### State Space:
- n*n Grid (Continious Space).
- At any instant of time, n random coordinates out of 2n fixed coordinates have food.

### Action Space:
- Move forward by v blocks in 1 second.
- Decide to turn right or left for the next timestep.

### Rules:
	1. The agent must eat the food to grow.
	2. If the agent collides with the opponent:
		Let s1 and s2 be the scores of the two agents. 
		If s1>s2, s1 -> s1 + s2/3 and s2 -> s2/2
	3. After collison, the agent with the lesser score is randomly respawned.
	4. The dimensions of the agent is directly proportional to the score it possess at any instant.
		If s < 10 -> 1*1
		If 10 < s < 20 -> 1*2
		If 20 < s < 30 -> 1*3 ...

### Reward System:
- (-1) for legal moves
- (-2) for illegal moves
- (+4) for consuming food
- (+s/3) or (-s/2) for collision

### Evaluation Metric:
- Every game lasts for a maximum of max_iter=1000 iterations.
- The agent with the greater score wins the game.

- Play max_games=100 games against the opponent.
- The agent with higher number of victories wins the bracket.

### Visualisation:

![caption](https://user-images.githubusercontent.com/80670240/126877788-2e90d653-4c93-4ee8-bacf-8ff299d8bbb9.mp4)

