from environment import Vasuki

from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
from nqueens import Queen

import numpy as np
import random
import os

import matplotlib
import matplotlib.pyplot as plt

import cv2

from collections import namedtuple, deque
from itertools import count
from base64 import b64encode

# ----------------------------------------------------- #

class Runner():
    def __init__(self, model_A, model_B, checkpoint):
        # Path to store the Video
        self.checkpoint = checkpoint
        # Defining the Environment
        config = {'n': 8, 'rewards': {'Food': 4, 'Movement': -1, 'Illegal': -2}, 'game_length': 100} # Should not change for evaluation
        self.env = Vasuki(**config)
        self.runs = 1000
        # Trained Policies
        self.model_A = model_A # Loaded model with weights
        self.model_B = model_B # Loaded model with weights
        # Results
        self.winner = {'Player_A': 0, 'Player_B': 0}

    def reset(self):
        self.winner = {'Player_A': 0, 'Player_B': 0}

    def evaluate_A(self):
        # Uses self.env as the environment and returns the best action for Player A (Blue)

        return action_A # Action in {0, 1, 2}

    def evaluate_B(self):
        # Uses self.env as the environment and returns the best action for Player B (Red)
        
        return action_B # Action in {0, 1, 2}

    def visualize(self, run):
        self.env.reset()
        done = False
        video = []
        while not done:
            # Actions based on the current state using the learned policy 
            actionA = self.evaluate_A()
            actionB = self.evaluate_B()
            action = {'actionA': actionA, 'actionB': actionB}
            rewardA, rewardB, done, info = self.env.step(action)
            # Rendering the enviroment to generate the simulation
            if len(self.env.history)>1:
                state = self.env.render(actionA, actionB)
                encoded, _ = self.env.encode()
                state = np.array(state, dtype=np.uint8)
                video.append(state)
        # Recording the Winner
        if self.env.agentA['score'] > self.env.agentB['score']:
            self.winner['Player_A'] += 1
        elif self.env.agentB['score'] > self.env.agentA['score']:
            self.winner['Player_B'] += 1
        # Generates a video simulation of the game
        if number%100==0:
            aviname = os.path.join(self.checkpoint, f"game_{run}.avi")
            mp4name = os.path.join(self.checkpoint, f"game_{run}.mp4")
            w, h, _ = video[0].shape
            out = cv2.VideoWriter(aviname, cv2.VideoWriter_fourcc(*'DIVX'), 2, (h, w))
            for state in video:
                assert state.shape==(256,512,3)
                out.write(state)
            cv2.destroyAllWindows()
            os.popen("ffmpeg -i {input} {output}".format(input=aviname, output=mp4name))
            # os.popen("rm -f {input}".format(input=aviname))

    def arena(self):
        # Pitching the Agents against each other
        for run in range(1, self.runs+1, 1):
            self.visualize(run)
        return self.winner