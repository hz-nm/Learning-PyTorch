"""_summary_
Link -> https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
This tutorial walks us through the fundamentals of Deep Reinforcement Learning.
At the end we will implement an AI-Powered Mario (using Double Deep Q-Networks) that can play the game by itself.

Look at the CHEATSHEET here, -> https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N
More Related RL Concepts here, -> https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
Full code -> https://github.com/yuansongFeng/MadMario/
             https://github.com/pytorch/tutorials/blob/master/intermediate_source/mario_rl_tutorial.py
Colab -> https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/c195adbae0504b6504c93e0fd18235ce/mario_rl_tutorial.ipynb

"""

from tkinter import Frame
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

import gym      # ? An OpenAI toolkit for RL
from gym.spaces import Box
from gym.wrappers import FrameStack

from nes_py.wrappers import JoypadSpace         # ! Not working currently as Visual Studio 14.0 or more is required.
import gym_super_mario_bros

# ! Some Definitions to keep in mind,
# ? Environment - The world that an agent reacts with and learns from.
# ? State - s - The current characteristics of an environment. The set of all possible states is called a state space.
# ? Action - a - How the agent responds the environment
# ? Reward - r - Reward is the feedback that the agent gets from the environment.
# ? Optimal Action Value Function - Q*(s, a) - Gives the expected return if you start in a state s.

# %%
# * Let's initialize the Super Mario Environment.
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)

# Limit the action space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A'"]])
env.reset()

next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape}. \n {reward}, \n {done}, \n {info}")

# %%
# * Preprocess the Environment
# Environment data is returned to the agent in next_state.
# Each state is represented by a [3, 240, 256] size array. Often that is more than we need since the color of the sky or the pipes don't matter to mario
# We use **WRAPPERS** to preprocess the environment data before sending it to the agent.

# GrayScaleObservation is a common wrapper to transform an RGB Image to Grayscale. This in turn reduces size of the data and hence the complexity.
# ResizeObservation downsamples each observation into a square image.
# SkipFrame is a custom wrapper that inherits from gym.Wrapper and implements the step() function. Since consecutive frames don't vary that much.
# FrameStack is a wrapper that allows us to quash consecutive frames of the environment into a single observation point to feed to our learning model.
#       This way we can identify if Mario was landing or jumping based on the direction of his movement in the previous several Frames.

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        # ? Return only every skip -th frame
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        # ? Repeat action and sum reward
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward

            if done:
                break
            return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # Permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)


# * After applying the above wrappers to the environment, the final wrapped state consists of 4 Gray-Scaled consecutive frames
# * stacked together.
# * The structure is represented by a 3-D Array of size [4, 84. 84].

# ! Agent
# * We will create a class Mario to represent our agent in the game. Mario should be able to:

# ? ACT - according to optimal action policy based on the current state (of the environment)
# ? REMEMBER - experiences. Experience = (current state, current action, reward, next_state). Mario caches and later recalls his experiences to update his action policy.
# ? LEARN - A better action policy over time.

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ! Let's now create Mario's DNN to predict the most optimal action - We implement this in the learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()        # ? Implemented in the Learn Section
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5       # no. of experiences between saving MarioNet


    def act(self, state):
        # ? For any given state, an agent can choose to do the most optimal action(exploit) or a random action(explore)
        # ? Mario randomly explores with a chance of self.exploration_rate;
        # ? When he chooses to exploit, he relies on MarioNet (implemented in Learn Section)
        # Given a state, choose an epsilon greedy action
        """

        Args:
            Input = state(LazyFrame): A single observation of the current state, dimension is (state_dim)
            Output = action_idx(int) : An integer representing which action Mario will perform
        """

        # EXPLORE

        pass
    def cache(self, experience):
        # Add experience to the memory
        pass
    def recall(self):
        # Sample experiences from the memory
        pass
    def learn(self):
        # Update online action value (Q) function with a batch of experiences.
        pass


