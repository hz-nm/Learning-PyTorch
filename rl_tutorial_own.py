"""
Author: Ameer H.
Description: This Python file will contain code from PyTorch's documentation about regarding Re-inforcement Learning Using PyTorch.
But instead of using the Mario Game, I will use some other game and program the code accordingly.

The code will follow similar style of the original MARIO RL Tutorial.

Elegant Robot's Code,
https://elegantrl.readthedocs.io/en/latest/tutorial/BipedalWalker-v3.html
"""

# ! Some facts about the LunarLander-v2
# ! Action Space
# * 1-Do Nothing, 2-Fire Left Rocket 3-Fire Main Engine 4-Fire Right Rocket

# ! Observation Space
# * 

from platform import release
import re
from tkinter import Frame
import torch
from torch import nn
from torchvision import transforms as T

from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque

import random, datetime, os, copy
import time
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# * Let's initialize the LunarLander Environment
if gym.__version__ < '0.26':
    env = gym.make("LunarLander-v2", render_mode="human", new_step_api=True, continuous=True)
else:
    env = gym.make("LunarLander-v2", render_mode="rgb", apply_api_compatibility=True, continous=True)

# ! Put the action space overe here
# ! Example Action -> np.array([main, lateral])
# ! if main < 0 - main thruster is turned off
# ! 0 <= main <= 1.
# ! lateral - Has two possibilities,
# ! if -1.0 < lateral < -0.5 -> Left booster will fire
# ! if 0.5 < lateral < 1 -> Right booster will fire.

# env.action_space = Box
# ! An example action space.
action_space_ex = env.action_space.sample()

# Maybe we don't need to define the action space here and we can simply,
# put random values in the act!


env.reset()

next_state, reward, done, trunc, info = env.step(action=0)
print(f"State Shape: {next_state.shape}. \n Reward: {reward}, \n Done: {done} \n Info: {info}")


# * Preprocess the Environment

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        # ? Return only every skip -th frame
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        # ? Repeat action and sum reward
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate all rewards and repeat the action
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward

            if done:
                break
            return obs, total_reward, done, trunc, info
    
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def permute_orientation(self, observation):
        # [H, W, C] to [C, H, W] tensor
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
            self.shape(shape, shape)
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

# ! Now we apply WRAPPERS to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)

if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

# ! Let's Create the Agent

class Lander:
    def __init__(self, state_dim, action_dim, save_dir):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = LanderNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5      # das

        self.memory = deque(maxlen=100000)

        self.batch_size = 32
        self.gamma = 0.9
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4
        self.learn_every = 3
        self.sync_every = 1e4

    
    def act(self, state):
        """Action

        Args:
            state (LazyFrame): A single observation of the current state.

        Returns:
            action_idx (int): Best action based on Explore or Exploit
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
            # ! maybe it will be,
            # action_idx = env.action_space.sample()
        
        #EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()

            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model='online')
            print(action_values)
            action_idx = torch.argmax(action_values, axis=1).item()

        
        # decrease the exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        """The landers memory

        Args:
            state (LazyFrame)
            next_state (LazyFrame)
            action (int)
            reward (float)
            done (bool)
        """

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)

        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """Sample experiences from memory by random samples

        Returns:
            Return a batch
        """

        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        """
        Putting it all together
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        # ! Sample from MEMORY
        state, next_state, action, reward, done = self.recall()

        # ! GET TD Estimate
        td_est = self.td_estimate(state, action)

        # ! GET TD Target
        td_trgt = self.td_target(reward, next_state, done)

        # ! Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_trgt)

        return (td_est.mean().item, loss)

    # ! TD Estimate & TD Learning
    # * Disable gradient to avoid backprop
    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[
            np.arange(0, self.batch_size), action
        ]   # Q_online
        return current_Q
    
    @torch.no_grad
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[
            np.arange(0, self.batch_size), best_action
        ]

        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    # ! Update the model
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    # ! Save the checkpoint
    def save(self):
        save_path = (
            self.save_dir / f"lander_net_{int(self.curr_step // self.save_every)}"
        )

        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Lander Net save to {save_path} at step {self.curr_step}")

# ! FINALLY THE LanderNet DQNN Algorithm
class LanderNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != 84:
            raise ValueError
        if w != 84:
            raise ValueError

        # !! YE HAI ONLINE
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        # * The forward pass
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)


# ! ------------------------
# ! LOGGING
# ! ------------------------
class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode ' :>8}{'Step ' :>8}{'Epsilon' :>10}{'MeanReward' :>15}"
                f"{'MeanLength ':>15}{'MeanLoss ':>15}{'MeanQValue ':>15}"
                f"{'TimeDelta ':>15}{'Time ':>20}\n" 
            )

        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History Metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call ti record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # current episode metric
        self.init_episode()

        # TIMING
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1

        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        # Mark end of episode
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)

        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_qs[-100:]), 3)

        self.moving_avg_ep_avg_rewards.append(mean_ep_reward)
        self.moving_avg_ep_avg_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - ",
            f"Step {step} - ",
            f"Epsilon {epsilon} - ",
            f"Mean Reward {mean_ep_reward} - ",
            f"Mean Length {mean_ep_length} - ",
            f"Mean Loss {mean_ep_loss} - ",
            f"Mean Q Value {mean_ep_q} - ",
            f"Time Delta {time_since_last_record} - ",
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d} {step:8d} {epsilon:10.3f}"
                f"{mean_ep_reward:15.3f} {mean_ep_length:15.3f} {mean_ep_loss:15.3f} {mean_ep_q:15.3f}",
                f"{time_since_last_record:15.3f}",
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))        # ? GET THE ATTRIBUTE NAMED metric.  getattr(x, 'y' simply means x.y but I guess some difference in classes)
            plt.clf()


# ! ~><~~><~~><~~><~~><~~><~~><~~><~~><~~>
# ? TIME TO PLAY
# ! ~><~~><~~><~~><~~><~~><~~><~~><~~><~~>

use_cuda = torch.cuda.is_available()
print(f"Are we USING CUDE? \n{use_cuda}")
print()
save_dir = Path("checkpoints") / datetime.date.now().strftime('%Y-%m-%dT%H:%M:%S')
save_dir.mkdir(parents=True)

lander = Lander(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
logger = MetricLogger(save_dir=save_dir)

episodes = 10
for e in range(episodes):
    state = env.reset()

    # * PLAY THE GAME
    while True:
        # * Run agent on the state
        action = lander.act(state)

        # * Agent performs the action
        next_state, reward, done, trunc, info = env.step(action)

        # * Remember
        lander.cache(state, next_state, action, reward, done)

        # * Learn
        q, loss = lander.learn()

        # * Log
        logger.log_step(reward, loss, q)

        # * Update step
        state = next_state

        # * Check if end of GAME
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=lander.exploration_rate, step=lander.curr_step)





        

    
