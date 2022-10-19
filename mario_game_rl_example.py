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

        self.memory = deque(maxlen=100000)      # ? Doubly Ended Queue - Type of list with faster append and pop operations. With max length of 100000 in this case.
        self.batch_size = 32

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.paramters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4       # ? min. experiences before training
        self.learn_every = 3    # ? no. of experiences between updates to Q_online
        self.sync_every = 1e4   # ? no. of experiences between Q_target & Q_online sync


    def act(self, state):
        # ? For any given state, an agent can choose to do the most optimal action(exploit) or a random action(explore)
        # ? Mario randomly explores with a chance of self.exploration_rate;
        # ? When he chooses to exploit, he relies on MarioNet (implemented in Learn Section)
        # * Given a state, choose an epsilon greedy action
        """

        Args:
            Input = state(LazyFrame): A single observation of the current state, dimension is (state_dim)
            Output = action_idx(int) : An integer representing which action Mario will perform
        """

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__() # ! __array__() returns either a new reference to self if dtype is not given or a new array of provided data type.
            # ? UNSQUEEZE -> Returns a new tensor with a dimension of size one inserted at the specified position. KIND OF LIKE FLATTEN
            state = torch.tensor(state, device=self.device).unsqueeze(0)    # ? torch.unsqueeze(input, dim) → Tensor 
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        # ? The cache and recall functions serve as Mario's memory process
        # ? cache() - Each time Mario performs an action, he stores the experience to his memory.
                # ?   The experience includes the current state, action performed, reward from the action, the next state and whether the game is done.
        
        """Stores the experience to self.memory

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

        self.memory.append((state, next_state, action, reward, done,))


    def recall(self):
        # Sample experiences from the memory
        # ? recall() - Mario randomly samples a batch of experiences from his memory, and uses that to learn the game.
        """Retrieve a batch of experience from memory

        """

        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))     # ? stack the tensors in each column containing info 
                                                                                    # ? regarding a single action into a single tensor tuple.
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()    # ? squeeze - remove a dimension of length 1.

    # ? Putting it all together.        
    def learn(self):
        # Update online action value (Q) function with a batch of experiences.
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        
        if self.curr_step % self.save_every == 0:
            self.save()
        
        if self.curr_step < self.burnin:
            return None, None
        
        # ! Sample from memory
        state, next_state, action, reward, done = self.recall()

        # ! GET TD Estimate
        td_est = self.td_estimate(state, action)

        # ! GET TD TARGET
        td_tgt = self.td_target(reward, next_state, done)

        # ! Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item, loss)


        pass
    
    # ? TD ESTIMATE & TD LEARNING
    # ? Two values are involved in learning
    # ? TD ESTIMATE - Optimal Q* for a given state 's' is TDe = Q*_online(s, a)
    # ? TD TARGET - Aggregation of current reward and estimated Q* in the next state s' is
    # ?                         a' = argmaxQonline(s', a)
    # ?                         TD_target = reward + discount x Q*+target(s', a')
    # ? Since we don't know what next action a' will be, we use the action a' that maximizes Qonline in the next state s'
    # ! We use the @torch.no_grad() decorator on td_target() to disable gradient calculations here. (BECAUSE we don't need to backpropagate on THETAtarget)
    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[        # ! Online learning is an approach used in Machine Learning that ingests sample of real-time data one observation at a time
            np.arange(0, self.batch_size), action
        ]   # Q_online(s, a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]       # ? Training output?

        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    # ! Updating the model
    # ? As Mario samples inputs from his replay buffer, we compute TD_target and TD_estimate and backpropagate this loss down
    # ? Qonline to update its parameters THETAonline (ALPHA is the learning rate lr passed to the optimizer)
    # ?     THETAonline <- THETAonline + ALPHA * del(TD_estimate - TD_target)
    # ? THETA_target does not update through backprop. Instead we periodically copy THETA_online to THETA_target
    
    def update_Q_online(self, td_estimate, td_target):
        # * CONSTANTLY?
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()      # ? Initialize
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    # ! SAVE THE CHECKPOINT
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )

        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"Mario net save to {save_path} at step {self.curr_step}")



# ! Let's now construct our Neural Network
# ? Mario uses DDQN Algorithm under the hood. DDQN uses two ConvNets - Qonline and Qtarget - that independently
# ? approximate the optimal action-value function

# ? In our implementation, we share feature generator features across Qonline and Qtarget, but maintain separate FC [fully connected] classifiers for each.
# ? THETAtarget (the parameters of Qtarget) is frozen to prevent updation by backprop. Instead it is periodically synced with THETAonline
class MarioNet(nn.Module):
    """mini cnn structure
    input -> (conv2d + relu) x 3 -> Flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim         # ? as evovled above
        if h != 84:
            raise ValueError(f"Expecting input height: 84 got {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84 got {w}")
        
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),     # kernel size is always prefered to be max in the beginning and
            nn.ReLU(),                                                              # and should be reduced in upcoming layers.
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)        # ? THETAs periodic syncing

        # Qtarget parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        # * THE FORWARD PASS
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

# ! |||||||<><><><><><><><>|||||||
# ! AND FINALLY IT'S TIME TO PLAY.
# ! |||||||<><><><><><><><>|||||||

# ? In this example we are running the loop for 10 episodes, but for Mario to truly learn the ways of his world,
# ? atleast 40,000 episodes should be played.

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()
save_dir = Path("checkpoints") / datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
logger = MetricLogger(save_dir)

episodes = 10
for e in range(episodes):

    state = env.reset()

    # * PLAY THE GAME
    while True:
        # * Run agent on the state
        action = mario.act(state)

        # * Agent performs the action
        next_state, reward, done, trunc, info = env.step(action)

        # * Remember
        mario.cache(state, next_state, action, reward, done)

        # * Learn
        q, loss = mario.learn()

        # * Logging
        logger.log_step(reward, loss, q)

        # * Update step
        state = next_state

        # * Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

