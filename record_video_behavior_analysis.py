import logging
import random
import sys

import numpy as np

import gym
import pfrl

import torch

from torch import nn
from gym.wrappers import RescaleAction

from util.ppo import PPO_KL
from util.modules import BetaPolicyModel


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

##########################################################
# Parameters
##########################################################

is_record = False

blue = 0.1
red = -0.1

TIME_STEPS = 300
NUM_TICK = 11
NUM_SAMPLE = 100

RADIUS = 5
NUM_FOOD = 6

env_id = "conditioned_trp_env:SmallLowGearAntCTRP-v0"

##########################################################
# Seed
##########################################################

seed = 100

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

env = gym.make(
    env_id,
    max_episode_steps=np.inf,
    internal_reset="setpoint",
    n_bins=20,
    sensor_range=16,
)
env = RescaleAction(env, 0, 1)
env = pfrl.wrappers.CastObservationToFloat32(env)

obs_space = env.observation_space
action_space = env.action_space

obs_size = obs_space.low.size
action_size = action_space.low.size

policy = BetaPolicyModel(obs_size=obs_size,
                         action_size=action_size,
                         hidden1=256,
                         hidden2=64)

value_func = torch.nn.Sequential(
    nn.Linear(obs_size, 256),
    nn.Tanh(),
    nn.Linear(256, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

model = pfrl.nn.Branched(policy, value_func)

opt = torch.optim.Adam(model.parameters())

agent = PPO_KL(
    model=model,
    optimizer=opt,
    gpu=-1,
)

dirname = "data/result_trp_therm_oct2021/trp-homeostatic_shaped2021-09-28-17-38-42/150000000_finish"

agent.load(dirname=dirname)

# Checking Learned Behavior

while True:
    obs = env.reset(initial_internal=(blue, red),
                    object_positions={"blue": [(RADIUS * np.cos(a), RADIUS * np.sin(a)) for a in
                                               np.random.uniform(-np.pi, np.pi, NUM_FOOD)],
                                      "red": [(RADIUS * np.cos(a), RADIUS * np.sin(a)) for a in
                                              np.random.uniform(-np.pi, np.pi, NUM_FOOD)]})
    done = False
    for i in range(TIME_STEPS):
        action = agent.act(obs)

        obs, _, done, _ = env.step(action)

        env.render()
        if done:
            break

