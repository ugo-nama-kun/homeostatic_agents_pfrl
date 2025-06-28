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

# "trp_env:SmallLowGearAntTRP-v0" or "thermal_regulation:SmallLowGearAntTHR-v3"
env_id = "trp_env:SmallLowGearAntTRP-v0" # "thermal_regulation:SmallLowGearAntTHR-v3"

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

if env_id == "trp_env:SmallLowGearAntTRP-v0":
    # dirname = "data/trp-homeostatic_shaped2021-08-17-19-14-42/150000000_finish"
    dirname = "data/result_trp_therm_oct2021/trp-homeostatic_shaped2021-09-28-17-38-42/150000000_finish"
else:
    # dirname = "data/therm-homeostatic_shaped2021-08-31-19-07-17/150000000_finish"
    dirname = "data/result_trp_therm_oct2021/therm-homeostatic_shaped2021-10-08-16-13-59/150000000_finish"

agent.load(dirname=dirname)

# Checking Learned Behavior
image_arrays = []

obs = env.reset()
done = False
for i in range(30000):
    action = agent.act(obs)
    obs, _, done, _ = env.step(action)
    if is_record:
        im = env.render(
            mode="rgb_array",
            height=1024,
            width=1024
        )
        image_arrays.append(im)
    else:
        env.render()
    if done:
        break

if is_record:
    import imageio
    imageio.mimsave("sample.mp4", np.array(image_arrays), fps=2.0 * int(np.round(1.0 / env.dt)))

env.close()

logger.info("done.")
