import argparse
import functools
import os
import random
from datetime import datetime
import logging
import sys

import gym
import numpy as np

import pfrl

import torch
from gym.wrappers import RescaleAction
from pfrl.experiments import LinearInterpolationHook
from torch import nn

from util.experiment import train_agent_batch_with_evaluation
from util.ppo import PPO_KL
from util.modules import BetaPolicyModel, ortho_init


logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

logger = logging.getLogger(__name__)

#####################################################
# Argparse
#####################################################

parser = argparse.ArgumentParser(description="TRP Experiment Setting")
parser.add_argument("--reward_setting",
                    choices=["hs", "h", "one", "hb", "g"],
                    help="Reward settings. It should be homeostatic_shaped (hs), homeostatic (h), one (o), homeostatic_biased (hb) or greedy (g)",
                    required=True)
parser.add_argument("--seed", type=int, help="seed value. integer", required=True)
parser.add_argument("--gpu", type=int, help="id of gpu. integer. -1 if use cpu.", required=True)
parser.add_argument("--ident", type=str, help="put group identifier", required=True)
parser.add_argument("--test", action='store_true')
parser.add_argument("--offline", action="store_true")

args = parser.parse_args()

is_test = args.test
is_offline = args.offline

reward_scale = None
reward_bias = None
reward_name = None
if args.reward_setting == "hs":
    reward_name = "homeostatic_shaped"
    reward_scale = 100
    reward_bias = None
elif args.reward_setting == "h":
    reward_name = "homeostatic"
    reward_scale = 0.1
    reward_bias = None
elif args.reward_setting == "one":
    reward_name = "one"
    reward_scale = 100
    reward_bias = None
elif args.reward_setting == "hb":
    reward_name = "homeostatic_biased"
    reward_scale = 1
    reward_bias = 0.1
elif args.reward_setting == "g":
    reward_name = "greedy"
    reward_scale = 1
    reward_bias = None
else:
    raise ValueError("I have to specify a reward setting!")

logger.info(f"Reward setting: {reward_name},Reward Scale: {reward_scale}, Reward Bias: {reward_bias}")
logger.info(f"Seed: {args.seed}")
#####################################################
# Parameters
#####################################################

seed = args.seed
env_id = "thermal_regulation:SmallLowGearAntTHR-v3"

# PPO Parameters
update_interval = 10000 if is_test else 300000
minibatch_size = 1000 if is_test else 50000
sgd_epochs = 30

learning_rate = 0.0003
gamma = 0.99
lambd = 0.95

clip_eps = 0.3
clip_eps_vf = 10.0

entropy_coeff = 0.001
coeff_vf = 0.5
kl_coef = 0.001

max_grad_norm = 0.5
eps_adam = 1e-5

# Sampling in the training params
if is_test:
    device = None
elif args.gpu == -1:
    device = None
else:
    device = args.gpu

n_env = 3 if is_test else 10

n_iterations = 3 if is_test else 500
max_steps = update_interval * n_iterations  # steps for 1000 iterations

# Test operation params
n_eval_runs = 3 if is_test else 10  # number of episodes for evaluation
eval_interval = update_interval if is_test else update_interval * 10  # Evaluate every 10 iterations
maximum_evaluation_steps = 60000  # maximum timesteps in evaluation

# Environment params

temp_diff = (42. - 38.) / 6.0
env_config = {"max_episode_steps": np.inf,
              "internal_reset": "random",
              "reward_setting": reward_name,
              "reward_bias": reward_bias,
              "coef_main_rew": reward_scale,
              "coef_ctrl_cost": 0.001,
              "coef_head_angle": 0.005,
              "energy_random_range": (-1 / 6., 1 / 6.),
              "temperature_random_range": (38. - temp_diff, 38. + temp_diff)}

no_done_at_end = False
if env_config["reward_setting"] in {"homeostatic_shaped", "homeostatic_biased", "homeostatic"}:
    no_done_at_end = True  # use reset at the terminal of the environment

#####################################################
# Setup config
#####################################################

# project_name = "thermal_wo_shade"  # wandb thermal_regulation 3.2.0-ns branch results. This is used in DeepRL workshop 2021
project_name = "thermal_random_climate"

if is_offline:
    os.environ["WANDB_MODE"] = "offline"
else:
    os.environ["WANDB_MODE"] = "run"

config = {
    "seed": seed,
    "env_id": env_id,
    "update_interval": update_interval,
    "minibatch_size": minibatch_size,
    "sgd_epochs": sgd_epochs,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "lambda": lambd,
    "clip_eps": clip_eps,
    "clip_eps_vf": clip_eps_vf,
    "entropy_coeff": entropy_coeff,
    "coeff_vf": coeff_vf,
    "kl_coef": kl_coef,
    "max_grad_norm": max_grad_norm,
    "eps_adam": eps_adam,
    "device": device,
    "n_env": n_env,
    "n_eval_runs": n_eval_runs,
    "eval_interval": eval_interval,
    "maximum_evaluation_steps": maximum_evaluation_steps,
    "n_iterations": n_iterations,
    "max_steps": max_steps,
    "no_done_at_end": no_done_at_end
}

config.update(env_config)

#####################################################
# Seed
#####################################################

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

process_seeds = np.arange(n_env) + seed * n_env


#####################################################
# Env Util
#####################################################


def make_env(process_idx, is_eval_run):
    if is_eval_run:
        env = gym.make(
            env_id,
            max_episode_steps=maximum_evaluation_steps,
            internal_reset=env_config["internal_reset"],
            reward_setting=env_config["reward_setting"],
            reward_bias=env_config["reward_bias"],
            coef_main_rew=env_config["coef_main_rew"],
            coef_ctrl_cost=env_config["coef_ctrl_cost"],
            coef_head_angle=env_config["coef_head_angle"],
            energy_random_range=env_config["energy_random_range"],
            temperature_random_range=env_config["temperature_random_range"],
            n_bins=20,
            sensor_range=16,
            random_climate=True
        )
    else:
        env = gym.make(
            env_id,
            max_episode_steps=env_config["max_episode_steps"],
            internal_reset=env_config["internal_reset"],
            reward_setting=env_config["reward_setting"],
            reward_bias=env_config["reward_bias"],
            coef_main_rew=env_config["coef_main_rew"],
            coef_ctrl_cost=env_config["coef_ctrl_cost"],
            coef_head_angle=env_config["coef_head_angle"],
            energy_random_range=env_config["energy_random_range"],
            temperature_random_range=env_config["temperature_random_range"],
            n_bins=20,
            sensor_range=16,
            random_climate=True
        )

    process_seed = int(process_seeds[process_idx])

    env_seed = 2 ** 32 - 1 - process_seed if is_eval_run else process_seed

    env.seed(env_seed)

    env = RescaleAction(env, 0, 1)
    env = pfrl.wrappers.CastObservationToFloat32(env)

    return env


def make_batch_env(is_eval_run):
    return pfrl.envs.MultiprocessVectorEnv(
        [
            functools.partial(make_env, idx, is_eval_run) for idx in range(n_env)
        ]
    )


#####################################################
# Network initialization
#####################################################


dummy_env = gym.make(
    env_id,
    max_episode_steps=env_config["max_episode_steps"],
    internal_reset=env_config["internal_reset"],
    reward_setting=env_config["reward_setting"],
    reward_bias=env_config["reward_bias"],
    coef_main_rew=env_config["coef_main_rew"],
    coef_ctrl_cost=env_config["coef_ctrl_cost"],
    coef_head_angle=env_config["coef_head_angle"],
    energy_random_range=env_config["energy_random_range"],
    temperature_random_range=env_config["temperature_random_range"],
    n_bins=20,
    sensor_range=16,
)

obs_space = dummy_env.observation_space
action_space = dummy_env.action_space

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


# Orthogonal Initialization of Weights
ortho_init(value_func[0], gain=1)
ortho_init(value_func[2], gain=1)
ortho_init(value_func[4], gain=1)

model = pfrl.nn.Branched(policy, value_func)

##########################################################
# Make an agent
##########################################################

opt = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=eps_adam)

agent = PPO_KL(
    model=model,
    optimizer=opt,
    gpu=device,
    lambd=lambd,
    update_interval=update_interval,
    minibatch_size=minibatch_size,
    epochs=sgd_epochs,
    clip_eps=clip_eps,
    clip_eps_vf=clip_eps_vf,
    value_func_coef=coeff_vf,
    entropy_coef=entropy_coeff,
    kl_coef=kl_coef,
    standardize_advantages=True,
    max_grad_norm=max_grad_norm,
)

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def lr_setter(env, agent, value):
    agent.optimizer.lr = value


lr_hook = LinearInterpolationHook(max_steps, learning_rate, 1e-5, lr_setter)

train_agent_batch_with_evaluation(
    agent=agent,
    env=make_batch_env(is_eval_run=False),
    eval_env=make_batch_env(is_eval_run=True),
    outdir="results/" + "therm-" + reward_name + timestamp,
    steps=max_steps,
    eval_n_steps=None,
    eval_n_episodes=n_eval_runs,
    eval_interval=eval_interval,
    max_episode_len=None,
    save_best_so_far_agent=False,
    project_name=project_name,
    entity_name="ugo-nama-kun",
    group_name="test-run-" + reward_name if is_test else reward_name + "-" + args.ident,
    run_name="run-" + timestamp,
    experiment_config=config,
    no_done_at_end=no_done_at_end,
    log_interval=update_interval,
    step_hooks=[lr_hook],
)
