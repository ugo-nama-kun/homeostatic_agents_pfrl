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

import torch  # torch should be <= v1.8.2

from gym.wrappers import RescaleAction
from pfrl.experiments import LinearInterpolationHook
from torch import nn

from util.env import VisionEnvWrapper, to_multi_modal
from util.experiment import train_agent_batch_with_evaluation
from util.ppo import PPO_EXT, PPO_KL
from util.modules import BetaPolicyModel, ortho_init, ResidualLayer, RandomShiftsAug

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

logger = logging.getLogger(__name__)

#####################################################
# Argparse
#####################################################

parser = argparse.ArgumentParser(description="TRP Experiment Setting")
parser.add_argument("--reward_setting",
                    choices=["hs", "h", "one", "hb", "g"],
                    default="hs",
                    help="Reward settings. It should be homeostatic_shaped (hs), homeostatic (h), one (o), homeostatic_biased (hb) or greedy (g)")
parser.add_argument("--seed", type=int, help="seed value. integer", required=True)
parser.add_argument("--gpu", type=int, help="id of gpu. integer. -1 if use cpu.", default=-1)
parser.add_argument("--ident", type=str, help="put group identifier", default="test")
parser.add_argument("--test", action='store_true')
parser.add_argument("--offline", action="store_true")
parser.add_argument("--enable_checkpoint", action="store_true")

args = parser.parse_args()

is_test = args.test
is_offline = args.offline
enable_checkpoint = args.enable_checkpoint

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
env_id = "trp_env:SmallLowGearAntTRP-v0"

# PPO Parameters
update_interval = 1000 if is_test else 20000
minibatch_size = 100 if is_test else 10000

sgd_epochs = 30

learning_rate = 0.0001
gamma = 0.99
lambd = 0.95

clip_eps = 0.3
clip_eps_vf = 10.0

entropy_coeff = 0.005
coeff_vf = 0.5
kl_coef = 0.001

max_grad_norm = 0.5
eps_adam = 1e-5

dim_vision_flatten = 200
n_frame_stack = 3
im_size = (64, 64)
n_channel = 4  # RGBD

# Sampling in the training params
if args.gpu == -1:
    device = None
else:
    device = args.gpu

print("Device GPU: ", device)

n_env = 2 if is_test else 10
n_env_eval = 2 if is_test else 10

n_iterations = 10 if is_test else 1000
max_steps = update_interval * n_iterations  # steps for n_iterations

# Test operation params
n_eval_runs = 2 if is_test else 10  # number of episodes for evaluation

eval_interval = update_interval * 3 if is_test else update_interval * 10  # Evaluate every 10 iterations
maximum_evaluation_steps = 60000  # maximum timesteps in evaluation

checkpoint_freq = update_interval * 50 if enable_checkpoint else None  # save checkpoint every 50 iterations

# Environment params

env_config = {"max_episode_steps": np.inf,
              "internal_reset": "random",
              "reward_setting": reward_name,
              "reward_bias": reward_bias,
              "coef_main_rew": reward_scale,
              "coef_ctrl_cost": 0.001,
              "coef_head_angle": 0.005,
              "internal_random_range": (-1. / 6, 1. / 6)}

no_done_at_end = False
if env_config["reward_setting"] in {"homeostatic_shaped", "homeostatic_biased", "homeostatic"}:
    no_done_at_end = True  # use reset at the terminal of the environment

#####################################################
# Setup config
#####################################################

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
    "no_done_at_end": no_done_at_end,
    # "max_recurrent_sequence_len": max_recurrent_sequence_len,
    "n_frame_stack": n_frame_stack,
    "im_size": im_size,
    "note": "rgbd" if n_channel == 4 else "rgb",
    "dim_vision_flatten": dim_vision_flatten,
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
            internal_random_range=env_config["internal_random_range"],
            n_bins=2,  # some useless value
            sensor_range=0.01,  # some useless value
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
            internal_random_range=env_config["internal_random_range"],
            n_bins=2,  # some useless value
            sensor_range=0.01,  # some useless value
        )

    process_seed = int(process_seeds[process_idx])

    env_seed = 2 ** 32 - 1 - process_seed if is_eval_run else process_seed

    env.seed(env_seed)

    env = RescaleAction(env, 0, 1)
    env = VisionEnvWrapper(env,
                           n_frame=n_frame_stack,
                           im_size=im_size,
                           mode="rgbd_array" if n_channel == 4 else "rgb_array")
    env = pfrl.wrappers.CastObservationToFloat32(env)

    return env


def make_batch_env(is_eval_run):
    return pfrl.envs.MultiprocessVectorEnv(
        [
            functools.partial(make_env, idx, is_eval_run) for idx in range(n_env_eval if is_eval_run else n_env)
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
    internal_random_range=env_config["internal_random_range"],
    n_bins=2,  # Jut setting useless config
    sensor_range=0.01,  # Jut setting useless config
)

obs_size_prop = 27
obs_size_intero = 2
action_size = 8


# vision encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.flatten_size = 26912  # 35 * 35 * 32

        # random shift augmentation
        self.aug = RandomShiftsAug(pad=2)

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(in_channels=n_channel * n_frame_stack, out_channels=32, kernel_size=3, stride=2, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, bias=False),
            nn.ELU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=self.flatten_size, out_features=dim_vision_flatten),
            nn.LayerNorm(normalized_shape=dim_vision_flatten),
            nn.Tanh()
        )

        # self.vision_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=n_channel * n_frame_stack, out_channels=32, kernel_size=3, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, 3, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, 3, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 32, 3, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Flatten(),
        # )

        # self.vision_encoder = nn.Sequential(
        #     ResidualLayer(in_channels=n_channel * n_frame_stack, out_channels=32),
        #     nn.ReLU(inplace=True), ResidualLayer(in_channels=32, out_channels=32),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.ReLU(inplace=True), ResidualLayer(in_channels=32, out_channels=32),
        #     nn.ReLU(inplace=True), ResidualLayer(in_channels=32, out_channels=32),
        #     nn.ReLU(inplace=True), nn.Flatten(),
        # )

        # Initialization
        for l in self.vision_encoder:
            if isinstance(l, nn.Conv2d):
                gain = nn.init.calculate_gain('relu')
                nn.init.orthogonal_(l.weight, gain=gain)
                if hasattr(l.bias, "data"):
                    l.bias.data.fill_(0.0)
            if isinstance(l, nn.Linear):
                nn.init.orthogonal_(l.weight, gain=1)
                nn.init.zeros_(l.bias)
            if isinstance(l, ResidualLayer):
                gain = nn.init.calculate_gain('relu')
                nn.init.orthogonal_(l.conv.weight, gain=gain)
                if l.downsample:
                    nn.init.orthogonal_(l.downsample.weight, gain)

    def forward(self, x, enable_aug=False):
        with torch.no_grad():
            x_prop, x_intero, x_vision = to_multi_modal(x,
                                                        im_size=im_size,
                                                        n_frame=n_frame_stack,
                                                        n_channel=n_channel)

        if enable_aug:
            x_vision = self.aug(x_vision.float())

        feature_vision = self.vision_encoder(x_vision)
        # print("feature", feature_vision.shape)

        return feature_vision, x_prop, x_intero


dim_joint_feature = dim_vision_flatten + obs_size_prop + obs_size_intero


# compose a total model
class FFModel(nn.Module):
    def __init__(self):
        super(FFModel, self).__init__()

        self.encoder = Encoder()

        self.policy_network = BetaPolicyModel(obs_size=dim_joint_feature,
                                              action_size=action_size,
                                              activation=nn.Tanh,
                                              hidden1=300,
                                              hidden2=200)

        self.value_network = nn.Sequential(
            nn.Linear(dim_joint_feature, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 1)
        )

        ortho_init(self.value_network[0], gain=1)
        ortho_init(self.value_network[2], gain=1)
        ortho_init(self.value_network[4], gain=1)

    def forward(self, x, enable_aug=False):
        # enable data augmentation while training
        feature_vision, x_prop, x_intero = self.encoder(x, enable_aug)

        # policy path
        x_policy = torch.cat((feature_vision.detach(), x_prop, x_intero), dim=1)
        distribs = self.policy_network(x_policy)

        # value path
        x_value = torch.cat((feature_vision, x_prop, x_intero), dim=1)
        vs_pred = self.value_network(x_value)

        return (distribs, vs_pred)


model = FFModel()

#####################################################
# Model
#####################################################

from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


count_parameters(model)

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
    recurrent=False,
)

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def lr_setter(env, agent, value):
    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = value


# linearly decrease learning rate during first 1000 iterations
lr_hook = LinearInterpolationHook(update_interval * 1000, learning_rate, 1e-5, lr_setter)

train_agent_batch_with_evaluation(
    agent=agent,
    env=make_batch_env(is_eval_run=False),
    eval_env=make_batch_env(is_eval_run=True),
    outdir="results/" + "vision-" + reward_name + timestamp,
    checkpoint_freq=checkpoint_freq,
    steps=max_steps,
    eval_n_steps=None,
    eval_n_episodes=n_eval_runs,
    eval_interval=eval_interval,
    max_episode_len=None,
    save_best_so_far_agent=True,
    project_name="vision_trp",
    entity_name="ugo-nama-kun",
    group_name="test-run-" + reward_name if is_test else reward_name + "-" + args.ident,
    run_name="run-" + timestamp,
    experiment_config=config,
    no_done_at_end=no_done_at_end,
    log_interval=update_interval,
    step_hooks=[lr_hook],
)
