# train.py
# Script to train policie   s in Isaac Gym
#
# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import datetime

import numpy as np


import random
import os
import time

import torch

from hydra.utils import to_absolute_path

from gym.spaces import Box
from hydra.experimental import compose, initialize

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.rlgames_utils import RLGPUAlgoObserver

from rl_games.common import tr_helpers
from rl_games.algos_torch.players import PpoPlayerContinuous

from robot import Robot
## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
pick_cfg = None
place_cfg = None
screw_cfg = None

def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')


# use the gripper pose(end effect) to create the fingerpoint pose
# the linvel and angel is calculated (has no domination)
def state_kinova2panda(pose, extra_info, pre_pose=None) :
    poselist = [pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    linvellist = [0.1, 0.1, 0.1]
    angvellist = [0.1, 0.1, 0.1]
    poselist.extend(linvellist)
    poselist.extend(angvellist)
    poselist.extend(extra_info)

    return torch.FloatTensor(poselist)

def check_pick_success(kinova_gripper_pose, tolerance=0.01):
    if kinova_gripper_pose.position.x >= -tolerance and kinova_gripper_pose.position.x <=tolerance:
        if kinova_gripper_pose.position.y >= -tolerance and kinova_gripper_pose.position.y <= tolerance:
            if kinova_gripper_pose.position.z >= -tolerance and kinova_gripper_pose.position.z <= tolerance:
                return True
    else:
        return False
def play(player, env, task_config):

    is_determenistic = player.is_determenistic
    need_init_rnn = player.is_rnn

    extral_info = []
    if task_config == "config_Pick.yaml":
        extral_info = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]  # set the nut [position, orientation] at the fix position ([0,0,0])

    kinova_gripper_pose = env.gripper_group.get_current_pose().pose
    kinova_arm_pose = env.arm_group.get_current_pose().pose

    panda_obs = state_kinova2panda(kinova_gripper_pose, extra_info=extral_info)
    obs = panda_obs.to(player.device)

    if need_init_rnn:
        player.init_rnn()

    done = True
    for n in range(player.max_steps):

        action = player.get_action(obs, is_determenistic)

        action = action.detach().cpu().numpy()
        position_add = action[0:3]

        goal_position = [kinova_arm_pose.position.x+position_add[0], kinova_arm_pose.position.y+position_add[1], kinova_arm_pose.position.z+position_add[2]]
        done&=env.move(pose = goal_position, tolerance=0.0001)

        kinova_gripper_pose = env.gripper_group.get_current_pose().pose
        kinova_arm_pose = env.arm_group.get_current_pose().pose

        if task_config == "config_Pick.yaml":
            extral_info = extral_info

        panda_obs = state_kinova2panda(kinova_gripper_pose, extra_info=extral_info)
        obs = panda_obs.to(player.device)

        if task_config == "config_Pick.yaml":
            if check_pick_success(kinova_gripper_pose, tolerance=0.01):
                done &=env.reach_gripper_position(0.465)
                goal_position = [kinova_arm_pose.position.x,
                                 kinova_arm_pose.position.y,
                                 kinova_arm_pose.position.z + 0.3]
                done &= env.move(pose=goal_position, tolerance=0.001)
                return done

    return False

def completeconfig(cfg):

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    # sets seed. if seed is -1 will pick a random one
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    # register the rl-games adapter to use inside the runner

    return cfg

def load_config(params):
    seed = params.get('seed', None)
    if seed is None:
        seed = int(time.time())
    if params["config"].get('multi_gpu', False):
        seed += int(os.getenv("LOCAL_RANK", "0"))
    print(f"self.seed = {seed}")

    if seed:

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # deal with environment specific seed if applicable
        if 'env_config' in params['config']:
            if not 'seed' in params['config']['env_config']:
                params['config']['env_config']['seed'] = seed
            else:
                if params["config"].get('multi_gpu', False):
                    params['config']['env_config']['seed'] += int(os.getenv("LOCAL_RANK", "0"))

    config = params['config']
    config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
    if 'features' not in config:
        config['features'] = {}
    config['features']['observer'] = RLGPUAlgoObserver()
    return params

def load_player(cfg, env_info):
    args = {
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': None
    }
    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict['params']['config']['env_info'] = env_info

    params = load_config(rlg_config_dict["params"])
    player = PpoPlayerContinuous(params)
    _restore(player, args)
    _override_sigma(player, args)

    return player

def reset_env(env=None):
    pass
def play_task(task_config):
    initialize(config_path="./cfg") #change together with code in isaacgymenvs.make
    cfgs = []
    tasks = []
    players = []

    cfg = compose(task_config)
    cfg = completeconfig(cfg)

    env = Robot()

    env_info = {'observation_space': Box(-float('inf'),float('inf'),[20]), 'action_space': Box(-1,1,[12]), 'agents': 1,
     'value_size': 1}

    cfg_dict = omegaconf_to_dict(cfg.task)
    cfg_dict['env']['numEnvs'] = 128
    cfgs.append(cfg_dict)
    player = load_player(cfg, env_info)
    #players[-1].max_steps = 1000
    player.print_stats = False
    player.max_steps = 200

    game_num = 100
    dons = []
    for n in range(game_num):

        reset_env(env)


        done = play(player, env, task_config)
        #if i==2:
            #print(env.nut_dist_to_target)
        print(done)
        dons.append((done))
    print("___________________________\nsuccess : ",end="")
    print(np.mean(dons))
if __name__ == "__main__":
    play_task("config_Pick.yaml")


