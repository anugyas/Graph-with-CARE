#!/usr/bin/env python3
from re import M
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

# from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBufferGCare
import utils

# import dmc2gym
import hydra

import numpy as np
import math, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import copy
import os, sys
import yaml
from torch.utils import tensorboard as tb
from agent.sac import SACAgent

# hidden_dim = 64
# max_step = 5000 #originally 500
# GAMMA = 0.99
# n_episode = 1000 #originally 800
# i_episode = 0
buffer_size = 100000 #change back to 65000
# batch_size = 64 #change back to 64
# n_epoch = 100 #orginally 25
# epsilon = 0.7 #originally 0.9
# score = 0
# tau = 0.98

GRID_DIM = 10  # TODO: Tune this
NUM_TASKS = 2  # TODO: Tune this
EVAL_STEPS = 10000
# ADJ_THRESHOLD = GRID_DIM / 4  # TODO: Tune this
# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
#                                                                                                                 **kwargs)

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        # print("CFGAAAAAAAAA:", cfg)
        # print('cfg: {}'.format(cfg))
        self.logger = Logger(self.work_dir,
                             save_tb=self.cfg['log_save_tb'],
                             log_frequency=self.cfg['log_frequency'],
                             agent=self.cfg['defaults'][0]['agent'])

        utils.set_seed_everywhere(self.cfg['seed'])
        self.device = torch.device(self.cfg['device'])
        # self.env = utils.make_env(cfg)
        self.env = utils.GridWorldWithCare(GRID_DIM, NUM_TASKS)

        # cfg.agent.params.obs_dim = self.env.len_obs
        # cfg.agent.params.action_dim = self.env.n_action
        # cfg.agent.params.action_range = [0,3]
        # self.agent = hydra.utils.instantiate(self.cfg.agent)

        self.agent = SACAgent(obs_dim=self.env.len_obs,
                              action_dim=self.env.n_action,
                              action_range=[0,3],
                              device='cuda',
                              critic_cfg="TODO",
                              actor_cfg="TODO",
                              discount=0.99,
                              init_temperature=0.1,
                              alpha_lr=1e-4,
                              alpha_betas=[0.9, 0.999],
                              actor_lr=1e-4,
                              actor_betas=[0.9, 0.999],
                              actor_update_frequency=1,
                              critic_lr=1e-4,
                              critic_betas=[0.9, 0.999],
                              critic_tau=0.005,
                              critic_target_update_frequency=2,
                              batch_size=1024,
                              learnable_temperature=True,
                              n_tasks=NUM_TASKS)

        self.replay_buffer = ReplayBufferGCare(buffer_size, self.env.len_obs, self.env.n_action, self.env.n_tasks)

        # self.video_recorder = VideoRecorder(
        #     self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg["num_eval_episodes"]):
            obs, adj = self.env.reset()
#             self.agent.reset()
#             self.video_recorder.init(enabled=(episode == 0))
#             done = False
            eval_step = 0
            episode_reward = 0
            while eval_step < EVAL_STEPS:
                obs = np.resize(obs, (self.agent.n_tasks, self.env.len_obs))
                obs = np.array([obs])
                adj = np.array([adj])
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, adj, sample = False)
                obs, adj, reward, _ = self.env.step(action)
#                 self.video_recorder.record(self.env)
                episode_reward += reward
            average_episode_reward += episode_reward
#             self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg["num_eval_episodes"]
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < float(self.cfg["num_train_steps"]):
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > float(self.cfg["num_seed_steps"])))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg["eval_frequency"] == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs, adj = self.env.reset()
                
#                 self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg["num_seed_steps"]:
                action = np.random.rand(self.env.n_action)
            else:
                with utils.eval_mode(self.agent):
                    obs = np.resize(obs, (self.agent.n_tasks, self.env.len_obs))
                    obs = np.array([obs])
                    adj = np.array([adj])
                    action = self.agent.act(obs, adj, sample=True)

            # run training update
            if self.step >= self.cfg["num_seed_steps"]:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            
            # Take argmax as the action to apply
            action_to_apply = np.argmax(action, axis=0)
            next_obs, next_adj, reward, done = self.env.step(action_to_apply)

            # allow infinite bootstrap
            done = float(done)
#             done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done #?
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, adj, next_adj, done)

            obs = next_obs
            adj = next_adj
            episode_step += 1
            self.step += 1


# @hydra.main(config_path="/scratch/ig2283/Graph-with-CARE/SAC/config", config_name="train")
def main():
    with open("./config/train.yaml") as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
        # print(cfg)
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
