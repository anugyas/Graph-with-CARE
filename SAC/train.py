#!/usr/bin/env python3
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

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import dmc2gym
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
from torch.utils import tensorboard as tb

GRID_DIM = 100  # TODO: Tune this
NUM_TASKS = 5  # TODO: Tune this
ADJ_THRESHOLD = GRID_DIM / 4  # TODO: Tune this
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


def is_legal(x, y):
    return (x >= 0) & (x < GRID_DIM) & (y >= 0) & (y <= GRID_DIM)


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


class GridWorldWithCare(object):

    def __init__(self, n_tasks):
        """
        Initialize the gridworld

        Params:
        n_tasks:
        """
        super(GridWorldWithCare, self).__init__()
        self.n_action = 4
        self.n_tasks = n_tasks
        # TODO: maybe include food as part of task, reach dest with > 0 food or something
        self.tasks = [0] * self.n_tasks
        self.agent = [-1, -1]
        self.build_env()

        self.dones = np.zeros(
            self.n_tasks)  # Array to indicate whether each task is done or not -- used to calculate rewards
        self.steps = 0
        self.len_obs = (self.n_tasks + 1) * 2

    def reset(self):
        """
        Reset the gridworld

        Returns:
        obs:
        adj:
        """

        self.build_env()
        self.dones = np.zeros(self.n_tasks)
        self.steps = 0
        return self.get_obs(), self.get_adj()

    def build_env(self):
        """
        Build the gridworld
        """
        for i in range(self.n_tasks):
            x = np.random.randint(0, GRID_DIM)
            y = np.random.randint(0, GRID_DIM)
            self.tasks[i] = [x, y]
            print("TASK NUMBER ", i, " DEST: ", x, y)
        self.agent[0] = np.random.randint(0, GRID_DIM)
        self.agent[1] = np.random.randint(0, GRID_DIM)

    def get_obs(self):
        """
        Get observations

        Returns:
        obs:
        """
        # TODO: change this for MTRL
        obs = []

        x_agent = self.agent[0]
        y_agent = self.agent[1]

        obs.append(x_agent / GRID_DIM)
        obs.append(y_agent / GRID_DIM)

        # 		for i in range(-1,2):
        # 			for j in range(-1,2):
        # 				obs.append(self.maze[x_agent+i][y_agent+j])

        for i in range(self.n_tasks):
            obs.append((self.tasks[i][0] - x_agent) / GRID_DIM)
            obs.append((self.tasks[i][1] - y_agent) / GRID_DIM)

        # TODO: 1. if we include maze state or not, and if we do, we would need to figure out
        # how to effectively send that along with task destinations

        # Idea: use distance between agent and task as obs

        return obs

    def get_adj(self):  # TODO: Change this to use task description encoding.
        # In this case task description is the location of the destination.
        """
        Get adjacency matrix

        Returns:
        adj:
        """
        adj = np.zeros((self.n_tasks, self.n_tasks))

        # Calculate adjacency regarding to the distances of the tasks respect to the agent
        x_agent, y_agent = self.agent[0], self.agent[1]

        # HARD ATTENTION
        # Traverse through the tasks and calculate the Euclidean distance between them and the agent
        #         for i in range(self.n_tasks):
        #             x_task_i, y_task_i = self.tasks[i][0] - x_agent, self.tasks[i][1] - y_agent
        #             for j in range(self.n_tasks):
        #                 x_task_j, y_task_j = self.tasks[j][0] - x_agent, self.tasks[j][1] - y_agent
        #                 task_dist = math.sqrt((x_task_j - x_task_i)**2 + (y_task_i - y_task_j)**2)
        #                 if task_dist <= ADJ_THRESHOLD:
        #                     adj[i,j] = 1
        #                     adj[j,i] = 1

        # SOFT ATTENTION
        #         adj = np.ones((self.n_tasks, self.n_tasks)) # NOTE:
        for i in range(self.n_tasks):
            x_task_i, y_task_i = self.tasks[i][0] - x_agent, self.tasks[i][1] - y_agent
            for j in range(self.n_tasks):
                x_task_j, y_task_j = self.tasks[j][0] - x_agent, self.tasks[j][1] - y_agent
                # Instead of having 1 or 0s, have their vectoral positions according to each other
                task_dist = math.sqrt((x_task_j - x_task_i) ** 2 + (y_task_j - y_task_i) ** 2)

                #                 print('x_task_i: {}, y_task_i: {}, x_task_j: {}, y_task_j: {}, task_dist: {}'.format(
                #                         x_task_i, y_task_i, x_task_j, y_task_j, task_dist
                #                 ))

                # Set this distance / GRID_DIM
                adj[i, j] = 1 - float(task_dist) / GRID_DIM  # Extract from 1 bc the closer the better
                adj[j, i] = 1 - float(task_dist) / GRID_DIM

        #         print("ADJACENCY: {}".format(adj))

        #         print('x_agent: {}, y_agent: {}'.format(x_agent, y_agent))

        return adj

    def step(self, action):
        """
        Take one step in the gridworld according to the given actions

        Params:
        action:

        Returns:
        obs:
        adj:
        reward:
        all_tasks_done:
        """

        # There are 4 different actions for the agent
        # If there is any place to go in the maze then the agent will go
        # 0: Move up, 1: Move down, 2: Move left, 3: Move right

        self.steps += 1
        x_agent, y_agent = self.agent[0], self.agent[1]
        #         print("AGENT LOCATION: ", agent_x, agent_y)
        #         print("ACTION: ", action)
        if action == 0:  # Move up (decrease x by one)
            if is_legal(x_agent - 1, y_agent):
                # Change the agent and the maze
                self.agent[0] -= 1

        elif action == 1:  # Move down (increase x by one)
            if is_legal(x_agent + 1, y_agent):
                # Change the agent and the maze
                self.agent[0] += 1

        elif action == 2:  # Move left (decrease y by one)
            if is_legal(x_agent, y_agent - 1):
                # Change the agent and the maze
                self.agent[1] -= 1

        elif action == 3:  # Move right (increase y by one)
            if is_legal(x_agent, y_agent + 1):
                # Change the agent and the maze
                self.agent[1] += 1

        # Calculate the rewards for each task
        rewards = [0] * self.n_tasks
        total_reward = 0

        # Check if you reached to any destinations here
        new_agent_x, new_agent_y = self.agent[0], self.agent[1]
        for i in range(self.n_tasks):
            if self.tasks[i][0] == new_agent_x and self.tasks[i][1] == new_agent_y:
                if self.dones[i] == 0:
                    self.dones[i] = 1
                    rewards[i] = 1
                    total_reward += 1
                    print("Task ", i, " completed at step ", self.steps)
            else:
                total_reward += 1.0 / float(
                    (math.sqrt((self.tasks[i][0] - new_agent_x) ** 2 + (self.tasks[i][1] - new_agent_y) ** 2)))

        # Only if all the tasks are done, then the episode is done
        all_tasks_done = not (0 in self.dones)

        return self.get_obs(), self.get_adj(), total_reward, all_tasks_done


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # self.env = utils.make_env(cfg)
        self.env = GridWorldWithCare(NUM_TASKS)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
