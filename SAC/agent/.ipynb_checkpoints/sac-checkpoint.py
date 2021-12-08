import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agent import Agent
from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor
import utils

import hydra


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, n_tasks):
        super().__init__()

        self.n_tasks = n_tasks
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = DoubleQCritic(n_tasks=self.n_tasks,
                                    obs_dim=self.obs_dim,
                                    action_dim=self.action_dim,
                                    hidden_dim=64,
                                    hidden_depth=2)

        self.critic = self.critic.to(self.device)

        self.critic_target = DoubleQCritic(n_tasks=self.n_tasks,
                                           obs_dim=self.obs_dim,
                                           action_dim=self.action_dim,
                                           hidden_dim=64,
                                           hidden_depth=2)
        
        self.critic_target = self.critic_target.to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DeterministicActor(n_tasks=self.n_tasks,
                                        obs_dim=self.obs_dim,
                                        action_dim=self.action_dim,
                                        hidden_dim=64,
                                        hidden_depth=2)

        self.actor = self.actor.to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, adj, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device)
#         dist = self.actor(obs, adj)
#         action = dist.sample() if sample else dist.mean
#         action = action.clamp(*self.action_range) # Shape: (1,2,4)
#         action = action[:,0,:] # # Shape: (1,4)
        act_probs = self.actor(obs, adj) # Shape: (BATCH_SIZE, N_TASKS, 4)
        act_probs = act_probs[:,0,:] # Shape (BATCH_SIZE,4)
        assert action.ndim == 2 and action.shape[0] == 1 # BATCH_SIZE needs to be 1 in act()
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, adj, next_adj, done, logger,
                      step):
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        next_adj = torch.FloatTensor(next_adj).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        dist = self.actor(next_obs, next_adj)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action, next_adj)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        reward = np.resize(reward, (self.batch_size, self.n_tasks, 1))
        reward = torch.FloatTensor(reward).to(self.device)
        done.resize_(reward.shape[0], self.n_tasks, 1)
        target_Q = reward + ((1-done) * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        obs = torch.FloatTensor(obs).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device)
        action = np.resize(action, (self.batch_size, self.n_tasks, self.action_dim))
        action = torch.FloatTensor(action).to(self.device)
        current_Q1, current_Q2 = self.critic(obs, action, adj)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, adj, logger, step):
        obs = torch.FloatTensor(obs).to(self.device)
        adj = torch.FloatTensor(adj).to(self.device)
        dist = self.actor(obs, adj)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action, adj)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, adj, next_adj, done = replay_buffer.get_batch(
            self.batch_size)
        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, adj, next_adj, done, logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, adj, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
