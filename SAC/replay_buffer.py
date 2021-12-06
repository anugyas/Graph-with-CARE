import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max


class ReplayBufferGCare(object):
    """
    Replay buffer for storing the agent's experiences
    """

    def __init__(self, buffer_size, obs_space, n_action, n_tasks):
        """
        Initialize the replay buffer

        Params:
        buffer_size:
        obs_space:
        n_action:
        n_tasks:
        """
        self.buffer_size = buffer_size
        self.n_tasks = n_tasks
        self.pointer = 0
        self.len = 0
        self.actions = np.zeros((self.buffer_size, 1), dtype=np.int32)
        self.rewards = np.zeros((self.buffer_size, 1))
        self.dones = np.zeros((self.buffer_size, 1))
        self.obs = np.zeros((self.buffer_size, n_tasks, obs_space))
        self.next_obs = np.zeros((self.buffer_size, n_tasks, obs_space))
        self.matrix = np.zeros((self.buffer_size, self.n_tasks, self.n_tasks))
        self.next_matrix = np.zeros((self.buffer_size, self.n_tasks, self.n_tasks))

    def getBatch(self, batch_size):
        """
        Sample a batch of random entries from the replay buffer

        Params:
        batch_size:

        Returns:
        obs:
        action:
        reward
        next_obs:
        matrix:
        next_matrix:
        done:
        """
        index = np.random.choice(self.len, batch_size, replace=False)
        return self.obs[index], self.actions[index], self.rewards[index], self.next_obs[index], self.matrix[index], \
               self.next_matrix[index], self.dones[index]

    def add(self, obs, action, reward, next_obs, matrix, next_matrix, done):
        """
        Add to the replay buffer

        Params:
        obs:
        action:
        reward:
        next_obs:
        matrix:
        next_matrix:
        done:
        """
        self.obs[self.pointer] = obs
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_obs[self.pointer] = next_obs
        self.matrix[self.pointer] = matrix
        self.next_matrix[self.pointer] = next_matrix
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.len = min(self.len + 1, self.buffer_size)