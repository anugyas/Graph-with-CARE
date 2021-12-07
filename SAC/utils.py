import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
#import gym
import os
from collections import deque
import random
import math

#import dmc2gym


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

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

def is_legal(x, y):
    return (x >= 0) & (x < GRID_DIM) & (y >= 0) & (y <= GRID_DIM)

class MTRL_ATT(nn.Module):
    """
    """
    def __init__(self, din):
        super(MTRL_ATT, self).__init__()
        self.fc1 = nn.Linear(din, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = F.sigmoid(self.fc3(y))
        return y

class MTRL_Encoder(nn.Module): # TODO: Need to make it a CNN for higher dim obs space like MetaWorld
    """
    """
    def __init__(self, din=32, hidden_dim=128):
        super(MTRL_Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)


    def forward(self, x):
        embedding = F.tanh(self.fc(x))
        return embedding

class MTRL_AttModel(nn.Module):
    """
    """
    def __init__(self, n_node, din, hidden_dim, dout):
        super(MTRL_AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, hidden_dim)
        self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        v = F.tanh(self.fcv(x))
        q = F.tanh(self.fcq(x))
        k = F.tanh(self.fck(x)).permute(0,2,1)
        att = F.softmax(torch.mul(torch.bmm(q,k), mask) - 9e15*(1 - mask),dim=2)
        # Note: Order of applying adj matrix is different than that in paper. Don't get confused!
        out = torch.bmm(att,v)
        return out

class MTRL_Q_Net(nn.Module):
    """
    """
    def __init__(self, hidden_dim, dout):
        super(MTRL_Q_Net, self).__init__()
        # NOTE: This is now modified to have both h vectors from both of the attention layers
        # concatenated - originally it was only getting the h vector of the last layer
        # so the input dim of the linear layer was hidden_dim
        self.fc = nn.Linear(hidden_dim*2, dout)

    def forward(self, x):
        q = F.relu(self.fc(x))
        return q

    
class MTRL_DGN(nn.Module):
    """
    """
    def __init__(self,
                 n_tasks,
                 num_inputs,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super(MTRL_DGN, self).__init__()

        self.encoder = MTRL_Encoder(num_inputs,hidden_dim)
        # TODO: Try both single encoder and mix of encoder settings
        # Will remain same for MTRL
        self.att_1 = MTRL_AttModel(n_tasks,hidden_dim,hidden_dim,hidden_dim)
        self.att_2 = MTRL_AttModel(n_tasks,hidden_dim,hidden_dim,hidden_dim)
        # self.q_net = MTRL_Q_Net(hidden_dim,num_actions)
        self.mlp = MLP(hidden_dim*2, hidden_dim, output_dim, hidden_depth, output_mod)
        # Q Net remains same for MTRL

    def forward(self, x, mask):
        h1 = self.encoder(x)
        h2 = self.att_1(h1, mask)
        h3 = self.att_2(h2, mask) 
        # TODO: try concatentation for MTRL
        
        h4 = torch.cat((h2,h3),dim=2)
        # q = self.q_net(h4)
        op = self.mlp(h4)
        # Note: No concatenation done. Output of last attention head used directly
        # Note: 2 attention heads used
        return q 

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

class GridWorldWithCare(object):
    
    def __init__(self, grid_dim, n_tasks):
        """
        Initialize the gridworld
        
        Params:
        n_tasks:
        """
        super(GridWorldWithCare, self).__init__()
        self.n_action = 4
        self.grid_dim = grid_dim
        self.n_tasks = n_tasks
        # TODO: maybe include food as part of task, reach dest with > 0 food or something
        self.tasks = [0]*self.n_tasks
        self.agent = [-1, -1]
        self.build_env()

        self.dones = np.zeros(self.n_tasks) # Array to indicate whether each task is done or not -- used to calculate rewards
        self.steps = 0
        self.len_obs = (self.n_tasks+1)*2

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
            x = np.random.randint(0, self.grid_dim)
            y = np.random.randint(0, self.grid_dim)
            self.tasks[i] = [x, y]
            print("TASK NUMBER ", i, " DEST: ", x, y)
        self.agent[0] = np.random.randint(0, self.grid_dim)
        self.agent[1] = np.random.randint(0, self.grid_dim)

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

        obs.append(x_agent/self.grid_dim)
        obs.append(y_agent/self.grid_dim)

        # 		for i in range(-1,2):
        # 			for j in range(-1,2):
        # 				obs.append(self.maze[x_agent+i][y_agent+j])

        for i in range(self.n_tasks):
            obs.append((self.tasks[i][0]-x_agent)/self.grid_dim)
            obs.append((self.tasks[i][1]-y_agent)/self.grid_dim)

        # TODO: 1. if we include maze state or not, and if we do, we would need to figure out
        # how to effectively send that along with task destinations
        
        #Idea: use distance between agent and task as obs
        
        return obs

    def get_adj(self): # TODO: Change this to use task description encoding. 
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
            x_task_i, y_task_i = self.tasks[i][0]-x_agent, self.tasks[i][1]-y_agent
            for j in range(self.n_tasks):
                x_task_j, y_task_j = self.tasks[j][0]-x_agent, self.tasks[j][1]-y_agent
                # Instead of having 1 or 0s, have their vectoral positions according to each other
                task_dist = math.sqrt((x_task_j - x_task_i)**2 + (y_task_j - y_task_i)**2)
                
        #                 print('x_task_i: {}, y_task_i: {}, x_task_j: {}, y_task_j: {}, task_dist: {}'.format(
        #                         x_task_i, y_task_i, x_task_j, y_task_j, task_dist
        #                 ))
                        
                # Set this distance / GRID_DIM
                adj[i,j] = 1 - float(task_dist)/self.grid_dim # Extract from 1 bc the closer the better
                adj[j,i] = 1 - float(task_dist)/self.grid_dim
                
        
                
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
        if action == 0: # Move up (decrease x by one)
            if is_legal(x_agent-1, y_agent):
                # Change the agent and the maze
                self.agent[0] -= 1

        elif action == 1: # Move down (increase x by one)
            if is_legal(x_agent+1, y_agent):
                # Change the agent and the maze
                self.agent[0] += 1

        elif action == 2: # Move left (decrease y by one)
            if is_legal(x_agent, y_agent-1):
                # Change the agent and the maze
                self.agent[1] -= 1

        elif action == 3: # Move right (increase y by one)
            if is_legal(x_agent, y_agent+1):
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
                total_reward += 1.0/float((math.sqrt((self.tasks[i][0]-new_agent_x)**2 + (self.tasks[i][1]-new_agent_y)**2)))
                

        # Only if all the tasks are done, then the episode is done
        all_tasks_done = not (0 in self.dones)



        return self.get_obs(), self.get_adj(), total_reward, all_tasks_done
