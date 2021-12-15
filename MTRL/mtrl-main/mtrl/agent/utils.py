# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtrl.agent.components import moe_layer
from mtrl.utils.types import ModelType, TensorType


class eval_mode(object):
    def __init__(self, *models):
        """Put the agent in the eval mode"""
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


def soft_update_params(net: ModelType, target_net: ModelType, tau: float) -> None:
    """Perform soft udpate on the net using target net.

    Args:
        net ([ModelType]): model to update.
        target_net (ModelType): model to update with.
        tau (float): control the extent of update.
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed: int) -> None:
    """Set seed for reproducibility.

    Args:
        seed (int): seed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def preprocess_obs(obs: TensorType, bits=5) -> TensorType:
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def weight_init_linear(m: ModelType):
    assert isinstance(m.weight, TensorType)
    nn.init.xavier_uniform_(m.weight)
    assert isinstance(m.bias, TensorType)
    nn.init.zeros_(m.bias)


def weight_init_conv(m: ModelType):
    # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    assert isinstance(m.weight, TensorType)
    assert m.weight.size(2) == m.weight.size(3)
    m.weight.data.fill_(0.0)
    if hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)  # type: ignore[operator]
    mid = m.weight.size(2) // 2
    gain = nn.init.calculate_gain("relu")
    assert isinstance(m.weight, TensorType)
    nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def weight_init_moe_layer(m: ModelType):
    assert isinstance(m.weight, TensorType)
    for i in range(m.weight.shape[0]):
        nn.init.xavier_uniform_(m.weight[i])
    assert isinstance(m.bias, TensorType)
    nn.init.zeros_(m.bias)


def weight_init(m: ModelType):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        weight_init_linear(m)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        weight_init_conv(m)
    elif isinstance(m, moe_layer.Linear):
        weight_init_moe_layer(m)


def _get_list_of_layers(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
) -> List[nn.Module]:
    """Utility function to get a list of layers. This assumes all the hidden
    layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module]
    if num_layers == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    return mods


def build_mlp_as_module_list(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
) -> ModelType:
    """Utility function to build a module list of layers. This assumes all
    the hidden layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module] = _get_list_of_layers(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )
    sequential_layers = []
    new_layer = []
    for index, current_layer in enumerate(mods):
        if index % 2 == 0:
            new_layer = [current_layer]
        else:
            new_layer.append(current_layer)
            sequential_layers.append(nn.Sequential(*new_layer))
    sequential_layers.append(nn.Sequential(*new_layer))
    return nn.ModuleList(sequential_layers)


def build_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
) -> ModelType:
    """Utility function to build a mlp model. This assumes all the hidden
    layers are using the same dimensionality.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): dimension of the hidden layers.
        output_dim (int): dimension of the output layer.
        num_layers (int): number of layers in the mlp.

    Returns:
        ModelType: [description]
    """
    mods: List[nn.Module] = _get_list_of_layers(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )
    return nn.Sequential(*mods)



# Add DGN models
class MTRL_AttModel(nn.Module):
    """
    """
    def __init__(self, din, hidden_dim, dout):
        super(MTRL_AttModel, self).__init__()
        self.fcv = nn.Linear(din, hidden_dim).to('cuda')
        self.fck = nn.Linear(din, hidden_dim).to('cuda')
        self.fcq = nn.Linear(din, hidden_dim).to('cuda')

    def forward(self, x, mask):
        v = torch.tanh(self.fcv(x))
        q = torch.tanh(self.fcq(x))
        k = torch.transpose(torch.tanh(self.fck(x)),-1,-2)

        att = F.softmax(torch.mul(torch.matmul(q,k), mask) - 9e15*(1 - mask), dim=-1)
        out = torch.matmul(att,v)
        return out

class MTRL_DGN(nn.Module):
    """
    """
    def __init__(self,
                 input_dim, # hidden_dim
                 hidden_dim, # hidden_dim
                 output_dim, # output_dim
                 attention_num_layers,
                 mlp_num_layers):
        super(MTRL_DGN, self).__init__()

        # self.encoder = MTRL_Encoder(num_inputs,hidden_dim)
        # TODO: Try both single encoder and mix of encoder settings
        # Will remain same for MTRL
        self.attention_num_layers = attention_num_layers
        self.attention_layers: List[nn.Module] = []
        for _ in range(self.attention_num_layers):
            self.attention_layers.append(MTRL_AttModel(input_dim, hidden_dim, hidden_dim))

        # self.att_1 = MTRL_AttModel(hidden_dim,hidden_dim,hidden_dim)
        # self.att_2 = MTRL_AttModel(hidden_dim,hidden_dim,hidden_dim)
        
        # The input to the final mlp is the concatenation of obs embeddings, and two other attention heads 
        self.mlp = build_mlp((self.attention_num_layers+1)*hidden_dim, hidden_dim, output_dim, mlp_num_layers)
        # Q Net remains same for MTRL

    def forward(self, x, mask): # Shape(x): (BATCH_SIZE, HIDDEN_DIM(OBS_EMB_SHAPE)) -> (BATCH_SIZE, NUM_TASKS, HIDDEN_DIM(OBS_EMB_SHAPE))
        # h1 = self.encoder(x)
        # print('MTRL_DGN.FORWARD: x.shape: {}'.format(x.shape))
        # print('MTRL_DGN.FORWARD: mask.shape: {}'.format(mask.shape))

        num_tasks = mask.shape[-1] # get the number of tasks
        x = torch.unsqueeze(x, 1) # add a new dimension for tasks
        x = x.expand(-1, num_tasks, -1) # expand the second dimension with num_tasks

        # print('MTRL_DGN.FORWARD INPUT EXPANDED: x.shape: {}'.format(x.shape))

        attention_heads = []
        attention_head = x
        for i in range(self.attention_num_layers):
            attention_head = self.attention_layers[i](attention_head, mask)
            attention_heads.append(attention_head)

        attention_heads = [x] + attention_heads
        # attention_heads = torch.Tensor(attention_heads)
        # print('attention_heads: {}'.format(attention_heads))
        output = torch.cat(attention_heads, dim=-1)
        # print('cat.shape: {}'.format(output.shape))
        output = self.mlp(torch.cat(attention_heads, dim=-1))

        # h1 = self.att_1(x, mask)
        # h2 = self.att_2(h1, mask) 
        
        # output = torch.cat((x,h1,h2), dim=-1)
        # op = self.mlp(h4)

        # NOTE: Since we expanded the values we can just get the first value
        # TODO: MAKE SURE THAT THIS IS AN OKAY APPROACH!!!! DISCUSS WITH ANUGYA!!!!
        return output[:,0,:] 
    
    def __iter__(self):
        return iter(self._modules.values())
    
    def __len__(self) -> int:
        return len(self._modules)

class GCModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, att_num_layers,
                 mlp_num_layers, trunk_num_layers):

        super(GCModel, self).__init__()
        self.trunk = build_mlp(  # type: ignore[assignment]
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=trunk_num_layers,
        )

        # print('trunk.input_dim: {}, trunk.hidden_dim: {}, trunk.output_dim: {}, trunk.num_layers: {}'.format(
        #     input_dim, hidden_dim, hidden_dim, trunk_num_layers
        # ))

        self.attention = MTRL_DGN(
            input_dim=hidden_dim, # hidden_dim
            hidden_dim=hidden_dim, # hidden_dim
            output_dim=output_dim, # output_dim
            attention_num_layers=att_num_layers,
            mlp_num_layers=mlp_num_layers
        )

    def forward(self, x, mask):
        # print('x.shape: {}'.format(x.shape))
        embedding = F.relu(self.trunk(x))
        # print('embedding_after_trunk.shape: {}'.format(embedding.shape))
        output = self.attention(embedding, mask)
        # print('output_after_attention.shape: {}'.format(output.shape))
        return output
