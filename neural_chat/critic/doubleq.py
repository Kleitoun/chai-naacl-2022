from typing import List
import torch
from torch import nn
from gym import Space
from neural_chat.utils import MLP
from neural_chat.transforms import OneHotFlatten
from neural_chat.logger import simpleloggable

@simpleloggable
class DoubleQCritic(nn.Module):

    def __init__(self, obs_spec: Space, action_spec: Space, hidden_dim: List[int])
        nn.Module.__init__(self)

    ## flatten observation specifications tensor
    self.obs_spec = OneHotFlatten(self.obs_spec)
    ## flatten actor specifications tensor
    self.action_spec = OneHotFlatten(self.action_spec)

    ## initiate two Q learning MLP using the dimensions of the obs/act
    ## first dim is sum of both dims, second is hidden dim, and output dim is q (q score)
    self.q1 = MLP(self.obs_spec.after_dim + self.action_spec.after_dim,hidden_dim,1)
    self.q2 = MLP(self.obs_spec.after_dim + self.action_spec.after_dim,hidden_dim,1)

    ## produce a doule q algorithm that concatenates both observation/act (with dim=-1) and then passes both through q1/q2 and then computes both values
    def double_q(self,obs,action):
        obs_action = torch.cat(self.obs_spec(obs),self.action_spec(action),dim=-1)
        q1,q2=self.q1(obs_action),self.q2(obs_action)
        self.log("qval_1",q1)
        self.log("qval_2",q2)
        return q1,q2
    ## function that returns the lowest q value
    def forward(self,obs,action):
        return torch.min(*self.double_q(obs,action))
