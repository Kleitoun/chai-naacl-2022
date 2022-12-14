from typing import List
from torch import nn
from neural_chat.logger import Loggable

class MLP(nn.Module, Loggable):
    def __init__(self ,input_dim: int, hidden_dim: List[int],output_dim: int)
        nn.Module.__init__(self)
        Loggable.__init__(self)

    sizes = [input_dim] + hidden_dim + [output_dim]
    sizes_one_off = sizes[1:]
    activation = nn.ReLU(inplace=True)
    mods = sum([[nn.Linear(i ,j), activation] for i, j in zip(sizes,sizes_one_off)],[],)
    mods = mods[:-1]

    self.mlp = nn.Sequential(*mods)
    self.hyperparams = {"input_dim" = input_dim,
                        "output_dim" = output_dim,
                        "hidden_dim" = hidden_dim}

    def forward(self, x):
        return self.mlp.forward(x)

    def get_hyperparams(self):
        return self.hyperparams

    def get_local_epoch(self):
        return self.mlp.state_dict()
