from typing import OrderedDict, Tuple 

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .dqn import DQN, n_actions
from .my_types import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------ RNN ----------------------------------- #


class DQN_LSTM(DQN):

    n_state: int
    prev_hidden: Tuple[Tensor, Tensor]

    def __init__(self, n_observations: int, n_actions: int):
        super(DQN_LSTM, self).__init__(n_observations, n_actions)
        self.layer1 = nn.Linear(128, 128)
        self.rnn = nn.LSTM(n_observations, 128, batch_first=False)

        # self.layer3 = nn.Linear(128, 128)
        # self.rnn = nn.LSTM(128, n_actions, batch_first=False)

        self.prev_hidden = torch.zeros(1, 128), torch.zeros(1, 128)

    def forward(self, x: Tensor) -> torch.Tensor:
        # ! x is a sequence of batch
        is_batched = x.dim() == 3

        x = x[..., : self.n_state]
        if not is_batched:
            x, hidden = self.rnn.forward(x.unsqueeze(0), self.prev_hidden)
            self.prev_hidden = hidden
        else:
            x, _ = self.rnn.forward(x)

        if not is_batched:
            x = x[-1, :]
        else:
            x = x[:, -1, :]

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def __call__(self, x) -> torch.Tensor:
        return super(DQN_LSTM, self).__call__(x)

    @classmethod
    def from_path(cls, path: str):
        state_dict:OrderedDict = torch.load(path)
        # for k,v in state_dict.items():
        #     if k.startswith('rnn'):
        #         print(k, v)

        # n_state: int = state_dict["rnn.weight"].shape[-1]
        n_state = state_dict["rnn.weight_ih_l0"].shape[-1]
        obj = cls(n_state, n_actions())
        obj.load_state_dict(state_dict)
        return obj
