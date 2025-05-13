import os
from math import inf
from time import time

import numpy as np
import torch
from torch import Tensor, logical_and, nn, tensor
from torch.nn import functional as F

from game.engine.hand_evaluator import HandEvaluator

from .memory import History
from .my_types import *
from .utils import (
    ValidActionInfo,
    get_idx_from_prob,
    get_stacks,
    one_hot,
    one_hot_cards,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """
    General network structure for DQN
    """

    n_state: int

    def __init__(self, n_observations: int, n_actions: int):
        self.n_state = n_observations

        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: Tensor) -> torch.Tensor:
        # x = x[: self.n_state]
        # x=  x.take()
        # torch.take_along_dim(x,  torch.arange(self.n_state), -1)
        x = x[..., : self.n_state]

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def __call__(self, x) -> torch.Tensor:
        return super(DQN, self).__call__(x)

    @classmethod
    def from_path(cls, path: str):
        state_dict = torch.load(path)
        n_state: int = state_dict["layer1.weight"].shape[1]
        obj = cls(n_state, n_actions())
        obj.load_state_dict(state_dict)
        return obj

    @classmethod
    def from_name(cls, name: str):
        return cls.from_path(f"{os.path.dirname(__file__)}/model/{name}.pt")


# ----------------------------- Encoding/Decoding ---------------------------- #


def n_observations() -> int:
    return 67


def n_actions() -> int:
    return RAISE_IDX + len(RAISE_AMOUNTS)


WIN_RATE_SIMULATIONS = 100
MAX_ROUND = 20


def get_state_feature(history: History) -> Tensor:
    """
    Get feature (i.e. state) vector from env

    State: [
        round_count,
        street (one hot),
        player stacks,

        hand strength,

        win rate,
        commnuity cards (one hot),
        pot amount,

        bluff_count,

        (todo) action history,
    ]
    """
    # ! Newly added state must be at last

    st = time()

    round_state = history.round_state
    hole = history.hole_cards
    community = history.community_cards
    hand = HandEvaluator.eval_hand(hole, community)
    hand_rank_info: HandRankInfo = HandEvaluator.gen_hand_rank_info(hole, community)
    win_rate = history.win_rate

    t1 = time() - st

    feat = torch.tensor(
        [
            round_state["round_count"] / MAX_ROUND,
            *one_hot(STREET_TUPLE.index(round_state["street"]), len(STREET_TUPLE)),
            *map(lambda x: x / 2000, get_stacks(round_state, history.uuid)),
            #
            hand / (1 << 16),
            hand_rank_info["hand"]["high"] / 14,
            hand_rank_info["hand"]["low"] / 14,
            hand_rank_info["hole"]["high"] / 14,
            hand_rank_info["hole"]["low"] / 14,
            #
            win_rate,
            *one_hot_cards(community),
            round_state["pot"]["main"]["amount"] / 2000,
            #
            history.bluff_count,
        ],
        device=device,
    )
    # print(t1 / (time() - st))

    assert len(feat) == n_observations(), len(feat)

    return feat


# discrete amounts for raise actions
RAISE_AMOUNTS = [
    10,
    25,
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
]
RAISE_IDX = 3


Stretagy = Literal["greedy", "epsilon", "boltzmann"]


def decode_action(
    action_q: Tensor,
    actions_info: ValidActionInfo,
    stretagy: Stretagy,
    epsilon=0.1,
) -> DispatchAction:
    """
    Action Q: [
        fold,
        call,
        all in,
        raise 10,
        raise 50,
        ...
    ]
    """

    assert action_q.shape == (n_actions(),)
    action_q = mask_valid_actions(action_q, actions_info)

    r = torch.rand(1)
    if stretagy == "greedy" or (stretagy == "epsilon" and r > epsilon):
        # exploit
        idx = torch.argmax(action_q)
    elif stretagy == "epsilon" and r <= epsilon:
        # explore
        idx = torch.randint(0, n_actions(), (1,))
        while action_q[idx] == -inf:
            idx = torch.randint(0, n_actions(), (1,))
    elif stretagy == "boltzmann":
        prob = F.softmax(action_q, dim=0)
        idx = get_idx_from_prob(prob)
    else:
        raise ValueError(f"Invalid strategy: {stretagy}")

    return idx_to_action(int(idx), actions_info)


def action_to_idx(action: DispatchAction, actions_info: ValidActionInfo):
    if action[0] == "fold":
        return 0
    elif action[0] == "call":
        return 1
    elif action[0] == "raise":
        if action[1] == actions_info.raise_max:
            return 2
        return RAISE_IDX + int(np.searchsorted(RAISE_AMOUNTS, action[1]))
    else:
        raise ValueError(f"Invalid action: {action}")


def idx_to_action(idx: int, actions_info: ValidActionInfo) -> DispatchAction:
    if idx == 0:
        return "fold", 0
    elif idx == 1:
        return "call", actions_info.call_amount
    elif idx == 2:
        return "raise", actions_info.raise_max

    amount = RAISE_AMOUNTS[idx - RAISE_IDX]
    # try:
    assert actions_info.raise_min <= amount <= actions_info.raise_max
    # except:
    #     print(( actions_info))
    #     print(amount)
    #     exit()
    return "raise", amount


def mask_valid_actions(
    action_q: Tensor,
    actions_info: ValidActionInfo,
    inplace=True,
) -> Tensor:
    if not inplace:
        action_q = action_q.clone()

    raise_amounts = tensor(RAISE_AMOUNTS)
    mask = logical_and(
        raise_amounts >= actions_info.raise_min,
        raise_amounts <= actions_info.raise_max,
    )

    raise_actions = action_q[RAISE_IDX:]
    assert len(raise_actions) == len(RAISE_AMOUNTS)
    raise_actions[~mask] = -inf
    return action_q


class DQN_Dueling(DQN):

    n_actions: int

    def __init__(self, n_observations: int, n_actions: int):
        super(DQN_Dueling, self).__init__(n_observations, n_actions)
        self.advantage = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)

        self.n_actions = n_actions

    def forward(self, x: Tensor) -> torch.Tensor:
        x = x[..., : self.n_state]

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        A = self.advantage.forward(x)
        V = self.value.forward(x).expand(A.shape)
        A = A / A.sum() + V
        # assert A.shape == (self.n_actions,), A.shape
        return A
