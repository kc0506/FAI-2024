from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from agents.call_player import CallPlayer

# from tqdm.autonotebook import tqdm
from agents.random_player import RandomPlayer
from baseline0 import setup_ai as baseline0_ai
from benchmark import benchmark, get_ai
from dqn import DQN, DQN_Dueling, n_actions, n_observations
from dqn_lstm import DQN_LSTM
from emulator import calc_mean_stacks, start_game
from equity_player import EquityPlayer
from game.players import BasePokerPlayer
from heuristic import MyHeuristicPlayer
from memory import MemoryName, SavePokerPlayer
from dqn_player import MyDQNPlayer
from train import (
    convert_multi_steps,
    convert_sequence,
    convert_single_step,
    train,
    train_model,
)
from utils import get_stacks


def post_train_05():

    n=100

    main_net = DQN.from_name("heuristic_0-5_single")
    train_model(
        "heuristic_0-5_5-7_single",
        [
            "heuristic_baseline5",
            "heuristic_baseline6",
            "heuristic_baseline7",
            "dqn_baseline0",
            "dqn_baseline7",
        ],
        get_ai(7),
        convert_single_step,
        num_episodes=n,
        main_net=main_net,
        out_memory_name="dqn_baseline7",
    )

    train_model(
        "heuristic_0-5_5-7_single",
        [
            "heuristic_baseline5",
            "heuristic_baseline6",
            "heuristic_baseline7",
            "dqn_baseline0",
            "dqn_baseline5",
        ],
        get_ai(5),
        convert_single_step,
        num_episodes=n,
        main_net=main_net,
        out_memory_name="dqn_baseline5",
    )

def train_double():
    train_model(
        "dqn_double",
        [
            "heuristic_baseline0",
            "heuristic_baseline1",
            "heuristic_baseline2",
            "heuristic_baseline3",
            "heuristic_baseline4",
            "heuristic_baseline5",
            "heuristic_baseline6",
            "heuristic_baseline7",
            "dqn_baseline0",
            "dqn_baseline5",
            "dqn_double_baseline5",
        ],
        get_ai(5),
        convert_single_step,
        num_episodes=100,
        out_memory_name="dqn_baseline5",
        is_double=True
    )



def train_d3qn():
    train_model(
        "d3qn",
        [
            "heuristic_baseline0",
            "heuristic_baseline1",
            "heuristic_baseline2",
            "heuristic_baseline3",
            "heuristic_baseline4",
            "heuristic_baseline5",
            "heuristic_baseline6",
            "heuristic_baseline7",
            "dqn_baseline0",
            # "dqn_baseline5",
            "d3qn_baseline5",
        ],
        get_ai(5),
        convert_single_step,
        num_episodes=200,
        out_memory_name="d3qn_baseline5",
        is_double=True,
        DQN_cls=DQN_Dueling,
    )


def train_dueling():
    train_model(
        "dqn_dueling",
        [
            "heuristic_baseline0",
            "heuristic_baseline1",
            "heuristic_baseline2",
            "heuristic_baseline3",
            "heuristic_baseline4",
            "heuristic_baseline5",
            "heuristic_baseline6",
            "heuristic_baseline7",
            "dqn_baseline0",
            # "dqn_baseline5",
            # "d3qn_baseline5",
            "dqn_dueling_baseline5",
        ],
        get_ai(5),
        convert_single_step,
        num_episodes=200,
        out_memory_name="dqn_dueling_baseline5",
        is_double=False,
        DQN_cls=DQN_Dueling,
    )



def train_lstm():
    stacks, losses = train_model(
        "dqn_lstm",
        [
            "heuristic_baseline0",
            "heuristic_baseline1",
            "heuristic_baseline2",
            "heuristic_baseline3",
        ],
        get_ai(0),
        convert_sequence,
        num_episodes=100,
        DQN_cls=DQN_LSTM,
    )


# train_dueling()
train_d3qn()
# train_double()
# post_train_05()
