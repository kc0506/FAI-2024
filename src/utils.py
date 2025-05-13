import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Set, Union

import numpy as np
from torch import Tensor

from game.engine.card import Card
from .lib import estimate_hole_card_win_rate
from .my_types import STREET_TUPLE, Action, HandRankInfo, RoundState, Seat, Street


def to_cards(cards: List[str]) -> List[Card]:
    return list(map(Card.from_str, cards))


# @overload
# def get_stacks(round_state: RoundState, uuid: str): ...


# @overload
# def get_stacks(round_state: List[Seat], uuid: str): ...
def get_stacks(round_state: Union[RoundState, List[Seat]], uuid: str):
    if isinstance(round_state, list):
        seats = round_state
    else:
        seats = round_state["seats"]
    if seats[1]["uuid"] == uuid:
        seats = seats[::-1]
    assert len(seats) == 2
    assert seats[0]["uuid"] == uuid

    return seats[0]["stack"], seats[1]["stack"]


def get_blind_amount(round_state: RoundState, uuid: str) -> int:
    histories = round_state["action_histories"]
    actions = histories["preflop"]
    return sum(a["amount"] for a in actions if a["uuid"] == uuid and a["action"].endswith("BLIND"))


@dataclass
class ValidActionInfo:
    actions: Set[Literal["fold", "call", "raise"]]
    call_amount: int
    raise_min: int
    raise_max: int


def get_actions_info(actions: List[Action]) -> ValidActionInfo:
    actions_set = set()
    call_amount: Optional[int] = None
    raise_min: Optional[int] = None
    raise_max: Optional[int] = None
    for a in actions:
        assert a["action"] in ("fold", "call", "raise")
        actions_set.add(a["action"])
        if a["action"] == "call":
            call_amount = a["amount"]
        if a["action"] == "raise":
            raise_min = a["amount"]["min"]
            raise_max = a["amount"]["max"]

    assert call_amount is not None
    assert raise_min is not None
    assert raise_max is not None

    return ValidActionInfo(actions_set, call_amount, raise_min, raise_max)


def remap(value: float, old_min: float, old_max: float, seq: List[int]) -> int:
    value = max(min(value, old_max), old_min)
    ratio = (value - old_min) / (old_max - old_min)
    idx = int(ratio * (len(seq) - 1))
    return seq[idx]


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def one_hot(idx: int, total: int):
    arr = [0] * total
    arr[idx] = 1
    return arr


def one_hot_cards(cards: Union[List[str], List[Card]]):
    arr = [0] * 52
    for c in cards:
        if isinstance(c, str):
            c = Card.from_str(c)
        arr[c.to_id() - 1] = 1
    return arr


def get_idx_from_prob(prob: Union[Tensor, np.ndarray, List[float]]) -> int:
    """
    Out put index by p(i) = prob[i].
    """
    if isinstance(prob, Tensor):
        prob = prob.detach().numpy()

    prob = np.cumsum(prob) / np.sum(prob)
    assert np.isclose(prob[-1], 1.0, atol=1e-5)

    r = np.random.rand()
    return int(np.searchsorted(prob, r))


WIN_RATE_SIMULATIONS = 50


def get_win_rate(hole: List[Card], community: List[Card]):
    return estimate_hole_card_win_rate(WIN_RATE_SIMULATIONS, 2, hole, community)


def gen_card(rank: int, community: List[Card]):
    suits = [s for s in Card.SUIT_MAP if all(Card(s, rank) != c for c in community)]
    return Card(random.choice(suits), rank)


BLUFF_THRES = 0.4


def is_bluff(round_state: RoundState, hand_rank_info: HandRankInfo):
    community = to_cards(round_state["community_card"])
    hole_ranks = hand_rank_info["hole"]["high"], hand_rank_info["hole"]["low"]
    hole_cards = [gen_card(r, community) for r in hole_ranks]

    win_rate = get_win_rate(hole_cards, community)
    return win_rate < BLUFF_THRES


def prev_street(s: Street):
    if s == "preflop":
        return None
    return STREET_TUPLE[STREET_TUPLE.index(s) - 1]


def get_last_acton(round_state: RoundState, uuid: str):
    street = round_state["street"]
    cur_actions = round_state["action_histories"][street]
    assert not cur_actions or cur_actions[-1]["uuid"] != uuid
    if cur_actions:
        last_action = cur_actions[-1]
    elif prev := prev_street(street):
        last_action = round_state["action_histories"][prev][-1]
    else:
        last_action = None
    return last_action
