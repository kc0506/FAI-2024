import typing
from typing import Dict, List, Literal, Tuple, TypedDict, Union


class Seat(TypedDict):
    name: str
    uuid: str
    stack: int
    state: str


class Rule(TypedDict):
    initial_stack: int
    max_round: int
    small_blind_amount: int
    ante: int
    blind_structure: dict


class GameInfo(TypedDict):
    player_num: int
    rule: Rule
    seats: List[Seat]


history = {
    "preflop": [
        {"action": "SMALLBLIND", "amount": 5, "add_amount": 5, "uuid": "hazxaqluochxvntqayvkef"},
        {"action": "BIGBLIND", "amount": 10, "add_amount": 5, "uuid": "ninehmbqclapvntylzagrp"},
        {"action": "CALL", "amount": 10, "paid": 5, "uuid": "hazxaqluochxvntqayvkef"},
        {"action": "CALL", "amount": 10, "paid": 0, "uuid": "ninehmbqclapvntylzagrp"},
    ],
    "flop": [
        {"action": "CALL", "amount": 0, "paid": 0, "uuid": "hazxaqluochxvntqayvkef"},
        {"action": "CALL", "amount": 0, "paid": 0, "uuid": "ninehmbqclapvntylzagrp"},
    ],
    "turn": [
        {"action": "CALL", "amount": 0, "paid": 0, "uuid": "hazxaqluochxvntqayvkef"},
        {"action": "CALL", "amount": 0, "paid": 0, "uuid": "ninehmbqclapvntylzagrp"},
    ],
    "river": [],
}


Street = Literal["preflop", "flop", "turn", "river"]
STREET_TUPLE: Tuple[Street, ...] = typing.get_args(Street)

HistoryAction = Literal["SMALLBLIND", "BIGBLIND", "CALL", "RAISE", "FOLD"]
HistoryRecord = TypedDict(
    "HistoryRecord",
    {
        "action": HistoryAction,
        "amount": int,
        "add_amount": int,
        "uuid": str,
    },
)


class RoundState(TypedDict):

    street: Street
    pot: dict
    community_card: List[str]
    dealer_btn: int
    next_player: int
    small_blind_pos: int
    big_blind_pos: int
    round_count: int
    small_blind_amount: int
    seats: List[Seat]
    action_histories: Dict[Street, List[HistoryRecord]]


class Action1(TypedDict):
    action: Literal["fold", "call"]
    amount: int


class Amount(TypedDict):
    min: int
    max: int


class Action2(TypedDict):
    action: Literal["raise"]
    amount: Amount


Action = Union[Action1, Action2]


__round_state = {
    "street": "preflop",
    "pot": {"main": {"amount": 15}, "side": []},
    "community_card": [],
    "dealer_btn": 0,
    "next_player": 1,
    "small_blind_pos": 1,
    "big_blind_pos": 0,
    "round_count": 1,
    "small_blind_amount": 5,
    "seats": [
        {"name": "p2", "uuid": "lotrzopbayqcjcfbqcyvhi", "stack": 990, "state": "participating"},
        {"name": "me", "uuid": "ogxlurncuewyzajfbigzoq", "stack": 995, "state": "participating"},
    ],
    "action_histories": {
        "preflop": [
            {
                "action": "SMALLBLIND",
                "amount": 5,
                "add_amount": 5,
                "uuid": "ogxlurncuewyzajfbigzoq",
            },
            {"action": "BIGBLIND", "amount": 10, "add_amount": 5, "uuid": "lotrzopbayqcjcfbqcyvhi"},
        ]
    },
}


class GameResult(TypedDict):
    rule: Rule
    players: List[Seat]


DispatchAction = Tuple[Literal["fold", "call", "raise"], int]


HandInfo = TypedDict("HandInfo", {"strength": str, "high": int, "low": int})
HoleInfo = TypedDict("HoleInfo", {"high": int, "low": int})


class HandRankInfo(TypedDict):
    hand: HandInfo
    hole: HoleInfo

class RoundResultHandInfo(TypedDict):
    uuid: str
    hand: HandRankInfo

