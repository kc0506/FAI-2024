import os
import pickle
import random
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import (
    Deque,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
)


from game.engine.card import Card
from game.players import BasePokerPlayer
from .my_types import *
from .utils import (
    ValidActionInfo,
    get_actions_info,
    get_blind_amount,
    get_stacks,
    get_win_rate,
    is_bluff,
    to_cards,
)


@dataclass
class History:
    uuid: str
    hole_cards: List[Card]
    community_cards: List[Card]
    round_state: RoundState
    win_rate: float
    #
    next_history: Optional["History"]
    action: Tuple[Literal["fold", "call", "raise"], int]
    actions_info: ValidActionInfo
    #
    bluff_count: int = 0  # per game


# Memory records are generic and can be converted to vectors in different manners.
MemoryRecord = Tuple[History, int]  # (history, reward)

MemoryName = Literal[
    "heuristic_random",
    "heuristic_baseline0",
    "heuristic_baseline1",
    "heuristic_baseline2",
    "heuristic_baseline3",
    "heuristic_baseline4",
    "heuristic_baseline5",
    "heuristic_baseline6",
    "heuristic_baseline7",
    #
    "equity_baseline0",
    "equity_baseline1",
    "equity_baseline2",
    "equity_baseline3",
    "equity_baseline4",
    "equity_baseline5",
    "equity_baseline6",
    "equity_baseline7",
    #
    "dqn_baseline0",
    "dqn_baseline1",
    "dqn_baseline2",
    "dqn_baseline3",
    "dqn_baseline4",
    "dqn_baseline5",
    "dqn_baseline6",
    "dqn_baseline7",
    #
    "dqn_double_baseline5",
    "dqn_dueling_baseline5",
    "d3qn_baseline5",
]


class ReplayMemory:
    DATA_DIR = Path(__file__).parent / "./data"

    queue: Deque[MemoryRecord]

    def __init__(self, max_size=100000) -> None:
        self.queue = deque(maxlen=max_size)

    def append(self, record: MemoryRecord) -> None:
        self.queue.append(record)

    def clear(self):
        self.queue.clear()

    def sample(self, batch_size: int) -> List[MemoryRecord]:
        return random.sample(self.queue, batch_size)

    def __add__(self, other: "ReplayMemory") -> "ReplayMemory":
        self.queue.extend(other.queue)
        return self

    def __len__(self):
        return len(self.queue)

    @classmethod
    def from_name(cls, path: MemoryName) -> "ReplayMemory":
        return cls.__from_name(path)

    # ! hacky singleton
    @classmethod
    @lru_cache(maxsize=None)
    def __from_name(cls, path: MemoryName) -> "ReplayMemory":
        memory = ReplayMemory()

        dir = cls.DATA_DIR / (path + ".pkl")

        # for p in dir.glob("*"):
        p = dir
        if dir.exists():
            with open(p, "rb") as f:
                m = pickle.load(f)
                memory += m
        return memory

    @classmethod
    def from_names(cls, paths: List[MemoryName] = []) -> "ReplayMemory":
        memory = ReplayMemory()
        for p in paths:
            memory += ReplayMemory.from_name(p)
        return memory

    def save_file(self, path: MemoryName) -> None:
        # dir = self.DATA_DIR / path
        dir = self.DATA_DIR / (path + ".pkl")
        # os.makedirs(dir, exist_ok=True)
        # file_name = datetime.now().strftime("%m%d_%H%M") + ".pkl"
        # with open(dir / file_name, "wb") as f:
        with open(dir, "wb") as f:
            pickle.dump(self, f)


try:
    os.makedirs(ReplayMemory.DATA_DIR, exist_ok=True)
except:
    pass


class SavePokerPlayer(BasePokerPlayer):

    enable_save_flag: bool = False
    save_dir: MemoryName

    @property
    def memory(self):
        return ReplayMemory.from_name(self.save_dir)

    def enable_save(self, name: MemoryName):
        print("hi name")
        self.set_save_params(True, name)

    def set_save_params(self, enable: bool, save_dir: MemoryName):
        self.enable_save_flag = enable
        self.save_dir = save_dir

    def save(self):
        memory = ReplayMemory.from_name(self.save_dir)
        if self.enable_save_flag and self.save_dir and len(memory):
            memory.save_file(self.save_dir)
        print(f"saved {len(memory)} records to {self.save_dir}")


T = TypeVar("T", Type[SavePokerPlayer], None)


def save_to_replay_multi(cls: T) -> T:
    """
    Q(s,a) = r + γ^{n} * max_a' Q(s',a')
    s': first action of next round, and r
    """

    assert cls is not None

    # original_init = cls.__init__
    original_new = cls.__new__

    original_declare_action = cls.declare_action
    original_receive_game_start_message = cls.receive_game_start_message
    original_receive_round_start_message = cls.receive_round_start_message
    original_receive_round_result_message = cls.receive_round_result_message

    def new(cls, *args, **kwargs):
        obj: SavePokerPlayer = original_new(cls)

        total = 0
        round_start_stacks = -1

        bluff_count = 0

        # memory = ReplayMemory()
        histories: List[History] = []

        prev_history: Optional[History] = None

        def declare_action(
            valid_actions: List[Action], hole_card: List[str], round_state: RoundState
        ) -> Tuple[Literal["fold", "call", "raise"], int]:
            nonlocal bluff_count

            action = original_declare_action(obj, valid_actions, hole_card, round_state)
            hole = to_cards(hole_card)
            community = to_cards(round_state["community_card"])
            win_rate = get_win_rate(hole, community)
            new_history = History(
                obj.uuid,
                hole,
                community,
                round_state,
                win_rate,
                None,
                action,
                get_actions_info(valid_actions),
                #
                bluff_count=bluff_count,
            )
            histories.append(new_history)

            nonlocal prev_history
            if prev_history:
                prev_history.next_history = new_history

            return action

        def receive_game_start_message(game_info: GameInfo):
            nonlocal bluff_count
            bluff_count = 0

        def receive_round_start_message(round_count: int, hole_card: List[str], seats: List[Seat]):
            nonlocal round_start_stacks
            nonlocal histories
            round_start_stacks = get_stacks(seats, obj.uuid)[0]
            histories.clear()

            return original_receive_round_start_message(obj, round_count, hole_card, seats)

        def receive_round_result_message(
            winners: List[Seat], hand_info: List[RoundResultHandInfo], round_state: RoundState
        ):

            round_end_stacks = get_stacks(round_state, obj.uuid)[0]
            # return original_receive_round_result_message(self, winners, hand_info, round_state)
            blind = get_blind_amount(round_state, obj.uuid)
            round_delta = round_end_stacks - (round_start_stacks + blind)
            nonlocal total
            total += round_delta

            nonlocal bluff_count
            if hand_info:
                try:
                    enemy_hand = next(h for h in hand_info if h["uuid"] != obj.uuid)
                    bluff_count += is_bluff(round_state, enemy_hand["hand"])
                except StopIteration:
                    pass

            # print('delta', round_delta)
            # if bluff_count:
            #     print(bluff_count)
            setattr(obj, "bluff_count", bluff_count)

            if obj.enable_save_flag and obj.save_dir:
                memory = ReplayMemory.from_name(obj.save_dir)
                for h in histories:
                    memory.append((h, round_delta))
            return original_receive_round_result_message(obj, winners, hand_info, round_state)

        obj.declare_action = declare_action
        obj.receive_game_start_message = receive_game_start_message
        obj.receive_round_start_message = receive_round_start_message
        obj.receive_round_result_message = receive_round_result_message

        return obj

    cls.__new__ = new  # type: ignore

    return cls


def get_steps_to_next_round(history: History) -> int:
    steps = 1
    while history:
        next_history = history.next_history
        if next_history is None or next_history.round_state["street"] == "preflop":
            break
        steps += 1
        history = next_history
    assert steps < 5
    return steps


def is_new_round(history: Optional[History]) -> bool:
    return history is None or history.round_state["street"] == "preflop"


def extend_history(history: History, depth: int):
    cur_his = history

    res: List[Optional[History]] = []
    for _ in range(depth):
        if cur_his is None:
            break

        res.append(cur_his)
        cur_his = cur_his.next_history
    for _ in range(depth - len(res)):
        res.insert(0, None)
    return res
