import random
from typing import List, Literal, Tuple


from .dqn import DQN, DQN_Dueling, Stretagy, decode_action, get_state_feature
from .memory import History, SavePokerPlayer, save_to_replay_multi
from .my_types import *
from .utils import get_actions_info, get_last_acton, get_win_rate, to_cards


class MyDQNPlayer(SavePokerPlayer):

    dqn_model: DQN
    stretagy: Stretagy
    epsilon: float = 0.1

    def __init__(self, qdn_model: DQN, stretagy: Stretagy):
        self.dqn_model = qdn_model
        self.stretagy = stretagy

    def declare_action(
        self, valid_actions: List[Action], hole_card: List[str], round_state: RoundState
    ) -> Tuple[Literal["fold", "call", "raise"], int]:
        actions_info = get_actions_info(valid_actions)

        hole = to_cards(hole_card)
        community = to_cards(round_state["community_card"])
        win_rate = get_win_rate(hole, community)

        _action_not_used = ("fold", 0)
        history = History(
            self.uuid,
            hole,
            community,
            round_state,
            win_rate,
            None,
            _action_not_used,
            actions_info,
        )

        action_q = self.dqn_model(get_state_feature(history))
        return decode_action(action_q, actions_info, self.stretagy, self.epsilon)

    def receive_game_start_message(self, game_info: GameInfo):
        pass

    def receive_round_start_message(
        self, round_count: int, hole_card: List[str], seats: List[Seat]
    ):
        pass

    def receive_street_start_message(self, street: str, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class DQNPlayerHeuristic(MyDQNPlayer):

    dqn_model: DQN  # type: ignore

    def __init__(
        self,
        qdn_model: DQN,
        stretagy: Stretagy,
        # heuristic parameters
        fold=0.2,
        all_in=0.8,
        avoid_amount=25,
        bluff_p=0.1,
        bluff_raise_amount=100,
        bye_noise=0.6,
        bye_thres=0.4,
    ):
        super().__init__(qdn_model, stretagy)

        self.fold = fold
        self.all_in = all_in
        self.avoid_amount = avoid_amount

        self.bluff_p = bluff_p
        self.bluff_raise_amount = bluff_raise_amount

        self.bye_noise = bye_noise
        self.bye_thres = bye_thres

    def declare_action(
        self,
        valid_actions: List[Action],
        hole_card: List[str],
        round_state: RoundState,
    ) -> Tuple[Literal["fold", "call", "raise"], int]:

        actions_info = get_actions_info(valid_actions)

        hole = to_cards(hole_card)
        community = to_cards(round_state["community_card"])
        win_rate = get_win_rate(hole, community)

        street = round_state["street"]
        last_action = get_last_acton(round_state, self.uuid)
        if street == "river" and last_action and win_rate < 0.75:
            if last_action["action"] == "raise":
                return "fold", 0

        if (
            last_action
            and last_action["action"] == "RAISE"
            and win_rate < self.bye_thres
            and random.random() < self.bye_noise
        ):
            return "fold", 0

        if win_rate < self.fold:
            return "fold", 0
        if win_rate > self.all_in:
            return "raise", actions_info.raise_max
        if win_rate < 0.5 and random.random() < self.bluff_p:
            return "raise", self.bluff_raise_amount

        return super().declare_action(valid_actions, hole_card, round_state)


MyDQNPlayer = save_to_replay_multi(MyDQNPlayer)  # type: ignore
DQNPlayerHeuristic = save_to_replay_multi(DQNPlayerHeuristic)  # type: ignore



def setup_ai():
    return MyDQNPlayer(DQN.from_name('heuristic_0-5_5-7_single'), 'greedy')