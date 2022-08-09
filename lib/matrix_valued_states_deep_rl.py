# TODO (probably never gonna do this): put this and matrix_valued_states into a directory instead of prefix naming
from typing import List

from open_spiel.python import rl_agent
import numpy as np 

class SyntheticState:
    def __init__(self, information_state_string, legal_actions, current_player):
        self._information_state_string = information_state_string
        self._legal_actions = legal_actions
        self._current_player = current_player
    def legal_actions(self, pl):
        return self._legal_actions
    def information_state_string(self, pl=None):
        return self._information_state_string
    def current_player(self):
        return self._current_player

class PolicyAgent(rl_agent.AbstractAgent):
    def __init__(self, player_id, num_actions, rng, policy):
        self._player_id = player_id
        self._rng = rng
        self._num_actions = num_actions
        self._policy = policy
    def action_probabilities(self, *args, **kwargs):
        return self._policy.action_probabilities(*args, **kwargs)
    def step(self, time_step, is_evaluation=False):
        if time_step.last():
            return
        synthetic_state = SyntheticState(
            time_step.observations['information_state_string'][self._player_id],
            time_step.observations['legal_actions'][self._player_id],
            time_step.observations["current_player"],
        )
        pol = self._policy.action_probabilities(synthetic_state, self._player_id)
        pol_list = list(zip(*pol.items()))
        action = self._rng.choice(pol_list[0], p=pol_list[1])
        probs = np.zeros(self._num_actions)
        for k, v in pol.items():
            probs[k] = v
        return rl_agent.StepOutput(action=action, probs=probs)
class MixedPolicyAgent(rl_agent.AbstractAgent):
    def __init__(self, player_id, num_actions, rng, trunk_policy, continuation_policies: List[rl_agent.AbstractAgent]):
        self._player_id = player_id
        self._rng = rng
        self._num_actions = num_actions
        self._trunk_policy = trunk_policy
        self._continuation_choice = None
        self._continuation_policies = continuation_policies
    def select_continuation_policy(self, subgame_leaf_information_state_string):
        synthetic_state = SyntheticState(
            subgame_leaf_information_state_string,
            list(range(len(self._continuation_policies)))
        )
        pol = self._trunk_policy.action_probabilities(synthetic_state, self._player_id)
        pol_list = list(zip(*pol.items()))
        action = self._rng.choice(pol_list[0], p=pol_list[1])
        self._continuation_choice = action
        return action

    def _reset(self):
        self._continuation_choice = None

    def step(self, time_step, is_evaluation=False):
        if time_step.first():
            self._reset()
        if time_step.last():
            # this seems hacky? Not sure if there's somewhere else to reset internal state.  (I guess the idea is that agents shouldn't have internal state?)
            # hm... maybe the better design would be if I make a wrapper around the *policy* instead of around the *rl agent* and so this agent just always
            # delegates to the policy in the same way?
            self._reset()
            return 
        if self._continuation_choice is None:
            synthetic_state = SyntheticState(
                time_step.observations['information_state_string'][self._player_id],
                time_step.observations['legal_actions'][self._player_id]
            )
            pol = self._trunk_policy.action_probabilities(synthetic_state, self._player_id)
            pol_list = list(zip(*pol.items()))
            action = self._rng.choice(pol_list[0], p=pol_list[1])
            probs = np.zeros(self._num_actions)
            for k, v in pol.items():
                probs[k] = v
            return rl_agent.StepOutput(action=action, probs=probs)
        else:
            return self._continuation_policies[self._continuation_choice].step(time_step, is_evaluation)