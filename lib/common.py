"""the abbreviation 'dl' used in variable names stands for depth-limited. a dl-leaf is a leaf of a dl-subgame."""
from collections import defaultdict, Counter, namedtuple
from abc import ABC, abstractmethod
import sys
import random

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.observation import make_observation

import copy

import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

try:
#     from tqdm.notebook import tqdm
    from tqdm import tqdm
except ImportError as e:
    print('{} -- (tqdm is a cosmetic-only progress bar) -- setting `tqdm` to the identity function instead'.format(e))
    tqdm = lambda x: x

import os
import sys
from . import utils
from .utils import debug, info, debug_game, game_score_of_best_response, paste_subpolicy_onto_policy, ResultContainer
    
np.set_printoptions(suppress=True)
utils.DEBUG_LEVEL = 1

###########################

class GameTraversingDataCollector:
    def __init__(self, get_public_tensor_from_state=None, get_targets_from_state=None):
        """get_tensor_from_state should give us public state
        get_target_from_state should give us both private states"""
        self.x_data = []
        self.y_datas = [[], []]
        self.get_public_tensor_from_state = get_public_tensor_from_state
        if get_targets_from_state is None:
            get_targets_from_state = lambda state: state.history_str() # or something
        self.get_targets_from_state = get_targets_from_state
    def repeatedly_traverse(self, iterations, *args, **kwargs):
        for i in tqdm(range(iterations)):
            traverse_game(*args, **kwargs, iteration=i)
        debug("done traversing!  now processing!")
        self.process_data()
    def traverse_game(self, state, pol, training_player, iteration=None):
        while not state.is_terminal():
            if state.current_player() == training_player:
                self.x_data.append(np.array(self.get_public_tensor_from_state(state)))
                p1, p2 = self.get_targets_from_state(state)
                self.y_datas[0].append(p1)
                self.y_datas[1].append(p2)
            legal_actions = state.legal_actions()
            if state.is_chance_node():
                # Sample a chance event outcome.
                outcomes_with_probs = state.chance_outcomes()
            else:
                outcomes_with_probs = list(pol.action_probabilities(state).items())
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
    def process_data(self):
        self.x_data = np.array(self.x_data) # ok this takes a long time... specify the dtype?
        debug('done making x np array')
        for i in range(2):
            self.y_datas[i] = np.array(self.y_datas[i])
            debug('done making y np array for player {}'.format(i))
        # self.x_data = np.reshape(self.x_data, newshape=(len(self.data)/5, 5))
        # self.y_data = keras.utils.to_categorical(self.y_data)

class BiasedPolicy:
    def __init__(self, pol, action_to_bias):
        self.pol = pol
        self.action_to_bias = action_to_bias
    def action_probabilities(self, state):
        original = self.pol.action_probabilities(state)
        if not any(x == self.action_to_bias for x in original):
            return original
        d = {k:original[k] for k in original}
        d[self.action_to_bias] *= 10
        denom = sum(d.values())
        for k in d:
            d[k]/= denom
        return d

def get_biased_policy_from_policy(pol, action_to_bias):
    """action_to_bias is the action we'll weight 10x more than we would in pol.
    i think 0 = fold, 1 = check/call, 2 = bet/raise"""
    return BiasedPolicy(pol, action_to_bias)

def get_value_from_policies(state, blueprint, opponent_policy):
    # replace expected_game_score with a monte-carlo version for larger games
    # (could also implement some caching here, or replace with DNN)
    return expected_game_score.policy_value(state, [blueprint, opponent_policy])

def get_value_from_policies_monte_carlo(state: pyspiel.State, p0, p1, playthroughs=1000):
    num_players = 2
    ans = np.zeros(shape=num_players)
    for i in range(playthroughs):
        s = state.clone()
        while not s.is_terminal():
            if s.is_player_node():
                pol = list([p0, p1][s.current_player()].action_probabilities(s).items())
            else:
                pol = s.chance_outcomes()
            action = random.choices(pol, [p[1] for p in pol])
            s = s.child(action[0][0])
        ans += s.returns()
    return ans/playthroughs
    
def get_value_function_for_policy(blueprint, opponent_policy):
    """returns a function that maps state -> value"""
    cache = dict()
    def value_function(state):
        if state.history_str() in cache:
            return cache[state.history_str()]
        val = get_value_from_policies(state, blueprint, opponent_policy)
        cache[state.history_str()] = val
        return val
    return value_function

def get_biased_opponent_strategies(blueprint):
    opponent_strategies = [blueprint,
                           get_biased_policy_from_policy(blueprint, 0),
                           get_biased_policy_from_policy(blueprint, 1),
                           get_biased_policy_from_policy(blueprint, 2)]
    opponent_strategy_functions = [get_value_function_for_policy(blueprint, strat) for strat in opponent_strategies]
    return opponent_strategy_functions

def make_random_strategy_value_function(blueprint):
    random_strategy = make_random_strategy()
    return get_value_function_for_policy(blueprint, random_strategy)

def make_random_strategy():
    global global_seed
    global_seed += 1
    return RandomPolicy(global_seed)

global_seed = 0
class RandomPolicy():
    def __init__(self, seed_salt):
        self.seed_salt = seed_salt
    def action_probabilities(self, state, player=None):
        random.seed(state.information_state_string()+str(self.seed_salt))
        denom = 0
        ans = dict()
        for action in state.legal_actions():
            r = random.randint(0,4)
            ans[action] = r
            denom += r
        if denom == 0:
            return {a:1/len(state.legal_actions()) for a in ans}
        return {a:ans[a]/denom for a in ans}

def get_max_game_depth(game):
    return max(get_all_states.get_all_states(game).values(), key=lambda s: s.move_number()).move_number()

def to_one_player_tabular_policy(game, pol, pid):
    """the default to_tabular() will error for one-player policies (e.g. best responses)"""
    class OnePlayerTabularPolicy(policy.TabularPolicy):
        def action_probabilities(self, state, player_id=None):
            if player_id is None:
                player_id = state.current_player()
            if player_id != pid:
                raise ValueError(f'querying policy for player {player_id}, but the policy is only for player {pid} actions')
            return super().action_probabilities(state, player_id)
    to_return = OnePlayerTabularPolicy(game)
    for state in get_all_states.get_all_states(game).values():
        if state.current_player() == pid:
            p = to_return.policy_for_key(to_return._state_key(state, pid))[:]
            p.fill(0)
            for action, probability in pol.action_probabilities(state).items():
                p[action] = probability
    return to_return

def xdo_default_strategy(state):
    n = len(state.legal_actions())
    return {a:1/n for a in state.legal_actions()}

class LazyTabularPolicy(policy.Policy):
    """Takes in a partial policy and becomes a full policy, using a default policy for states where the partial policy is absent."""
    # TODO: duplicated code in xdo.py and sdo.py.  Also could be renamed to: PolicyWithDefault?
    def __init__(self, game, p, default_policy_fn, pids=None):
        """for SDO, default_policy_fn should be a default pure strategy, e.g. first-action-policy lambda s: {s.legal_actions()[0]: 1}
           for XDO, default_policy_fn should be uniform random i.e. lambda s: {a:1/n for a in state.legal_actions()}"""
        # TODO: why not just pass Policys instead of fns here?  More general.
        if pids is None:
            pids = [0,1]
        super().__init__(game, pids)
        self.p = p
        self.default_policy_fn = default_policy_fn
    def action_probabilities(self, state, player_id=None):
        try:
            return self.p.action_probabilities(state, player_id)
        except KeyError as e :
            # print(e)
            return self.default_policy_fn(state)
