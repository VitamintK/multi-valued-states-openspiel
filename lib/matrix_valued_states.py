"""the abbreviation 'dl' used in variable names stands for depth-limited. a dl-leaf is a leaf of a dl-subgame."""
from collections import defaultdict, Counter, namedtuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy
import time
from enum import Enum
import os
import sys
import random
import uuid

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.observation import make_observation

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

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from . import subgame_solving
from . import utils
from .utils import debug, info, debug_game, game_score_of_best_response, paste_subpolicy_onto_policy, ResultContainer
from .common import get_value_from_policies, get_max_game_depth, get_value_from_policies_monte_carlo

np.set_printoptions(suppress=True)
utils.DEBUG_LEVEL = 1

########

CFRSolver = cfr.CFRSolver
CFRPlusSolver = cfr.CFRPlusSolver
# CFRSolver = pyspiel.CFRSolver
# CFRPlusSolver = pyspiel.CFRPlusSolver
# TODO: ideally we'd like to use the C++ solvers since they're probably faster, but there are some issues that don't allow us to use them as drop-in replacements yet

CFRParameters = namedtuple('CFR_parameters', ['type', 'iterations'])
class CFRType(str, Enum):
    EXTERNALMCCFR = 'external_mccfr'
    VANILLA = 'vanilla'
    PLUS = 'plus'

def solve_game(game, cfr_parameters: CFRParameters):
    if cfr_parameters.type == CFRType.VANILLA:
        dl_cfr_solver = CFRSolver(game)
        for i in tqdm(range(cfr_parameters.iterations)):
            dl_cfr_solver.evaluate_and_update_policy()
    elif cfr_parameters.type == CFRType.EXTERNALMCCFR:
        dl_cfr_solver = external_mccfr.ExternalSamplingSolver(game, external_mccfr.AverageType.SIMPLE)
        for i in tqdm(range(cfr_parameters.iterations)):
            dl_cfr_solver.iteration()
    elif cfr_parameters.type == CFRType.PLUS:
        dl_cfr_solver = CFRPlusSolver(game)
        for i in tqdm(range(cfr_parameters.iterations)):
            dl_cfr_solver.evaluate_and_update_policy()
    else:
        raise ValueError('cfrparameters type {} not recognized'.format(cfr_parameters.type))
    return dl_cfr_solver

def get_tabular_policy(cfr_solver):
    avg_policy = cfr_solver.average_policy()
    if not isinstance(avg_policy, policy.TabularPolicy):
        avg_policy = avg_policy.to_tabular()
    return avg_policy

class StateValuer:
    def get_value(self, state):
        return NotImplemented

class NormalFormValuer(StateValuer):
    def __init__(self, policies, valuing_method='exhaustive', monte_carlo_iterations=100000):
        """valuing_method := 'exhaustive' | 'monte-carlo'
                exhaustive uses expected_game_score to calculate the exact expected score
                monte-carlo samples playthroughs of the game to estimate the expected score
        """
        assert len(policies) == 2
        self.valuing_method = valuing_method
        self.monte_carlo_iterations = monte_carlo_iterations
        self.players = 2
        self.policies = [policy for policy in policies]
        self.cache = dict()
    def get_value(self, policies,state):
        policies = tuple(policies)
        key = (policies,state.history_str())
        if key in self.cache:
            return self.cache[key]
        if self.valuing_method == 'exhaustive':
            val = get_value_from_policies(state, *[self.policies[i][policies[i]] for i in range(len(policies))])
        elif self.valuing_method == 'monte-carlo':
            val = get_value_from_policies_monte_carlo(state, *[self.policies[i][policies[i]] for i in range(len(policies))], playthroughs=self.monte_carlo_iterations)
        else:
            raise ValueError(f'{self.valuing_method = }')
        self.cache[key] = val
        return val
    def get_num_strategy_options(self):
        return [len(x) for x in self.policies]
    def add_policies(self,policies):
        assert len(policies) == self.players, f'not the right number of policies in {policies}'
        for i, policy in enumerate(policies):
            self.policies[i].append(policy)
    def add_policy(self, player, policy):
        self.policies[player].append(policy)
    def print_payoff_matrix(self, state):
        ans = []
        for i in range(len(self.policies[0])):
            row = []
            for j in range(len(self.policies[1])):
                row.append(self.get_value((i,j), state))
            ans.append(row)
        for row in ans:
            print(' '.join(f'{x[0]:5.2f}' for x in row))

###########################################################################################################################
###
### Define Depth-limited Subgame and Depth-limited Subgame State
### 
#############

class DepthLimitedSubgameConcreteStateBase(ABC):
    """static class instead of ostensibly better instance class so that deepcopying is more straightforward (flatter)"""
    @staticmethod
    @abstractmethod
    def current_player(context):
        raise NotImplementedError
    @staticmethod
    @abstractmethod
    def _legal_actions(context, player=None):
        raise NotImplementedError
    @staticmethod
    @abstractmethod
    def _apply_action(context, action):
        raise NotImplementedError
    @staticmethod
    @abstractmethod
    def undo_action(context, move, action):
        raise NotImplementedError
    @staticmethod
    @abstractmethod
    def is_terminal(context):
        raise NotImplementedError
    @staticmethod
    @abstractmethod
    def returns(context):
        raise NotImplementedError

@dataclass
class SubgameStateData:
    """Using this as a workaround for the issue in https://github.com/deepmind/open_spiel/issues/641 -- to define a custom __deepcopy__ class
    as opposed to just storing the data in the actual SubgameState class."""
    # note: just trying out dataclass for fun, this could easily be a normal class
    state_status: DepthLimitedSubgameConcreteStateBase
    _choosing_player: int
    _chancing_node: int
    chance_moves: list
    strategy_choices: list
    internal_state: pyspiel.State
    def __deepcopy__(self, memo=None):
        return SubgameStateData(
            state_status=self.state_status,
            _choosing_player=self._choosing_player,
            _chancing_node=self._chancing_node,
            chance_moves=copy.copy(self.chance_moves),
            strategy_choices=copy.copy(self.strategy_choices),
            internal_state=self.internal_state if self.internal_state is None else self.internal_state.clone(),
        )

def get_subgame_leaf_information_state_string(state, strategy_choices, player):
    return f"{state.information_state_string(player)}::{len(strategy_choices)}::{strategy_choices[player] if player < len(strategy_choices) else 'none'}"

Params = namedtuple('params', ['actual_game', 'private_chances', 'num_strategy_options', 'is_state_dl_leaf', 'private_chance_fn', 'strategy_value_fn'])
def make_augmented_subgame(actual_game, private_chances, num_strategy_options, is_state_dl_leaf, private_chance_fn, strategy_value_fn):
    """
    private_chances: [[(c1, p1), (c2, p2) ...], [...]] 2 lists (one per player) consisting of tuples of state "keys" and probs for each player's chance node at the root.
    private_chance_fn: [i,j] => state, where i and j are the indices of each player's chance outcome.  (indices, not the keys themselves) as of aug 19 2021
    num_strategy_options: [x,y] where x is the number of leaf continuation strategies that player 0 has, and y the number that player 1 has
    """
    params = Params(actual_game, private_chances, num_strategy_options, is_state_dl_leaf, private_chance_fn, strategy_value_fn)
    name = uuid.uuid1().hex
    # make the game_type
    overrides = {
                'provides_information_state_tensor': False,
                'provides_observation_tensor': False,
                'short_name': 'augmentedsubgame' + name,
                'long_name': 'augmentedsubgame' + name,
            }
    all_attrs = ["short_name","long_name","dynamics","chance_mode","information","utility","reward_model","max_num_players","min_num_players","provides_information_state_string","provides_information_state_tensor","provides_observation_string","provides_observation_tensor","parameter_specification"]
    kwargs = {k: (actual_game.get_type().__getattribute__(k) if k not in overrides else overrides[k]) for k in all_attrs}
    game_type = pyspiel.GameType(**kwargs)
    class AugmentedSubgame(pyspiel.Game):
        def __init__(self, params=None):
            # actual_game, private_chances, num_strategy_options, is_state_dl_leaf, private_chance_fn, dl_leaf_value_function = params
            self.game = actual_game
            self.private_chance_fn = private_chance_fn
            self.is_state_dl_leaf = is_state_dl_leaf
            self.private_chances = private_chances
            self.num_strategy_options = num_strategy_options
            self.strategy_value_fn = strategy_value_fn
            self._max_chance_outcomes =  max(*[len(x) for x in self.private_chances], self.game.max_chance_outcomes())
            self._num_distinct_actions = max(*self.num_strategy_options, self.game.num_distinct_actions())
            game_info = pyspiel.GameInfo(
                num_distinct_actions=self._num_distinct_actions,  
                max_chance_outcomes=self._max_chance_outcomes , 
                num_players=self.game.num_players(),
                min_utility=self.game.min_utility(),
                max_utility=self.game.max_utility(),
                utility_sum=0.0,
                max_game_length=int(self.game.max_game_length()) + 4 # TODO: this is hardcoded
            ) 
            super().__init__(self.get_type(), game_info, dict())
        def __getattr__(self, attr):
            # hacky hacky hacky hacky
            # if attr == '__deepcopy__':
            #     raise AttributeError
            print(attr, 'ahhhh!')
            if attr.startswith('__'):
                raise AttributeError()
            assert attr != 'new_initial_state'
            return self.game.__getattribute__(attr)
        def get_type(self):
            return game_type
        def new_initial_state(self):
            return DepthLimitedSubgameState(
                    game=self,
                    # is_state_dl_leaf=self.is_state_dl_leaf,
                    # private_chances=self.private_chances,
                    # private_chance_fn=self.private_chance_fn,
                    # num_strategy_options=self.num_strategy_options,
                    # strategy_value_fn=self.strategy_value_fn,
                )
        def make_py_observer(self, iig_obs_type=None, params=None):
            # return make_observation(self.game, iig_obs_type) # TODO, this is false
            return SubgameObserver(self.game, iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params)
        # def register(self):
        #     pyspiel.register_game(self.get_type(), AugmentedSubgame)
    pyspiel.register_game(game_type, AugmentedSubgame)

    class DepthLimitedSubgameState(pyspiel.State):
        """kinda trying State Pattern: https://refactoring.guru/design-patterns/state maybe overkill?
        Essentially this treats the game as a state machine that's either in pre-game (chance node chooses each player's private info),
        in-game (the original game), or post-game (each player chooses their strategy for the endgame).
        """
        def __init__(self,
                    game,
                    # is_state_dl_leaf,
                    # num_strategy_options,
                    # strategy_value_fn,
                    # private_chances,
                    # private_chance_fn
                    ):
            super().__init__(game)
            # self.game = game
            # self.is_state_dl_leaf = is_state_dl_leaf
            # self.num_strategy_options = num_strategy_options
            # self.strategy_value_fn = strategy_value_fn
            # self.private_chances = private_chances
            # self.private_chance_fn = private_chance_fn
            self.data = SubgameStateData(
                state_status = PregameSubgameState,
                _choosing_player = None,
                _chancing_node = 0,
                chance_moves = [],
                strategy_choices = [],
                internal_state = None,
            )
        def state_status(self):
            return self.data.state_status
        def get_internal_state(self):
            return self.data.internal_state
        def current_player(self):
            return self.state_status().current_player(self.data)
        def _legal_actions(self, player=None):
            return self.state_status()._legal_actions(self.data)
        def chance_outcomes(self):
            """Returns the possible chance outcomes and their probabilities."""
            return self.state_status().chance_outcomes(self.data)
        def _apply_action(self, action):
            self.state_status()._apply_action(self.data, action)
        def undo_action(self, player, action):
            self.state_status().undo_action(self.data, player, action)
        def returns(self):
            return self.state_status().returns(self.data)
        def is_terminal(self):
            return self.state_status().is_terminal(self.data)
        def __str__(self):
            return self.state_status().__str__(self.data)
        def information_state_string(self, player=None): # TODO: this should be implemented instead by a make_py_observer
            return self.state_status().information_state_string(self.data, player)
        def action_to_string(self, action):
            return self.state_status().action_to_string(self.data, action)

        # def __deepcopy__(self, memo=None):
        #     # if we don't override deepcopy here, it tries to do a full deepcopy of everything, including self.game
        #     print('is this even being called!?')
        #     clone = DepthLimitedSubgameState(
        #         game = self.game,
        #         is_state_dl_leaf = self.is_state_dl_leaf,
        #         num_strategy_options= self.num_strategy_options,
        #         strategy_value_fn= self.strategy_value_fn,
        #         private_chances=self.private_chances,
        #         private_chance_fn=self.private_chance_fn    
        #     )
        #     clone._choosing_player = self._choosing_player
        #     clone._chancing_node = self._chancing_node
        #     clone.chance_moves = copy.deepcopy(self.chance_moves)
        #     clone.strategy_choices = copy.deepcopy(self.strategy_choices)
        #     clone.state_status = self.state_status
        #     if self.internal_state is not None:
        #         clone.internal_state = self.internal_state.clone()
        #     return clone
        # def clone(self):
        #     # this doesn't call __deepcopy__() by default, so we define it explicitly here.
        #     return self.__deepcopy__()
        # def child(self, a):
        #     # this doesn't call self.clone() by default, so we define it explicitly here.
        #     c = self.clone()
        #     c.apply_action(a)
        #     return c

    class PregameSubgameState(DepthLimitedSubgameConcreteStateBase):
        @staticmethod
        def current_player(context):
            return pyspiel.PlayerId.CHANCE
        @staticmethod
        def _legal_actions(context, player=None):
            if context._chancing_node == len(params.private_chances) - 1:
                actions = []
                for action in range(len(params.private_chances[context._chancing_node])):
                    try:
                        params.private_chance_fn(context.chance_moves + [action])
                        actions.append(action)
                    except SynthesisError:
                        # TODO: this shouldn't happen using an exact model, right?
                        continue
                return actions
            else:
                return list(range(len(params.private_chances[context._chancing_node])))
        @staticmethod
        def _apply_action(context, action):
            context.chance_moves.append(action)
            context._chancing_node += 1
            if context._chancing_node >= len(params.private_chances):
                context.internal_state = params.private_chance_fn(context.chance_moves)
                if (not context.internal_state.is_terminal()) and params.is_state_dl_leaf(context.internal_state):
                    # TODO(code structure): this condition is simply the condition in _apply_action for IngameSubgameState, should be pulled out.
                    context.state_status = PostgameSubgameState
                    context._choosing_player = 0
                else:
                    context.state_status = IngameSubgameState
        @staticmethod
        def undo_action(context, player, action):
            if len(context.chance_moves) == 0:
                raise ValueError('cant undo initial state')
            context.chance_moves.pop()
            context._chancing_node -= 1
        @staticmethod
        def chance_outcomes(context):
            legal_actions = PregameSubgameState._legal_actions(context) #<----------------
            denom = sum(params.private_chances[context._chancing_node][i][1] for i in legal_actions)
            return [(i, params.private_chances[context._chancing_node][i][1]/denom) for i in legal_actions]
        @staticmethod
        def is_terminal(context):
            return False
        @staticmethod
        def returns(context):
            return [0,0]
        @staticmethod
        def action_to_string(context, action):
            return str(action)
        @staticmethod
        def __str__(context):
            return f'pregame: chance nodes picked: {context.chance_moves}'

    class IngameSubgameState(DepthLimitedSubgameConcreteStateBase):
        @staticmethod
        def current_player(context):
            return context.internal_state.current_player()
        @staticmethod
        def _legal_actions(context, player=None):
            if player is not None:
                return context.internal_state.legal_actions(player)
            else:
                return context.internal_state.legal_actions()
        @staticmethod
        def chance_outcomes(context):
            return context.internal_state.chance_outcomes()
        @staticmethod
        def _apply_action(context, action):
            context.internal_state.apply_action(action)
            if (not context.internal_state.is_terminal()) and params.is_state_dl_leaf(context.internal_state):
                context.state_status = PostgameSubgameState
                context._choosing_player = 0
        @staticmethod
        def undo_action(context, player, action):
            # TODO(correctness): should be able to recognize when to undo back to PregameSubgameState
            context.internal_state.undo_action(player, action)
        @staticmethod
        def is_terminal(context):
            return context.internal_state.is_terminal()
        @staticmethod
        def returns(context):
            return context.internal_state.returns()
        @staticmethod
        def action_to_string(context, action):
            return context.internal_state.action_to_string(action)
        @staticmethod
        def __str__(context):
            return context.internal_state.__str__()
        @staticmethod
        def information_state_string(context, player):
            if player is not None:
                return context.internal_state.information_state_string(player) # TODO: maybe this should be in the pyobserver?
            else:
                return context.internal_state.information_state_string()

    class PostgameSubgameState(DepthLimitedSubgameConcreteStateBase):
        @staticmethod
        def current_player(context):
            return context._choosing_player
        @staticmethod
        def _legal_actions(context, player=None):
            return list(range(params.num_strategy_options[context._choosing_player]))
        @staticmethod
        def _apply_action(context, action):
            context.strategy_choices.append(action)
            context._choosing_player += 1
            if context._choosing_player >= len(params.num_strategy_options):
                context.state_status = TerminalSubgameState
        @staticmethod
        def undo_action(context, player, action):
            if len(context.strategy_choices) == 0:
                context.state_status = IngameSubgameState
                context.internal_state.undo_action(player, action)
            else:
                context.strategy_choices.pop()
                context._choosing_player -= 1
        @staticmethod
        def is_terminal(context):
            return False
        @staticmethod
        def returns(context):
            return [0,0]
        @staticmethod
        def action_to_string(context, action):
            return str(action)
        @staticmethod
        def __str__(context):
            return f'{IngameSubgameState.__str__(context)} in postgame. strategy choices (per player): {context.strategy_choices}'
        @staticmethod
        def information_state_string(context, player):
            # TODO: this should be in pyobserver
            if player is None:
                player = context._choosing_player
            return get_subgame_leaf_information_state_string(context.internal_state, context.strategy_choices, player)

    class TerminalSubgameState(DepthLimitedSubgameConcreteStateBase):
        @staticmethod
        def current_player(context):
            return pyspiel.PlayerId.TERMINAL
        @staticmethod
        def _legal_actions(context, player=None):
            pass
        @staticmethod
        def _apply_action(context, action):
            pass
        @staticmethod
        def undo_action(context, player, action):
            context.state_status = PostgameSubgameState
            context.strategy_choices.pop()
            context._choosing_player -= 1
        @staticmethod
        def is_terminal(context):
            return True
        @staticmethod
        def returns(context):
            if context.internal_state.is_terminal():
                return context.internal_state.returns()
            return params.strategy_value_fn(context.strategy_choices,context.internal_state)
        @staticmethod
        def __str__(context):
            return f'terminal: {PostgameSubgameState.__str__(context)}'

    return AugmentedSubgame()

def get_subgame_status(state):
    # TODO(bad)
    # this doesn't seem like the best way to do this but not sure what is.
    # maybe a static method on each state_status that returns an enum?
    return state.state_status().__name__

class SubgameObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""
  def __init__(self, base_game, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    self.observer = make_observation(base_game, iig_obs_type, params)
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(row, column, cell type)`.
    # self.tensor = np.zeros(3 * 3 * 3, np.float32)
    self.tensor = self.observer.tensor
    # self.dict = {"observation": np.reshape(self.tensor, (3, 3, 3))}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    if state.data.internal_state is not None:
        self.observer.set_from(state.get_internal_state(), player)
    # raise NotImplementedError

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    return state.information_state_string(player)



###########################################################################################################################
###########################################################################################################################
##
## Define the solver classes
##
#############
class EndgameType(Enum):
    UNSAFE = 0
    SAFE = 1
    FROZENCFR = 2
    PORTFOLIO = 3

class DLSolver:
    def __init__(self, game, player, blueprint=None, model=None, num_strategy_options=None, dl_leaf_value_function=None, game_utils=None):
        self.game = game
        self.game_utils = game_utils
        self.player = player
        self.blueprint = blueprint
        self.model = model
        self.num_strategy_options = num_strategy_options
        self.dl_leaf_value_function = dl_leaf_value_function
        self.data = defaultdict(int)
    def make_subgame_root(self, state, is_state_dl_leaf=None):
        return NotImplemented
    def get_full_game_policy(self):
        """returns a Policy object for the original game"""
        # the general way to do this is to return a Policy object whose action_probabilities() method
        # builds a dl-subgame, solves it, and returns the action probabilities from doing so.
        return NotImplemented
    def get_trunk_policy(self, trunk_root, is_state_dl_leaf_fn, cfr_parameters=CFRParameters(CFRType.VANILLA, 1000)):
        """returns a policy for the subgame.  just glue it onto the blueprint yourself!"""
        subgame = self.make_subgame_root(
            trunk_root,
            is_state_dl_leaf=is_state_dl_leaf_fn,
        )
        solver = solve_game(subgame, cfr_parameters)
        self.data['infostates'] = solver.num_infostates_expanded
        return get_tabular_policy(solver)
        
    def solve_trunk_and_get_endgames(self, trunk_root, is_state_leaf_fn, endgame_type=EndgameType.UNSAFE, cfr_parameters=CFRParameters(CFRType.VANILLA, 1000)):
        """returns namedtuple(trunk_policy: AugmentedSubgame, endgames: [AugmentedSubgame])"""
        # TODO: ideally this shouldn't modify self.models and self.blueprint
        ReturnWrapper = namedtuple("ReturnWrapper", ['trunk_policy', 'endgames'])
        # subgame = self.make_subgame_root(trunk_root, is_state_dl_leaf=is_state_leaf_fn)
        trunk_policy = self.get_trunk_policy(trunk_root, is_state_leaf_fn, cfr_parameters)
        # set blueprint to result of training
        output = copy.copy(self.blueprint).to_tabular()
        paste_subpolicy_onto_policy(output, trunk_policy)
        info('rewriting self.blueprint')
        self.blueprint = output
        info('rewriting self.models')
        self.models = [GenerativeModelExact(self.game, self.game_utils, self.blueprint, i) for i in [0,1]]
        # collect all endgame roots
        all_endgame_roots = []
        all_endgame_roots_public_info = set()
        if endgame_type in [EndgameType.SAFE, EndgameType.UNSAFE]:
            def dfs(state):
                if is_state_leaf_fn(state):
                    if self.game_utils.get_public_tensor_from_state(state).tobytes() not in all_endgame_roots_public_info:
                        all_endgame_roots_public_info.add(self.game_utils.get_public_tensor_from_state(state).tobytes())
                        all_endgame_roots.append(state)
                    return
                for a in state.legal_actions():
                    dfs(state.child(a))
        elif endgame_type in [EndgameType.FROZENCFR]:
            def dfs(private_chances, state, reach_probability):
                if is_state_leaf_fn(state):
                    # since we know the algorithms operating on the subgame won't mutate the states, we can just give states here instead of
                    # private infos which we need to synthesize into states later.  I think.
                    private_chances.append((state, reach_probability))
                    return
                if state.is_chance_node():
                    probs = state.chance_outcomes()
                elif state.is_player_node():
                    probs = self.blueprint.action_probabilities(state).items()
                else:
                    # terminal node
                    return
                for a,p in probs:
                    dfs(private_chances, state.child(a), p*reach_probability)
        # make endgames
        endgames = []
        if endgame_type == EndgameType.SAFE:
            dfs(self.game.new_initial_state())
            debug('number of endgames:', len(all_endgame_roots))
            subgame_resolver = subgame_solving.SafeSubgameSolver(self.game, self.player)
            subgame_resolver.crawl_game(self.game.new_initial_state(), self.blueprint)
            for endgame in all_endgame_roots:
                endgame_root, endgame_subgame = subgame_resolver.make_augmented_subgame_root(endgame)
                endgames.append(endgame_subgame)
        elif endgame_type == EndgameType.UNSAFE:
            dfs(self.game.new_initial_state())
            debug('number of endgames:', len(all_endgame_roots))
            for endgame in all_endgame_roots:
                endgame_subgame = self.make_subgame_root(endgame,
                                                               is_state_dl_leaf=lambda s: False)
                endgames.append(endgame_subgame)
        elif endgame_type == EndgameType.FROZENCFR:
            # TODO(code quality): move this out into its own function
            # TODO(function): this doesn't seem to work properly
            print("this doesn't seem to work properly")
            dfs([], self.game.new_initial_state(), 1)
            denom = sum(x[1] for x in private_chances)
            private_chances = [(s,p/denom) for s,p in private_chances]
            get_state = lambda cs: private_chances[cs[0]][0].clone() # this clone seems like it shouldn't be necessary, but it is.
            endgame = AugmentedSubgame(self.game, [private_chances], [0,0], lambda s: False, get_state, lambda x: None)
            # return endgame
            endgames.append(endgame)
        else:
            raise ValueError(f'unrecognized endgame solving type: {endgame_type}')
        return ReturnWrapper(trunk_policy, endgames)

    def get_subgames_from_subgame_parameters(self, subgame_parameters):
        subgame_parameters.sort(key=lambda sp: sp[0].move_number())
        subgames = []
        for root_state, is_state_dl_leaf in subgame_parameters:
            subgame = self.make_subgame_root(root_state, is_state_dl_leaf)
            subgames.append(subgame)
        return subgames

    def get_full_game_policy_with_given_subgames(self, subgames, cfr_parameters):
        """subgame_parameters = [(root state, is_leaf_fn), ...]
        EAGERLY compute the policy here"""
        if self.blueprint is not None:
            policy_to_return = copy.copy(self.blueprint)
        else:
            policy_to_return = policy.TabularPolicy(self.game)
        for subgame in subgames:
            solver = solve_game(subgame, cfr_parameters)
            # TODO(speed): when subgames overlap, we needn't copy the entire subgames onto the output policy, just the relevant parts.
            # e.g. if we have a subgame rooted at each action, but the subgames have depth of 5, we only need to copy the policy at the root states
            # not all b^5 states (b=branching factor)
            paste_subpolicy_onto_policy(policy_to_return, solver.average_policy())
        return policy_to_return
            
    def get_nested_full_game_policy(self, players, depth, cfr_parameters):
        # EAGERLY compute the policy resulting from doing search
        subgame_roots = self._get_all_subgame_roots(players)
        # TODO(correctness): assuming here that all states in a public infoset have the same move number (not true with e.g. PTTT where you don't know how many failed moves your opponent made)
        # print(len(subgame_roots))
        # for sgr in subgame_roots:
        #     print(sgr)
        subgame_parameters = []
        def make_is_state_leaf_fn(d):
            return lambda s: s.move_number() >= d
        for state in subgame_roots:
            subgame_parameters.append((state, make_is_state_leaf_fn(state.move_number()+depth)))
        subgames = self.get_subgames_from_subgame_parameters(subgame_parameters)
        subgames = [subgame for subgame in subgames if (get_max_game_depth(subgame) >= depth+2)]
        print(len(subgames))
        return self.get_full_game_policy_with_given_subgames(subgames, cfr_parameters)

    def _get_all_subgame_roots(self, players):
        # collect all subgame roots
        all_subgame_roots = []
        all_subgame_roots_public_info = set()
        # if endgame_type in [EndgameType.SAFE, EndgameType.UNSAFE]:
        def dfs(state):
            # if is_state_leaf_fn(state):
            if state.current_player() in players:
                if self.game_utils.get_public_tensor_from_state(state).tobytes() not in all_subgame_roots_public_info:
                    all_subgame_roots_public_info.add(self.game_utils.get_public_tensor_from_state(state).tobytes())
                    all_subgame_roots.append(state)
            for a in state.legal_actions():
                dfs(state.child(a))
        dfs(self.game.new_initial_state())  
        return all_subgame_roots


class UnrestrictedDLSolver(DLSolver):
    def make_subgame_root(self, state, is_state_dl_leaf=None):
        pass

class MatrixValuedDLSolver(DLSolver):
    def __init__(self, game, game_utils, player, blueprint, models, dl_leaf_value_function, num_strategy_options):
        self.game = game
        self.game_utils = game_utils
        self.player = player
        self.blueprint = blueprint
        self.models = models
        self.num_strategy_options = num_strategy_options
        self.dl_leaf_value_function = dl_leaf_value_function
        self.data = defaultdict(int)
    def make_subgame_root(self, state, is_state_dl_leaf, hidden_states_per_player=(6,6)): # TODO: not 6.
        if state.current_player() != self.player:
            pass
            # print("player is {} but state is not a {} choice node, it's a {} node.".format(self.player, self.player, state.current_player()))
            # raise ValueError("player is {} but state is not a {} choice node, it's a {} node.".format(self.player, self.player, state.current_player()))
        assert len(self.models) == 2, 'make_subgame_root assumes that there are two players (and two generative models)'
        # public_tensor = self.game_utils.get_public_tensor_from_state(state)
        sampled_privates = [self.models[i].sample_privates(
            state,
            hidden_states_per_player[i],
            self.game_utils.get_private_tensors_from_state(state)[i] if i==self.player else None
        ) for i in range(2)]
        AugmentedSubgame = make_augmented_subgame(
            actual_game=self.game,
            private_chances = sampled_privates,
            num_strategy_options = self.num_strategy_options,
            is_state_dl_leaf = is_state_dl_leaf,
            private_chance_fn = lambda cs: self.game_utils.synthesize_state(state,[sampled_privates[i][cs[i]][0] for i in [0,1]]),
            strategy_value_fn = self.dl_leaf_value_function,
        )
        return AugmentedSubgame
        # return AugmentedSubgame(
        #     {
        #         'actual_game': self.game,
        #         'private_chances': sampled_privates,
        #         'num_strategy_options': self.num_strategy_options,
        #         'is_state_dl_leaf': is_state_dl_leaf,
        #         'get_state': 
        #     }
        # )
    def _get_full_game_policy_leduc(self, cfr_parameters: CFRParameters, endgame_solving=EndgameType.UNSAFE, use_full_game_instead_of_trunk=False):
        # TODO: DELETE THIS or at least move it into a leduc-only class.  I only copied this here to test quickly rn
        """mutates self.blueprint and self.models"""
        if use_full_game_instead_of_trunk:
            print('no-op leaf function!!')
            is_state_dl_leaf_fn = lambda s: False
        else:
            print('using is_chance_node() leaf function')
            is_state_dl_leaf_fn = lambda s: s.is_chance_node()

        trunk_subgame = self.make_subgame_root(self.game.new_initial_state().child(0).child(1),
                                                           is_state_dl_leaf=is_state_dl_leaf_fn)
        # train trunk
        # we don't necessarily have to make it tabular here but it's easier to work with and leduc is small enough
        solver = solve_game(trunk_subgame, cfr_parameters)
        tabular_policy =  get_tabular_policy(solver)
        
        # set blueprint to result of training
        output = copy.copy(self.blueprint).to_tabular()
        paste_subpolicy_onto_policy(output, tabular_policy)
        self.blueprint = output
        info('rewriting self.models')
        self.models = [GenerativeModelExact(self.game, self.game_utils, self.blueprint, i) for i in [0,1]]
        
        # collect all endgame roots
        all_endgame_roots = []
        all_endgame_roots_public_info = set()
        def dfs(state, chance_seen):
            if chance_seen == 3:
                if self.game_utils.get_public_tensor_from_state(state).tobytes() not in all_endgame_roots_public_info:
                    all_endgame_roots_public_info.add(self.game_utils.get_public_tensor_from_state(state).tobytes())
                    all_endgame_roots.append(state)
                return
            if state.is_chance_node():
                chance_seen += 1
            for a in state.legal_actions():
                dfs(state.child(a), chance_seen)

        # make endgames
        subgames = []
        if endgame_solving == EndgameType.SAFE:
            dfs(self.game.new_initial_state(), 0)
            debug('number of endgames:', len(all_endgame_roots))

            subgame_resolver = subgame_solving.SafeSubgameSolver(self.game, self.player)
            subgame_resolver.crawl_game(self.game.new_initial_state(), self.blueprint)
            for endgame in all_endgame_roots:
                endgame_root, endgame_subgame = subgame_resolver.make_augmented_subgame_root(endgame)
                subgames.append(endgame_subgame)
        elif endgame_solving == EndgameType.UNSAFE:
            dfs(self.game.new_initial_state(), 0)
            debug('number of endgames:', len(all_endgame_roots))

            for endgame in all_endgame_roots:
                endgame_subgame = self.make_subgame_root(endgame,
                                                               is_state_dl_leaf=lambda s: False)
                subgames.append(endgame_subgame)
        elif endgame_solving == EndgameType.FROZENCFR:
            # TODO(code quality): move this out into its own function
            # TODO(function): this doesn't seem to work properly
            print("this doesn't seem to work properly")
            private_chances = []
            # def dfs(state, reach_prob):
            #     if serialize(self.game_utils.get_public_tensor_from_state(state)) == serialize(public_tensor) and public.move_number() == state.move_number():
            #         # usually need to serialize here, but we cheat and private_tensors for kuhn just returns a scalar
            #         ans[serialize(self.game_utils.get_private_tensors_from_state(state)[self.player])] += reach_prob
            #     if state.is_chance_node():
            #         probs = state.chance_outcomes()
            #     elif state.is_player_node():
            #         probs = self.policy.action_probabilities(state).items()
            #     else:
            #         # terminal node
            #         return
            #     for a, p in probs:
            #         dfs(state.child(a), p*reach_prob)
            def dfs(state, reach_probability, chance_seen):
                if chance_seen == 3:
                    # since we know the algorithms operating on the subgame won't mutate the states, we can just give states here instead of
                    # private infos which we need to synthesize into states later.  I think.
                    private_chances.append((state, reach_probability))
                    return
                if state.is_chance_node():
                    chance_seen += 1
                    probs = state.chance_outcomes()
                elif state.is_player_node():
                    probs = self.blueprint.action_probabilities(state).items()
                else:
                    # terminal node
                    return
                for a,p in probs:
                    dfs(state.child(a), p*reach_probability, chance_seen)
            dfs(self.game.new_initial_state(), 1, 0)
            denom = sum(x[1] for x in private_chances)
            private_chances = [(s,p/denom) for s,p in private_chances]
            get_state = lambda cs: private_chances[cs[0]][0].clone() # this clone seems like it shouldn't be necessary, but it is.
            endgame = AugmentedSubgame(self.game, [private_chances], [0,0], lambda s: False, get_state, lambda x: None)
            # return endgame
            subgames.append(endgame)
        else:
            raise ValueError(f'unrecognized endgame solving type: {endgame_solving}')
            
        # solve each subgame and write to the output policy
        # output_policy = policy.tabular_policy_from_callable(self.game, self.blueprint)
        output_policy = copy.copy(self.blueprint).to_tabular()
        for i, subgame in enumerate(subgames):
            info('subgame {} out of {}'.format(i+1, len(subgames)))
            solver = solve_game(subgame, cfr_parameters)
            tabular_policy = get_tabular_policy(solver)
            print(f'exploitability of subgame: {exploitability.nash_conv(subgame, tabular_policy)}')
            paste_subpolicy_onto_policy(output_policy, tabular_policy)
        return output_policy

    def _get_full_game_policy_kuhn(self, cfr_parameters, safe_endgame_solving=True, use_full_game_instead_of_trunk=False):
        # TODO: delete use_full_game_instead_of_trunk, it's just for testing
        # TODO: DELETE THIS or at least move it into a kuhn-only class.  I only copied this here to test quickly rn
        """mutates self.blueprint and self.models"""
        if use_full_game_instead_of_trunk:
            print('no-op leaf function!!')
            is_state_dl_leaf_fn = lambda s: False
        else:
            is_state_dl_leaf_fn = lambda s: s.move_number()>=3

        trunk_subgame = self.make_subgame_root(self.game.new_initial_state().child(0).child(1),
                                                           is_state_dl_leaf=is_state_dl_leaf_fn)
        # train trunk
        solver = solve_game(trunk_subgame, cfr_parameters)
        tabular_policy = get_tabular_policy(solver)
        
        # set blueprint to result of training
        output = copy.copy(self.blueprint).to_tabular()
        paste_subpolicy_onto_policy(output, tabular_policy)
        self.blueprint = output
        info(f'game value of the blueprint: {-game_score_of_best_response(self.game, self.blueprint, 0)=}')
        info('rewriting self.models')
        self.models = [GenerativeModelExact(self.game, self.game_utils, self.blueprint, i) for i in [0,1]]
        
        # collect all endgame roots
        all_endgame_roots = []
        all_endgame_roots_public_info = set()
        def dfs(state):
            if state.move_number() >= 3 and not state.is_terminal():
                if self.game_utils.get_public_tensor_from_state(state).tobytes() not in all_endgame_roots_public_info:
                    all_endgame_roots_public_info.add(self.game_utils.get_public_tensor_from_state(state).tobytes())
                    all_endgame_roots.append(state)
                return
            for a in state.legal_actions():
                dfs(state.child(a))
        dfs(self.game.new_initial_state())
        debug('number of endgames:', len(all_endgame_roots))
        
        # make endgames
        subgames = []
        if safe_endgame_solving:
            subgame_resolver = subgame_solving.SafeSubgameSolver(self.game, self.player)
            subgame_resolver.crawl_game(self.game.new_initial_state(), self.blueprint)
            for endgame in all_endgame_roots:
                endgame_root, endgame_subgame = subgame_resolver.make_augmented_subgame_root(endgame)
                subgames.append(endgame_subgame)
        else:
            for endgame in all_endgame_roots:
                endgame_subgame = self.make_subgame_root(endgame,
                                                               is_state_dl_leaf=lambda s: False)
                subgames.append(endgame_subgame)
            
        # solve each subgame and write to the output policy
        # output_policy = policy.tabular_policy_from_callable(self.game, self.blueprint)
        output_policy = copy.copy(self.blueprint).to_tabular()
        for i, subgame in enumerate(subgames):
            info('subgame {} out of {}'.format(i+1, len(subgames)))
            solver = solve_game(subgame, cfr_parameters)
            tabular_policy = get_tabular_policy(solver)
            paste_subpolicy_onto_policy(output_policy, tabular_policy)
        return output_policy

#################################################################
###
### Game-specific implementations of DL Solvers
###
###########

class LeducDLSolver(MatrixValuedDLSolver):
    pass
    # def get_trunk_and_endgames(self):
    #     """returns (trunk: AugmentedSubgame, endgames: [AugmentedSubgame])"""
    # def get_trunk_and_endgames(self, trunk_root, is_state_leaf_fn, endgame_type=EndgameType.UNSAFE):
    #     """returns (trunk: AugmentedSubgame, endgames: [AugmentedSubgame])"""
    #     subgame = self.make_subgame_root(trunk_root, is_state_dl_leaf=is_state_leaf_fn)
    #     if endgame_type == EndgameType.SAFE:
    #         raise ValueError("Probably don't use Safe subgame solving without solving the trunk first")


##########################
class SynthesisError(ValueError):
    pass

class GameUtils(ABC):
    def __init__(self):
        self.public_observer = self.get_public_observer()
        self.privates_observer = self.get_privates_observer()
    def __str__(self):
        return f'GameUtils for {self.game}'
    @classmethod
    def serialize(cls, data):
        return tuple(data)
    @classmethod
    def deserialize(cls, data):
        return np.array(data)
    def get_public_observer(self):
        raise NotImplementedError
    def get_privates_observer(self):
        raise NotImplementedError
    def get_public_tensor_from_state(self, state):
        self.public_observer.set_from(state, player=0) # player doesn't (shouldn't) matter since we only want public info
        return self.public_observer.tensor.copy()
    def get_private_tensors_from_state(self, state):
        self.privates_observer.set_from(state, player=0) # player doesn't (shouldn't) matter since we want both players info
        return self.privates_observer.tensor.copy()
    def synthesize_state(self, state, hidden_infos):
        """hidden_infos: [synthesized private tensor, ]"""
        raise NotImplementedError

# universal poker:
#   // Layout of observation:
#   //   my player number: num_players bits
#   //   my cards: Initial deck size bits (1 means you have the card), i.e.
#   //             MaxChanceOutcomes() = NumSuits * NumRanks
#   //   public cards: Same as above, but for the public cards.
#   //   NumRounds() round sequence: (max round seq length)*2 bits
class UniversalPokerGameUtils(GameUtils):
    def __init__(self, game, deck_size, num_hole_cards):
        """deck_size = num of cards in total.  num_hole_cards = number of hole cards per player"""
        self.deck_size = deck_size
        self.num_hole_cards = num_hole_cards
        self.game = game
        super().__init__()
    @classmethod
    def serialize(cls, data):
        return tuple(data)
    @classmethod
    def deserialize(cls, data):
        return np.array(data)
    def get_public_observer(self):
        def remove_player_private_info(tensor):
            return tensor[2+self.deck_size:]
        class Observer:
            def __init__(self):
                self.tensor = None
            def set_from(self, state, player):
                try:
                    tensor = state.information_state_tensor(0) # we say player=0 here but it shouldn't matter either way
                except Exception as e:
                    print(state)
                    raise e
                tensor = remove_player_private_info(tensor)
                self.tensor = np.array(tensor)
        return Observer()
    def get_privates_observer(self):
        def get_only_private_info(tensor):
            return tensor[2:self.deck_size+2]
        class Observer:
            def __init__(self):
                self.tensor = None
            def set_from(self, state, player):
                self.tensor = np.stack([get_only_private_info(state.information_state_tensor(0)),
                                        get_only_private_info(state.information_state_tensor(1))])
        return Observer()
    def synthesize_state(self, state, hidden_infos):
        chance_actions = []
        for i in range(2):
            cards = np.where(hidden_infos[i])[0]
            # assert len(cards) == self.num_hole_cards
            # if len(cards) != self.num_hole_cards:
            #     print(hidden_infos)
            #     print(state)
            for card in cards:
                chance_actions.append(card)
            # {i:np.argmax(hidden_infos[i]) for i in range(self.num_hole_cards * 2)}
        s = self.game.new_initial_state()
        for i, action in enumerate(state.full_history()):
            if i < len(chance_actions):
                action_to_apply = chance_actions[i]
            else:
                action_to_apply = action.action
            if action_to_apply not in s.legal_actions():
                raise SynthesisError()
            else:
                s.apply_action(action_to_apply)
        return s

class LeducGameUtils(GameUtils):
    def __init__(self):
        self.game = pyspiel.load_game("leduc_poker")
        super().__init__()
    def get_public_observer(self):
        return make_observation(self.game, pyspiel.IIGObservationType(
            perfect_recall=True,
            public_info=True,
            private_info=pyspiel.PrivateInfoType.NONE))
    def get_privates_observer(self):
        def get_only_private_info(tensor):
            return tensor[2:8]
        class Observer:
            def __init__(self):
                self.tensor = None
            def set_from(self, state, player):
                self.tensor = np.stack([get_only_private_info(state.information_state_tensor(0)),
                                        get_only_private_info(state.information_state_tensor(1))])
        return Observer()
    def synthesize_state(self, state, hidden_infos):
        chance_actions = {i:np.argmax(hidden_infos[i]) for i in range(2)}
        s = self.game.new_initial_state()
        for i, action in enumerate(state.full_history()):
            if i in chance_actions:
                action_to_apply = chance_actions[i]
            else:
                action_to_apply = action.action
            if action_to_apply not in s.legal_actions():
                raise SynthesisError()
            else:
                s.apply_action(action_to_apply)
        return s

class KuhnGameUtils(GameUtils):
    def __init__(self, game=None):
        if game is None:
            self.game = pyspiel.load_game("kuhn_poker")
        else:
            self.game = game
        self.public_observer = self.get_public_observer()
    # @classmethod
    # def serialize(cls, data):
    #     return data
    # @classmethod
    # def deserialize(cls, data):
    #     return data
    def get_public_observer(self):
        return make_observation(self.game, pyspiel.IIGObservationType(
            perfect_recall=True,
            public_info=True,
            private_info=pyspiel.PrivateInfoType.NONE))
    def get_private_tensors_from_state(self, state):
        history = list(state.history()) # copy for safety
        while len(history) < 2:
            history.append(-1)
        return tuple([x] for x in history[:2])
    def synthesize_state(self, state, hidden_infos):
        chance_actions = {i:hidden_infos[i][0] for i in range(2)}
        # chance_actions = {i:np.argmax(hidden_infos[i]) for i in range(2)}
        s = self.game.new_initial_state()
        for i, action in enumerate(state.full_history()):
            if i in chance_actions:
                action_to_apply = chance_actions[i]
            else:
                action_to_apply = action.action
            if action_to_apply not in s.legal_actions():
                raise SynthesisError()
            else:
                s.apply_action(action_to_apply)
        return s

class RootGameUtils(GameUtils):
    """if just building a trunk, we can just start at the root of the game instead of synthesizing arbitrary states based on inferred private info"""
    def __init__(self, game):
        self.game = game
    @classmethod
    def serialize(cls, data):
        return tuple(data)
    @classmethod
    def deserialize(cls, data):
        return np.array(data)
    def get_public_tensor_from_state(self, state):
        return 'root'
    def get_private_tensors_from_state(self, state):
        return 'root'
    def synthesize_state(self, state, hidden_infos):
        return self.game.new_initial_state()

class DoubleGuessingGameUtils(GameUtils):
    def __init__(self, game):
        self.game = game
        self.public_observer = self.get_public_observer()
        self.private_observer = self.get_private_observer()
    @classmethod
    def serialize(cls, data):
        return tuple(data)
    @classmethod
    def deserialize(cls, data):
        return np.array(data)
    def get_public_observer(self):
        return make_observation(self.game, pyspiel.IIGObservationType(
            perfect_recall=True,
            public_info=True,
            private_info=pyspiel.PrivateInfoType.NONE))
    def get_private_observer(self):
        return make_observation(self.game, pyspiel.IIGObservationType(
            perfect_recall=True,
            public_info=False,
            private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER))
    def get_private_tensors_from_state(self, state):
        self.private_observer.set_from(state, player=0)
        t1 = self.private_observer.tensor.copy()
        self.private_observer.set_from(state, player=1)
        t2 = self.private_observer.tensor.copy()
        return np.stack([t1, t2])
    def synthesize_state(self, state, hidden_infos):
        N = self.game.N
        private_guess_1 = np.where(hidden_infos[0][N+2:N+N+2])[0]
        private_guess_2 = np.where(hidden_infos[1][N+2:N+N+2])[0]
        private_card_2 = np.where(hidden_infos[1][2:N+2])[0]
        s = self.game.new_initial_state()
        if len(private_card_2) > 0:
            s.apply_action(private_card_2[0])
            if len(private_guess_1) > 0:
                s.apply_action(private_guess_1[0])
                if len(private_guess_2) > 0:
                    s.apply_action(private_guess_2[0])
        return s

# def get_public_observer_liars_dice():
#     def remove_player_private_info(tensor):
#         return tensor[2+NUM_DICE*DICE_SIDES:]
#     class LiarsDiceObserver:
#         def __init__(self):
#             self.tensor = None
#         def set_from(self, state, player):
#             tensor = state.information_state_tensor()
#             tensor = remove_player_private_info(tensor)
#             self.tensor = tensor
#     return LiarsDiceObserver()

# def get_privates_observer_liars_dice():
#     def get_only_private_info(tensor):
#         return tensor[2:2+NUM_DICE*DICE_SIDES]
#     class LiarsDiceObserver:
#         def __init__(self):
#             self.tensor = None
#         def set_from(self, state, player):
#             tensor = np.concatenate([get_only_private_info(state.information_state_tensor(0)),
#                                          get_only_private_info(state.information_state_tensor(1))])
#             self.tensor = np.reshape(tensor, newshape=(NUM_DICE*2, DICE_SIDES))
#     return LiarsDiceObserver()


#########################################################################################################################
#########################################################################################################################
###
### Define the lesser ingredients
###
########################

# TODO: Move some of this to common?

# TODO: moved this to game_utils, so delete it here once all ported over
def serialize(data):
    return tuple(data)
def deserialize(data):
    return np.array(data)

def make_empirical_counters_from_traverser(traverser):
    counters = [defaultdict(lambda: defaultdict(int)) for player in traverser.y_datas]
    for player, y_data in enumerate(traverser.y_datas):
        for i, row in enumerate(traverser.x_data):
            target = y_data[i]
            counters[player][serialize(row)][serialize(target)] += 1
    return counters

class GenerativeModel:
    def __init__(self):
        pass
    def sample_privates(self, public, n):
        """takes in inputs public: x
        returns an array [(s0,p(s0|x)),...,(sn,p(sn|x))]"""
        return NotImplemented

class GenerativeModelRoot:
    def __init__(self, game):
        self.game = game
    def sample_privates(self, public, n, must_include=None):
        return [(self.game.new_initial_state(),1)]

class GenerativeModelDummy:
    # just a uniform random one
    def __init__(self, pop):
        self.pop = pop
    def sample_privates(self, public, n, must_include=None):
        print('ignoring n.')
        return [(p,1/len(self.pop)) for p in self.pop]

class GenerativeModelExact:
    """based on a policy, use Bayes rule to explicitly figure out what the probabilities of a given state are"""
    def __init__(self, game, game_utils: GameUtils, policy, player):
        self.game = game
        self.game_utils = game_utils
        self.policy = policy
        self.player = player
        self.cache = defaultdict(lambda : defaultdict(float))
        self._save_all_reach_probs()
    def _save_all_reach_probs(self):
        """traverse the entire game tree and save reach probs for self.player at each state"""
        serialize = self.game_utils.serialize
        def dfs(state, reach_prob):
            public_tensor = serialize(self.game_utils.get_public_tensor_from_state(state))
            # TODO(correctness): checking move_number may fail if two states have the same public states but different move numbers, e.g. variably 1 or 2 opponent hidden actions
            # such as phantom tic tac with "reveal-nothing" obstype
            self.cache[(public_tensor, state.move_number())][serialize(self.game_utils.get_private_tensors_from_state(state)[self.player])] += reach_prob
            # if serialize(self.game_utils.get_public_tensor_from_state(state)) == serialize(public_tensor) and public.move_number() == state.move_number():
            #     ans[serialize(self.game_utils.get_private_tensors_from_state(state)[self.player])] += reach_prob
            # self.cache[serialize(public_tensor)][serialize(self.game_utils.get_private_tensors_from_state(state)[self.player])] += reach_prob
            if state.is_chance_node():
                probs = state.chance_outcomes()
            elif state.is_player_node():
                probs = self.policy.action_probabilities(state).items()
            else:
                # terminal node
                return
            for a, p in probs:
                dfs(state.child(a), p*reach_prob)
        dfs(self.game.new_initial_state(), 1)
    def sample_privates(self, public, n, must_include=None):
        """could make this much more efficient by only exploring all legal_actions during chance nodes,
        and doing exactly the moves from `public` for choice nodes.

        We also need to look at state.move_number to get around this bug: https://github.com/deepmind/open_spiel/issues/548
        (note that even this hack doesn't work if we want to sample privates in chance nodes)
        If we fix the bug, we don't need to take in `player` anymore.
        """
        public_tensor = self.game_utils.get_public_tensor_from_state(public)
        serialize = self.game_utils.serialize
        move_number = public.move_number()
        key = (serialize(public_tensor), move_number)
        if key in self.cache:
            ans = self.cache[key]
        else:
            raise ValueError
        if n == len(ans):
            denom = sum(p for i,p in ans.items())
            ans = [(i,p/denom) for i,p in ans.items()]
        else:
            ans = list(ans.items())
            sampled_indices = np.random.choice(range(len(ans)), n, replace=False, p=[x[1] for x in ans])
            ans = [ans[i] for i in sampled_indices]
            denom = sum(p for _,p in ans)
            ans = [(i,p/denom) for i,p in ans]
        return ans

class GenerativeModelEmpirical(GenerativeModel):
    def __init__(self, game_utils, empirical_counter):
        self.game_utils = game_utils
        self.empirical_counter = empirical_counter
    def sample_privates(self, public, n, must_include=None):
        public_tensor = self.game_utils.get_public_tensor_from_state(public)
        if must_include is not None:
            n -= 1
        ans = []
        evidence = self.empirical_counter[serialize(public_tensor)]
        denom = sum(evidence.values())
        ans = []
        population = []
        probabilities = []
        for target, volume in evidence.items():
            if must_include is not None and target == serialize(must_include):
                ans.append((deserialize(target), volume/denom))
            else:
                population.append(target)
                probabilities.append(volume/denom)
        assert must_include is None or len(ans)>0, f'must_include is {must_include}, not none in {evidence} matched. state is {str(public_tensor)}'
        probsum = sum(probabilities)
        samples = np.random.choice(range(len(population)),n,replace=False, p=[p/probsum for p in probabilities])
        for sample in samples:
            ans.append((deserialize(population[sample]), probabilities[sample]))
        return ans

class GenerativeModelAutoregressive(GenerativeModel):
    pass

#################################################################################################
###
### Utils
###
#########################
    
# TODO(code quality): maybe this technically should be a function that returns a policy (kinda of like in PolicyAggregator) instead of a class that is a policy.
class MixedPolicy(policy.TabularPolicy):
    """logic should be similar to what's in XFP (https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/fictitious_play.py)
    but it's a little opaque and I don't understand it so I'll just implement my own here, heavily based on policy_aggregator (https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/algorithms/policy_aggregator.py)
    Coupled to AugmentedSubgame implementation.
    """
    def __init__(self, game, trunk_policy: policy.Policy, policies):
        trunk = trunk_policy.game
        # assert isinstance(trunk, AugmentedSubgame), f'{type(trunk)} should be {AugmentedSubgame}.  If it is, this is spurious due to importlib.reload() making the class different.  comment this out or make them match to go forth.'
        self.game = game
        self.trunk_policy = trunk_policy
        self.trunk = trunk
        self.policies = policies
        self._epsilon = 1e-40
        super().__init__(self.game)
        
        # write endgame policy based on mixture of strategies at DL-leaves
        self._make()
    
    def _make(self):
        # set all probabilities to 0?
        for key in self.state_lookup:
            probabilities = self.policy_for_key(key)
            self.policy_for_key(key)[:].fill(0)
        # copy trunk_policy to self
        paste_subpolicy_onto_policy(self, self.trunk_policy)
        self._seen = set()
        self._traverse_trunk_recursive(self.trunk.new_initial_state())
        # now normalize
        for key, state_id in self.state_lookup.items():
            probabilities = self.policy_for_key(key)
            if sum(probabilities) == 0:
                self.policy_for_key(key)[:] = self.legal_actions_mask[state_id] / np.sum(self.legal_actions_mask[state_id])
            else:
                self.policy_for_key(key)[:] = probabilities/sum(probabilities)
    
    def _traverse_trunk_recursive(self, state):
        if get_subgame_status(state) == 'PostgameSubgameState':
            # We are at a state that represents a trunk leaf. For each player's infostate, if not explored already,
            # we find the strategies and mixture % over them, and aggregate them and write to the aggregated policy.
            c = state
            weights = []
            for pid in [0,1]:
                weights.append([self.trunk_policy.action_probabilities(c)[k] for k in c.legal_actions()])
                c = c.child(state.legal_actions()[0])
            for pid in [0,1]:
                self._policy_pool = policy_aggregator.PolicyPool(self.policies)
                self._rec_aggregate(pid, state.get_internal_state(), weights)
            return
        for a in state.legal_actions():
            self._traverse_trunk_recursive(state.child(a))

    def _rec_aggregate(self, pid, state, my_reaches):
        """Recursively traverse game tree to compute aggregate policy."""
        if state.is_terminal():
            return
        elif state.is_simultaneous_node():
            raise NotImplementedError
        elif state.is_chance_node():
            # do not factor in opponent reaches
            outcomes, _ = zip(*state.chance_outcomes())
            for i in range(0, len(outcomes)):
                outcome = outcomes[i]
                new_state = state.clone()
                new_state.apply_action(outcome)
                self._rec_aggregate(pid, new_state, my_reaches)
            return
        else:
            turn_player = state.current_player()
            state_key = self._state_key(state, turn_player)
            legal_policies = self._policy_pool(state, turn_player)
            if pid == turn_player:
                # if state_key in self._seen:
                    # return
                self._seen.add(state_key)
                # update the current node
                # will need the observation to query the policies
                # if state_key not in self.state_lookup:
                #     raise ValueError(f"all states should have been built in super().__init__(): {quick_debug_state(state)}")
            used_moves = []
            for k in range(len(legal_policies)):
                used_moves += [a[0] for a in legal_policies[k].items()]
            used_moves = np.unique(used_moves)
            
            for uid in used_moves:
                new_reaches = copy.deepcopy(my_reaches) # !
                if pid == turn_player:
                    self.policy_for_key(state_key)[uid] = 0
                    for i in range(len(legal_policies)):
                        # compute the new reach for each policy for this action
                        new_reaches[turn_player][i] *= legal_policies[i].get(uid, 0)
                        # add reach * prob(a) for this policy to the computed policy
                        # print(f'{state_key},{uid},{legal_policies[i]}: {new_reaches[turn_player][i]}')
                        self.policy_for_key(state_key)[uid] += new_reaches[turn_player][i]
                # recurse
                new_state = state.clone()
                new_state.apply_action(uid)
                self._rec_aggregate(pid, new_state, new_reaches)
