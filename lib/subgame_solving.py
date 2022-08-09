import pyspiel
import numpy as np
from collections import defaultdict
from open_spiel.python.policy import TabularPolicy
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
import random
from open_spiel.python.algorithms import best_response
import copy
from open_spiel.python.algorithms import expected_game_score

try:
    from tqdm import tqdm
except ImportError as e:
    print('{} -- (tqdm is a cosmetic-only progress bar) -- setting `tqdm` to the identity function instead'.format(e))
    tqdm = lambda x: x

from . import utils

DEBUG_LEVEL = 1
def debug(*args):
    if DEBUG_LEVEL <= 0:
        print(*args)

def info(*args):
    if DEBUG_LEVEL <= 1:
        print(*args)

def debug_game(state, policy):
    prefix = ' '*state.move_number()
    if not state.is_chance_node() and not state.is_terminal():
        info = ',info: ' + state.information_state_string()
    else:
        info = ''
    debug(prefix,'his:',state.history_str(), info, 'pays out {}'.format(state.player_return(0)) if state.is_terminal() else '')
    for a in state.legal_actions():
        if state.is_chance_node():
            pass
            debug(prefix, state.history_str(), '->', a, ':', {a:p for a,p in state.chance_outcomes()}[a])
        else:
            debug(prefix, state.history_str(), '->', a, ':', policy.action_probabilities(state)[a])
        debug_game(state.child(a), policy)

def game_score_of_best_response(game, policy, player):
    """If `player` plays according to `policy`, what's the expected value payout of a best-responder?
    Lower is better: If this function returns a lower value for policy A than B, then  A is less exploitable than B!
    """
    best_responder = 1-player
    best_response_policy = best_response.BestResponsePolicy(game=game, policy=policy, player_id=best_responder)
    value = best_response_policy.value(game.new_initial_state())
    return value

def get_policy_value(game, policy):
    return expected_game_score.policy_value(game.new_initial_state(), [policy] * 2)[0]

class AugmentedSubgame(pyspiel.Game):
    def __init__(self, actual_game, chance_outcomes, num_distinct_actions):
        self.game = actual_game
        self._max_chance_outcomes = chance_outcomes
        self._num_distinct_actions = num_distinct_actions
        super().__init__(self, self.get_type(), self.get_info(), self.game.get_parameters())
    def __getattr__(self, attr):
        # hacky hacky hacky hacky
        assert attr != 'new_initial_state'
        return self.game.__getattribute__(attr)
    def get_info(self):
        return pyspiel.GameInfo(
            num_distinct_actions=self.num_distinct_actions(),
            max_chance_outcomes=self.max_chance_outcomes(),
            num_players=self.game.num_players(),
            min_utility=self.game.min_utility(),
            max_utility=self.game.max_utility(),
            utility_sum=self.game.utility_sum(),
            max_game_length=self.game.max_game_length())
    def get_type(self):
        overrides = {
            'provides_information_state_tensor': False,
            'provides_observation_tensor': False,
        }
        all_attrs = ["short_name","long_name","dynamics","chance_mode","information","utility","reward_model","max_num_players","min_num_players","provides_information_state_string","provides_information_state_tensor","provides_observation_string","provides_observation_tensor","parameter_specification"]
        kwargs = {k: (self.game.get_type().__getattribute__(k) if k not in overrides else overrides[k]) for k in all_attrs}
        return pyspiel.GameType(**kwargs)
    def new_initial_state(self):
        return self.root.clone()
    def set_root(self, root):
        self.root = root.clone()
    def max_chance_outcomes(self):
        return max(self._max_chance_outcomes, self.game.max_chance_outcomes())
    def num_distinct_actions(self):
        return max(self._num_distinct_actions, self.game.num_distinct_actions())
    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        class FooBarObserver:
            def __init__(self):
                self.tensor = None
            def set_from(self, state, player):
                pass
            def string_from(self, state, player):
                pass
        return FooBarObserver()
    def get(self):
        return False
    

glob_id = 0
class DummyState:
    def __init__(self, game, children, current_player, payoff=None, probabilities=None):
        self.game = game
        self.cur_player = current_player
        self.children = children
        self.payoff = payoff
        self.probabilities = probabilities
        global glob_id
        self._id = str(glob_id)
        glob_id += 1
    def is_initial_state(self):
        return self.history_str() == self.game.new_initial_state().history_str()
    def get_game(self):
        return self.game
    def history(self):
        """ NOT TRUE """
        return [0]
    def action_to_string(self, arg0, arg1=None):
        """NOT TRUE"""
        """Action -> string. Args either (player, action) or (action)."""
        player = self.current_player() if arg1 is None else arg0
        action = arg0 if arg1 is None else arg1
        return str(action)
    def current_player(self):
        return self.cur_player
    def legal_actions(self, player=None):
        if player is not None and player != self.current_player():
            return []
        return list(range(len(self.children)))
    def legal_actions_mask(self, player):
        if player is not None and player != self.current_player():
            return []
        elif self.is_terminal():
            return []
        else:
            length = self.game.max_chance_outcomes() if self.is_chance_node() else self.game.num_distinct_actions()
            action_mask = [0] * length
            for action in self.legal_actions():
                action_mask[action] = 1
            return action_mask
    def move_number(self):
        # HACK: only using this for my own debug_game() function
        return 0
    def chance_outcomes(self):
#         print(list(zip(self.children, self.probabilities)))
        assert len(self.probabilities) == len(self.children)
        return list(zip(range(len(self.probabilities)), self.probabilities))
    def apply_action(self, action):
        """Applies the specified action to the state."""
        next_state = self.children[action]
        # terrible hack
        self.__class__ = next_state.__class__
        self.__dict__ = copy.copy(next_state.__dict__)
    def child(self, action):
        return self.children[action].clone()
    def is_chance_node(self):
        return self.probabilities is not None
    def is_player_node(self):
        return not self.is_chance_node() and not self.is_terminal()
    def is_terminal(self):
        return len(self.children) == 0
    def returns(self):
        if self.is_terminal():
            # pylint: disable=invalid-unary-operand-type
            return [self.payoff, -self.payoff]
        return [0.0, 0.0]
    def player_return(self, player):
        return self.returns()[player]
    def information_state_string(self, pl=None):
        if pl is None:
            pl = self.current_player()
        return 'dummy' + ','.join(x.information_state_string(pl) for x in self.children)
    def history_str(self):
        hs = '138334' + ','.join(x.history_str() for x in self.children) + self._id
        return hs
    def clone(self):
        return copy.deepcopy(self)
    def is_simultaneous_node(self):
        return False
    def __str__(self):
        return self.history_str()

def erase_subgame_policy_recursive(state, policy, trunk_depth):
    if state.move_number() >= trunk_depth and state.is_player_node():
        x = 1/len(state.legal_actions())
        for action in state.legal_actions():
            policy.policy_for_key(state.information_state_string())[action] = x
    for action in state.legal_actions():
        erase_subgame_policy_recursive(state.child(action), policy, trunk_depth)

# trunk_policy = copy.copy(cfr_solver.average_policy())
# erase_subgame_policy_recursive(game.new_initial_state(), trunk_policy)
# seen = set()
# train_all_subgames_recursive(game.new_initial_state(), trunk_policy, seen)
class SafeSubgameSolver:
    def __init__(self, game, player):
        self.game = game
        self.player = player
        # self.trunk_depth = trunk_depth
        # self.blueprint_policy = blueprint_policy   

    def resolve_policy(self, trunk_policy, player, trunk_depth, subgame_cfr_iterations=100):
        seen = set()
        self.train_all_subgames_recursive(self.game.new_initial_state(), trunk_policy, seen, subgame_cfr_iterations, trunk_depth)
        return trunk_policy

    def train_all_subgames_recursive(self, state, combined_trunk_subgame_policy, seen, subgame_cfr_iterations, trunk_depth):
        """this method solves subgames, and writes the subgame policies to combined_trunk_subgame_policy.
        (combined_trunk_subgame_policy starts off as just the trunk policy)"""
        if state.move_number() >= trunk_depth and state.is_player_node():
            if state.history_str() in seen:
                return
            debug('----the subgame rooted at:', state.history_str(), 'and equivalent states')
            roots = self.get_all_equivalent_states(state, self.player)
            for r in roots:
                seen.add(r[1].history_str())
            augmented_subgame_root, augmented_subgame = self.make_augmented_subgame_root(state)
            subgame_solver = cfr.CFRSolver(augmented_subgame)
            for i in tqdm(range(subgame_cfr_iterations)):
                subgame_solver.evaluate_and_update_policy()
                # if i%10 == 0:
                #     conv = exploitability.nash_conv(augmented_subgame, subgame_solver.average_policy())/2
                #     debug('iteration {}: exploitable for: {}'.format(i, conv))
            conv = exploitability.nash_conv(augmented_subgame, subgame_solver.average_policy())/2
            debug_game(augmented_subgame_root, subgame_solver.average_policy())
            print("After {} iterations, exploitability on augmented subgame: {}".format(i+1, conv))
            debug("value of subgame: {}".format(get_policy_value(augmented_subgame, subgame_solver.average_policy())))
            for s in subgame_solver.average_policy().states:
                if s.information_state_string() in combined_trunk_subgame_policy.state_lookup:
                    for action, value in enumerate(subgame_solver.average_policy().policy_for_key(s.information_state_string())):
                        debug('writing to combined strategy')
                        debug(' at state {}, action {} with probability {}'.format(s.information_state_string(), action, value))
                        combined_trunk_subgame_policy.policy_for_key(s.information_state_string())[action] = value
        else:
            for action in state.legal_actions():
                self.train_all_subgames_recursive(state.child(action), combined_trunk_subgame_policy, seen, subgame_cfr_iterations, trunk_depth)
    def get_CBV(self, infostate, player):
        # for a state which is not the solving player's decision node
        # TODO: make this a method on BestResponsePolicy and submit pull request?
        br = self.best_response_policy
        value = 0
        debug('getting CBV for {} (player {})'.format(infostate, player))
        reach_prob_sum = 0
        for state in self.infostate_to_states[(player, infostate)]:
            # weight values for how likely it is to reach the state if player plays to get there
            debug('{}, CF_P: {}, BRV: {}'.format(state.history_str(), self.history_str_to_reach_probabilities[state.history_str()][1-player], br.value(state)))
            value += self.history_str_to_reach_probabilities[state.history_str()][1-player] * br.value(state)
            reach_prob_sum += self.history_str_to_reach_probabilities[state.history_str()][1-player]
        return value/reach_prob_sum
    def make_augmented_subgame_root(self, root):
        roots = self.get_all_equivalent_states(root, self.player)
        prob_sum = sum(x for x,_ in roots)
        debug('sum:', prob_sum)
        debug(str(root))
        other_player = 1-self.player
        root_parents = []
        augmented_subgame = AugmentedSubgame(self.game, len(roots), 3)
        for probability, root in roots:
            br_value = self.get_CBV(root.information_state_string(other_player), other_player)
            debug('best response value at {} ({}): {}'.format(root.information_state_string(other_player), root.history_str(), br_value))
            if other_player == 1:
                br_value *= -1
            alternative_payoff = DummyState(self.game, children=[], current_player=pyspiel.PlayerId.TERMINAL, payoff=br_value)
            root_parent = DummyState(self.game, children=[alternative_payoff, root], current_player=other_player)
            root_parents.append(root_parent)
        augmented_subgame_root = DummyState(self.game, children=root_parents, current_player=-1, probabilities=[x[0]/prob_sum for x in roots])
        augmented_subgame.set_root(augmented_subgame_root)
        return augmented_subgame_root, augmented_subgame
    def crawl_game(self, state, policy):
        self.best_response_policy = best_response.BestResponsePolicy(self.game,
          player_id=1-self.player,
          policy=policy,
        )
        self.infostate_to_states = defaultdict(list)
        self.history_str_to_reach_probabilities = dict()
        self.crawl_game_dfs(self.infostate_to_states, self.history_str_to_reach_probabilities, state, np.array([1, 1]), policy)
        
    def crawl_game_dfs(self, infostate_to_states, history_str_to_reach_probabilities, state, reach_probabilities, policy):
        """reach_probabilities = [x,y] means a prob of x of getting here if player TWO plays to get here and y prob if p ONE tries to get here"""
        history_str_to_reach_probabilities[state.history_str()] = reach_probabilities
        if state.is_terminal():
            return
        if state.current_player() >= 0:
            infostate_to_states[(0,state.information_state_string(0))].append(state)
            infostate_to_states[(1,state.information_state_string(1))].append(state)
        legal_actions = state.legal_actions()
        for action in legal_actions:
            new_reach_probabilities = np.array(reach_probabilities)
            if state.is_player_node():
                new_reach_probabilities[state.current_player()] *= policy.action_probabilities(state)[action]
            elif state.is_chance_node():
                chance_outcomes = {a:p for a,p in state.chance_outcomes()}
                new_reach_probabilities = new_reach_probabilities * chance_outcomes[action]
            self.crawl_game_dfs(infostate_to_states, history_str_to_reach_probabilities, state.child(action), new_reach_probabilities, policy)
            
    def get_all_equivalent_states(self, state, solving_player):
        states = set()
        histories = set()
        to_explore = [state]
        while len(to_explore) > 0:
            ex = to_explore.pop()
            reach_probabilities = self.history_str_to_reach_probabilities[ex.history_str()]
            states.add((reach_probabilities[solving_player], ex))# <------- should this be current_player? or should it be other_player?
            histories.add(ex.history_str())
            # union-find preprocess instead of doing this uberslow search here if you want to do this on bigger games
            for infostate in [(player, ex.information_state_string(player)) for player in (0,1)]:
                for s in self.infostate_to_states[infostate]:
                    if s.history_str() not in histories:
                        to_explore.append(s)
        return states

def get_policy_from_resolving(game, subgame_cfr_iterations=100, trunk_iterations=500, player=0, trunk_depth=3):
    cfr_solver = cfr.CFRSolver(game)
    for i in tqdm(range(trunk_iterations)):
        cfr_solver.evaluate_and_update_policy()
    conv = exploitability.nash_conv(game, cfr_solver.average_policy(),False)
    print("Original full strategy from {} iterations of CFR: exploitable for {}".format(i+1, conv.player_improvements))
    print("game value of blueprint:", game_score_of_best_response(game, cfr_solver.average_policy(), player))
    trunk_policy = copy.copy(cfr_solver.average_policy())
    erase_subgame_policy_recursive(game.new_initial_state(), trunk_policy, trunk_depth)
    conv = exploitability.nash_conv(game, trunk_policy, False)
    print("Just trunk strategy: exploitable for {}".format(conv.player_improvements))

    subgame_resolver = SafeSubgameSolver(game, player)
    subgame_resolver.crawl_game(game.new_initial_state(), cfr_solver.average_policy())
    new_policy = subgame_resolver.resolve_policy(trunk_policy, player, trunk_depth, subgame_cfr_iterations)
    conv = exploitability.nash_conv(game, new_policy, False)
    print("New strategy (combined trunk + subgame): {}".format(conv.player_improvements))
    print("game value of new strategy:", game_score_of_best_response(game, new_policy, player))
    return new_policy

if __name__ == '__main__':
    game = pyspiel.load_game("kuhn_poker")
    # the policy for any states with depth < TRUNK_DEPTH are in the trunk.
    # Anything with depth (move_number) >= TRUNK_DEPTH is considered in subgames
    # for kuhn poker, trunk_depth = 3 means the first betting action is in the trunk, and the response is a subgame.
    get_policy_from_resolving(game, 700, 300, trunk_depth=3)