import random
import json
import time

from open_spiel.python.algorithms import best_response

# utils.py vs. common.py:
# I'm thinking this file contains utilities that aren't only useful to the projects here.  They could have more general relevance outside of this project.
# or things that are more "meta-experimental" as opposed to dependencies of experiment code.  Or just little throw-away stuff.
class ResultContainer:
    def __init__(self, name, x=None, y=None, wall=None):
        if x is None:
            x = []
        if y is None:
            y = []
        if wall is None:
            wall = []
        self.name = name
        self.x = x
        self.y = y
        self.wall = wall
        self.infostates = []
        self.exploitability = []
        self.policies = []
        self.inner_exploitability = []
    def save(self, filename):
        print('not saving policies')
        with open(filename, 'w') as f:
            json.dump({
                'name': self.name,
                'x': self.x,
                'y': self.y,
                'wall': self.wall,
                'infostates': self.infostates,
                'inner_exploitability': self.inner_exploitability,
            }, f)

DEBUG_LEVEL = 1
def debug(*args):
    if DEBUG_LEVEL <= 0:
        print(*args)
        
def info(*args):
    if DEBUG_LEVEL <= 1:
        print(*args)

def quick_debug_state(state, with_action_strings=False):
    print(f'__str__: {state}')
    try:
        print(f'infostate str: {state.information_state_string()}')
    except Exception as e:
        pass
    print(f'move number: {state.move_number()}')
    if with_action_strings:
        print(f'legal actions: { {x:state.action_to_string(x) for x in state.legal_actions()}}')
    else:
        print(f'legal actions: {state.legal_actions()}')
    if state.is_terminal():
        print(f'is terminal: {state.returns()}')

def debug_game(state, policy, depth=0):
    prefix = '  '*depth
#     if not state.is_chance_node() and not state.is_terminal():
#         info = ',info: ' + state.information_state_string()
#     else:
#         info = ''
    print(prefix,'his:',state.history_str(), 'pays out {}'.format(state.player_return(0)) if state.is_terminal() else '')
#     print('state is a chance node:', state.is_chance_node())
    for a in state.legal_actions():
        if state.is_chance_node():
            print(prefix,'>', state.history_str(), '->', a, ':', {a:p for a,p in state.chance_outcomes()}[a])
        else:
            print(prefix,'>', state.history_str(), '->', a, ':', policy.action_probabilities(state)[a] if policy is not None else 'no policy')
        debug_game(state.child(a), policy, depth+1)

def debug_playthrough_policies(game, p0, p1):
    s = game.new_initial_state()
    while not s.is_terminal():
        if s.is_player_node():
            pol = list([p0, p1][s.current_player()].action_probabilities(s).items())
        else:
            pol = s.chance_outcomes()
        action = random.choices(pol, [p[1] for p in pol])
        s = s.child(action[0][0])
    print(s.history_str())
    print(s.returns())
    print(s)
    
def game_score_of_best_response(game, policy, player):
    """If `player` plays according to `policy`, what's the expected value payout of a best-responder?
    Lower is better: If this function returns a lower value for policy A than B, then  A is less exploitable than B!
    
    openspiel's exploitability.py isn't useful for comparisons since it compares the value of e.g. p1's strategy vs p2's strategy
    with p1's strategy vs. a best-response.
    It's then not useful to compare the "player improvement" value that uses the value of p1's old strat vs p2's old strat,
    with p1's new strat and p2's new strat as perhaps p1's "player improvement" is higher with new strat NOT because
    it's better but because p2's new strat is worse.
    
    Instead we should do an apples-to-apples comparison of p1's old policy vs a best-response vs. p1's new policy vs. a best-response
    """
    best_responder = 1-player
    best_response_policy = best_response.CPPBestResponsePolicy(game=game, policy=policy, best_responder_id=best_responder)
    value = best_response_policy.value(game.new_initial_state())
    return value

def paste_subpolicy_onto_policy(target_policy, source_policy):
    """Both arguments must be TabularPolicys.  This function mutates target_policy"""
    for s in source_policy.states:
        if s.information_state_string() in target_policy.state_lookup:
            for action, value in enumerate(source_policy.policy_for_key(s.information_state_string())):
                if action not in range(len(target_policy.policy_for_key(s.information_state_string()))):
                    assert value == 0
                    continue
                debug('writing to combined strategy')
                debug(' at state {}, action {} with probability {}'.format(s.information_state_string(), action, value))
                target_policy.policy_for_key(s.information_state_string())[action] = value

def build_and_delete_br_caches(game, br: best_response.BestResponsePolicy):
    """if you want to call br.best_response_action(s) on states that aren't reachable by the opponent, then br needs to have been initiated with cut_threshold=-1"""
    br.value(game.new_initial_state())
    del br.cache_value
    del br.infosets

class Timer:
    def __init__(self, print_fn=print):
        self.last_time = 0
        self.print_fn = print_fn
    def time_lap(self, label):
        self.print_fn(f'lap:\t{time.time() - self.last_time}\t{label}')
        self.last_time = time.time()
    lap = time_lap