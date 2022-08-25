from dataclasses import dataclass
import pyspiel
from open_spiel.python.observation import make_observation

@dataclass
class SubgameStateData:
    """Using this as a workaround for the issue in https://github.com/deepmind/open_spiel/issues/641 -- to define a custom __deepcopy__ class
    as opposed to just storing the data in the actual SubgameState class."""
    # note: just trying out dataclass for fun, this could easily be a normal class
    internal_state: pyspiel.State
    def __deepcopy__(self, memo=None):
        return SubgameStateData(
            internal_state=self.internal_state if self.internal_state is None else self.internal_state.clone(),
        )

def make_game_with_observations_as_infostates(og):
    # given a game og, make a new game g where g's state.information_state_string() = og's state.observation_string()
    # this is almost always wrong (i.e. it has woefully imperfect recall), but I'm trying it here to prototype CFR and DQN on large games like RBC
    # make the game_type
    short_name = og.get_type().short_name
    long_name = og.get_type().long_name
    overrides = {
                'provides_information_state_string': True,
                'provides_information_state_tensor': True,
                'short_name': 'wrappedgame' + short_name,
                'long_name': 'wrappedgame' + long_name,
            }
    all_attrs = ["short_name","long_name","dynamics","chance_mode","information","utility","reward_model","max_num_players","min_num_players","provides_information_state_string","provides_information_state_tensor","provides_observation_string","provides_observation_tensor","parameter_specification"]
    kwargs = {k: (og.get_type().__getattribute__(k) if k not in overrides else overrides[k]) for k in all_attrs}
    game_type = pyspiel.GameType(**kwargs)
    class WrappedGame(pyspiel.Game):
        def __init__(self, params=None):
            self.game = og
            game_info = pyspiel.GameInfo(
                num_distinct_actions=self.game.num_distinct_actions(),  
                max_chance_outcomes=self.game.max_chance_outcomes(), 
                num_players=self.game.num_players(),
                min_utility=self.game.min_utility(),
                max_utility=self.game.max_utility(),
                utility_sum=0.0,
                max_game_length=int(self.game.max_game_length())
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
            return WrappedState(game=self)
        def make_py_observer(self, iig_obs_type=None, params=None):
            # return make_observation(self.game, iig_obs_type) # TODO, this is false
            return WrappedObserver(self.game, iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False), params)
    class WrappedState(pyspiel.State):
        def __init__(self, game):
            super().__init__(game)
            self.data = SubgameStateData(
                internal_state = game.game.new_initial_state(),
            )
        def get_internal_state(self):
            return self.data.internal_state
        def current_player(self):
            return self.get_internal_state().current_player()
        def _legal_actions(self, player=None):
            return self.get_internal_state().legal_actions()
        def chance_outcomes(self):
            """Returns the possible chance outcomes and their probabilities."""
            return self.get_internal_state().chance_outcomes()
        def _apply_action(self, action):
            self.get_internal_state().apply_action(action)
        def undo_action(self, player, action):
            self.get_internal_state().undo_action(player, action)
        def returns(self):
            return self.get_internal_state().returns()
        def is_terminal(self):
            return self.get_internal_state().is_terminal()
        def __str__(self):
            return self.get_internal_state().__str__()
        def action_to_string(self, action):
            return self.get_internal_state().action_to_string(action)
    class WrappedObserver:
        """Observer, conforming to the PyObserver interface (see observation.py)."""
        def __init__(self, base_game, iig_obs_type, params):
            """Initializes an empty observation tensor."""
            new_iig_obs_type = pyspiel.IIGObservationType(
                perfect_recall=False, public_info=iig_obs_type.public_info,
                private_info=iig_obs_type.private_info
            )
            self.observer = make_observation(base_game, new_iig_obs_type, params)
        @property
        def dict(self):
            return self.observer.dict
        @property
        def tensor(self):
            return self.observer.tensor
        def set_from(self, state, player):
            """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
            if state.data.internal_state is not None:
                self.observer.set_from(state.get_internal_state(), player)
        def string_from(self, state, player):
            """Observation of `state` from the PoV of `player`, as a string."""
            # TODO: I'm adding RBC-specific stuff here but it shouldn't be here if this wrapper is to be general
            observation_string = self.observer.string_from(state.get_internal_state(), player)
            return observation_string
            # obs_split = observation_string.split()
            # return ' '.join([obs_split[0], obs_split[1], obs_split[2], obs_split[-1]])
    pyspiel.register_game(game_type, WrappedGame)
    return WrappedGame()