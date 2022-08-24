"""
depth-limited search on subgames consisting of valid root histories (with probabilities just uniform random at first
leaf values given by stockfish
"""

# python
import datetime
import logging
import math
from math import log2
import random
import time
from collections import defaultdict
from typing import Dict
import trout

# other
USE_TQDM = False
if USE_TQDM:
    from tqdm import tqdm
else:
    tqdm = lambda x: x

# reconchess
from reconchess import utilities
from reconchess import *

# openspiel
import pyspiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import external_sampling_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr
# from open_spiel.algorithms import outcome_sampling_mccfr
from open_spiel.python.algorithms import sequence_form_lp

print('openspiel imports done')

# deep learning
import numpy as np
print('dl imports done')

# local
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import lib.game_wrappers.observations_as_infostates
import lib.chess_utils
from lib.matrix_valued_states import make_augmented_subgame
from lib.utils import Timer
from botlib.utils import parse_san, pseudo_legal_castles, pretty_print_boards, PrettyPrintInput, clean_board, get_quick_attacks, opponent_is_attacker
from base import UsesStockfish
print('local imports done')

logger = logging.getLogger('nashfish')
logger.setLevel(logging.DEBUG)

def set_logger_config(player, opponent_name):
    truncated_opponent_name = opponent_name if len(opponent_name) <= 8 else opponent_name[:3]+'..'+opponent_name[-3:]
    color_string = 'w' if player.color else 'b'
    debug_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s: %(message)s') # | %(name)s |
    info_formatter = logging.Formatter(
        f'%(asctime)s | {color_string}.{truncated_opponent_name} | %(levelname)s: %(message)s' # | %(name)s |
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(info_formatter)

    logFilePath = f"logs/nf-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{opponent_name}.txt"
    file_handler = logging.FileHandler(logFilePath)

    file_handler.setFormatter(debug_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def fmt(*args):
    return ' '.join(str(a) for a in args)

# logger = logging.getLogger('simple_example')
# logger.setLevel(logging.DEBUG)

# # create console handler and set level to debug
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler(f"nashfish-{datetime.datetime.now().strftime('%I.%M.%S%p_%b-%d-%Y')}"),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

BOARD_SIZE = 8
SENSE_SIZE = 3
# DEPTH_LIMIT = 6
DEPTH_LIMIT = 3
CFR_ITERS = 4700
NUM_BOARDS = 100
NUM_BOARDS_SUBGAME = 35
# STOCKFISH_SECONDS = 0.0003
STOCKFISH_SECONDS = 0.0033
STOCKFISH_THREADS = 1 # 2
STOCKFISH_MB = 800
NONMATE_VALUE_OUT_OF_MATE = 0.985
MATE_VALUE_OUT_OF_ONE = 0.875 # as opposed to 1 for a realized victory
WIN_VALUE_OUT_OF_ONE = 0.5 # weigh a win comparatively less than a loss
LICHESS_EXP = -0.00115
HARDCODE_VS_ATTACKER = True
timer = Timer(logger.info)



class StockfishValuer:
    """https://python-chess.readthedocs.io/en/latest/engine.html#analysing-and-evaluating-a-position"""
    def __init__(self, engine):
        self.engine = engine
        self.cache = dict()
        self.board_cache = dict()
    def custom_expectation(self, score):
        # modified version of lichess WDL (and already divided by 1000)
        # use NONMATE_VALUE_OUT_OF_MATE to ensure that a mate value is always higher than a nonmate value
        if score.mate() is not None:
            if score == chess.engine.MateGiven:
                # need this check here because both "i have been checkmated" and "i am checkmating" result in score.mate() == 0
                return 1
            elif score.mate() == 0:
                # don't actually need this elif here because the logic below does the same thing but doing this for clarity
                return 0
            abs_score = 1 - abs(score.mate())/1200
            return abs_score if score.mate() > 0 else 1-abs_score
        else:
            return NONMATE_VALUE_OUT_OF_MATE / (1 + math.exp(LICHESS_EXP * score.score())) # 0.0011 or 0.0012
    def _get_board_score(self, board):
        fen = board.fen()
        if fen in self.board_cache:
            return self.board_cache[fen]
        # don't send boards to stockfish where the player-to-move can take the king (as when it sends illegal `pv`s back, python_chess will err)
        
        if board.status() & chess.STATUS_NO_BLACK_KING:
            score = -1
        elif board.status() & chess.STATUS_NO_WHITE_KING:
            score = 1
        elif board.status() & chess.STATUS_OPPOSITE_CHECK:
            score = MATE_VALUE_OUT_OF_ONE if not board.turn else -MATE_VALUE_OUT_OF_ONE
            # print(board.fen(), f'is already invalid, setting score of {score} for black')
        else:
            if board.ep_square is not None and board.is_check() and not board.has_legal_en_passant():
                board.ep_square = None
            if not board.is_valid() and not board.status()==chess.Status.INVALID_EP_SQUARE:
                logger.debug(fmt('board is not valid!', board.fen()))
                logger.debug(board.status())
                logger.debug(str(board))
            try:
                info = self.engine.analyse(board, chess.engine.Limit(time=STOCKFISH_SECONDS), info=chess.engine.Info.SCORE)
                # use lichess model because it seems to scale all the way up to 10 pawns, whereas stockfish (sf14) caps out at a 420 centipawn advantage
                # expected_wins = info['score'].black().wdl(model='lichess').expectation()
                expected_wins = self.custom_expectation(info['score'].black())
                score = (expected_wins * 2 - 1) * MATE_VALUE_OUT_OF_ONE
                # score = info['score'].black().score(mate_score=MATE_SCORE)/MATE_SCORE

            except chess.engine.EngineTerminatedError as e:
                logger.info(e)
                logger.info(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!! chess engine died on the board with fen: {board.fen()}')
                logger.info(board.move_stack)
                expected_wins = lib.chess_utils.get_piece_value_from_fen(fen.split()[0])
                score = (expected_wins * 2 - 1) * MATE_VALUE_OUT_OF_ONE
                logger.info(f'scoring as {score} instead')
                self._restart_engine()
            except Exception as e:
                logger.info('failure occurred while scoring the following state')
                logger.info(fen)
                logger.info(board.move_stack)
                logger.info(f'otherwise unhandled engine error: {e}')
                expected_wins = lib.chess_utils.get_piece_value_from_fen(fen.split()[0])
                score = (expected_wins * 2 - 1) * MATE_VALUE_OUT_OF_ONE
                # raise e
            MATE_SCORE = 30000
            # TODO: make mate in 2 better than mate in 3 so we actually win
            if random.random() < 0.003:
                logger.debug(f"{score:.02}\tmate={info['score'].black().mate()}\tcen={info['score'].black().score()}\tfen={fen}")
        if score > 0:
            pass
            # score *= WIN_VALUE_OUT_OF_ONE
        self.board_cache[fen] = score
        return score
    def get_value(self, policies, state):
        key = str(state)
        if key in self.cache:
            return self.cache[key]
        board = StockfishValuer.open_spiel_state_to_board(state)
        score = self._get_board_score(board)
        self.cache[key] = [score,-score]
        return [score, -score]
    def _restart_engine(self):
        self.engine.reset_engine()
    def open_spiel_state_to_board(state):
        return chess.Board(fen=str(state))

class MaterialValuer:
    def __init__(self):
        pass
    def get_value(self, policies, state):
        black_value = lib.chess_utils.get_piece_value_from_fen(str(state).split()[0])
        return [black_value, -black_value]

class Field:
    def __init__(self, dimensions):
        self.data = np.zeros(dimensions)
    def set(self, coords, value):
        self.data[coords] = value

class Allocator:
    def __init__(self):
        self.fields = []
        self.field_pointers = dict()
    def get(self, field_name, dimensions):
        if field_name in self.field_pointers:
            return self.fields[self.field_pointers[field_name]]
        field = Field(dimensions)
        self.field_pointers[field_name] = len(self.field_pointers)
        self.fields.append(field)
        return field
    def make_tensor(self):
        ans = []
        for field in self.fields:
            ans.extend(field.data.flatten())
        return ans 

def make_subgame_from_states(game, state_valuer, player, states, move_number):
    private_chances = [
        [(0,1)],
        [(i,states[1][i]) for i in range(len(states[1]))]
    ]
    is_state_dl_leaf = lambda s: (s.move_number() >= move_number+DEPTH_LIMIT)
    AugmentedSubgame = make_augmented_subgame(
        actual_game=game,
        private_chances = private_chances,
        num_strategy_options = [1,1],
        is_state_dl_leaf = is_state_dl_leaf,
        private_chance_fn = lambda cs: states[0][cs[1]].clone(),
        strategy_value_fn = state_valuer.get_value,
    )
    return AugmentedSubgame

def make_game_from_fen(fen):
    return lib.game_wrappers.observations_as_infostates.make_game_with_observations_as_infostates(pyspiel.load_game('rbc', {'board_size': BOARD_SIZE, 'fen':fen}))

def requested_move_to_taken_move(board, move):
    # import reconchess
    if move is None:
        return move
    return utilities.revise_move(board, move)
    # if move is None or move is chess.Move.null():
    #     return move
    # if board.piece_at(move.from_square).piece_type == chess.KNIGHT:
    #     return move
    # blocked_by = board.attacks(move.from_square) & chess.SquareSet.between(move.from_square, move.to_square) & board.pieces()
    # if len(blocked_by) == 1:
    #     return chess.Move(move.from_square, blocked_by.pop())
    # elif len(blocked_by) == 0:
    #     return move
    # else:
    #     raise ValueError(f'move {move=} on {board.fen()=} has two possible blockers: {blocked_by}')


class ObsType(Enum):
    your_action_result = 0
    sense = 1
    my_action_result = 2
class Observation:
    def __init__(self, obs_type: ObsType, sense_result=None, requested_move=None, taken_move=None, capture_square=None):
        self.obs_type = obs_type
        self.sense_result = None
        self.taken_move = None
        self.capture_square = None
        if self.obs_type == ObsType.sense:
            self.sense_result = sense_result
        elif self.obs_type == ObsType.my_action_result:
            self.requested_move = requested_move
            self.taken_move = taken_move 
            self.capture_square = capture_square
        elif self.obs_type == ObsType.your_action_result:
            self.capture_square = capture_square
        else:
            raise ValueError('hiya')
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return f'{self.obs_type}: {self.sense_result=} {self.taken_move=} {self.capture_square}'
        
def debug(f):
    def wrapped_f(*args, **kwargs):
        res = f(*args, **kwargs)
        if res == False:
            logger.debug(args, kwargs, res)
        return res
    return wrapped_f
class ActionObservationSequence:
    def __init__(self, color):
        self.color = color
        # self.actions = []
        # self.observations = []
        self.observations = defaultdict(list)
    def add_action(self, action):
        pass
    def add_observation(self, observation, ply):
        self.observations[ply].append(observation)
    def is_consistent_instant(self, board):
        assert board.ply() == len(board.move_stack), f'{board.fen()} {board.ply()} {len(board.move_stack)} {board.move_stack}'
        if board.status() & (chess.Status.NO_WHITE_KING | chess.Status.NO_BLACK_KING):
            return False
        for observation in self.observations[board.ply()]:
            if not self.is_consistent_with_observation(board, observation):
                return False
        return True
    # @debug
    def is_consistent_with_observation(self, board, observation):
        # print(f'checking to see if\n{board}\nis consistent with {observation}')
        if observation.obs_type == ObsType.my_action_result:
            # if observed action isn't possible: the board is inconsistent
            if observation.taken_move == None:
                if observation.requested_move is None:
                    return True
                if observation.requested_move not in utilities.move_actions(board):
                    return False
                if requested_move_to_taken_move(board, observation.requested_move) != observation.taken_move:
                    return False
                return True
            if observation.taken_move not in board.pseudo_legal_moves and not utilities.is_psuedo_legal_castle(board, observation.taken_move):
                return False
            if board.is_capture(observation.taken_move) != (observation.capture_square is not None):
                return False
            return True
        elif observation.obs_type == ObsType.your_action_result:
            move = board.pop()
            last_move_on_board_was_capture = board.is_capture(move)
            board.push(move)
            if last_move_on_board_was_capture != (observation.capture_square is not None):
                return False
            if observation.capture_square is not None:
                return (board.piece_at(observation.capture_square) is not None) and (board.piece_at(observation.capture_square).color != self.color)
            return True
        elif observation.obs_type == ObsType.sense:
            for square, piece in observation.sense_result:
                if board.piece_at(square) != piece:
                    return False
            return True
        else:
            raise ValueError('blah!')
    def __str__(self):
        return str({k:[str(obs) for obs in obss] for k,obss in self.observations.items()})
    # def is_consistent(self, board):
    #     """the most recent `self.observation` should correspond with `board`"""
    #     obs_pointer = len(self.observations)-1
    #     if not self.is_consistent_instant(board, self.observations[obs_pointer]):
    #         return False
    #     # while len(board.move_stack) > 0:

class Boards():
    def __init__(self, starting_board, num_board, fixer_fn):
        self._boards = dict()
        self._probs = dict()
        self.fixer_fn = fixer_fn

        self.add_board(starting_board, 1)
    def get_key(self, board):
        return board.fen()
    def maybe_remove(self, board_key):
        if len(self) == 1:
            logger.info("requested removal on last board in Boards: trying to fix instead")
            board = self._boards[board_key]
            self.remove(board_key)
            self.fixer_fn(board, 15*60)
            self.add_board(board, 1)
        else:
            self.remove(board_key)
    def remove(self, board_key):
        self._boards.pop(board_key)
        self._probs.pop(board_key)
    def keys(self):
        return self._boards.keys()
    def expand(self, board_key, move_probabilities, num_children, validity_fn=lambda x: True):
        """move_probabilities should contain only pseudolegal moves and should contain all pseudolegal moves"""
        board, parent_prob = self._boards[board_key], self._probs[board_key]
        self.remove(board_key)
        move_probs = list(zip(*move_probabilities.items()))
        move_probs[1] = [x/sum(move_probs[1]) for x in move_probs[1]]
        # children = random.sample(move_probs[0], k=min(num_children,len(move_probs[0])))
        children = move_probs[0]
        # print(children)
        child_keys = []
        for move in children:
            prob = move_probabilities[move]
            child = board.copy()
            child.push(move)
            if not validity_fn(child):
                continue
            child_keys.append(self.get_key(child))
            self.add_board(child, prob=parent_prob*prob)
        return child_keys
    def sample_keys(self, k):
        keys = list(self.keys())
        samples = random.sample(keys, [self.get_prob(key) for key in keys], k=k)
        return samples
    def sample_boards(self, k):
        keys = self.sample_keys(k)
        return [self.get_board(k) for k in keys]
    def add_board(self, board, prob):
        key = self.get_key(board)
        if key not in self._boards:
            self._boards[key] = board
            self._probs[key] = prob
        else:
            self._probs[key] += prob
    def get_board(self, board_key):
        return self._boards[board_key]
    def get_prob(self, board_key):
        return self._probs[board_key]
    def set_prob(self, board_key, prob):
        self._probs[board_key] = prob
    def peek(self):
        """returns an arbitrary board"""
        return next(iter(self._boards.values()))
    def __contains__(self, obj):
        return self.get_key(obj) in self._boards
    def __len__(self):
        return len(self._boards)
    def get_board_probs(self):
        return [(self.get_board(key), self.get_prob(key)) for key in self._boards]
    def normalize(self):
        denom = sum(prob for prob in self._probs.values())
        for k in self._probs:
            self._probs[k] /= denom
    def cull(self, n):
        self.normalize()
        keys = list(self._probs.keys())
        chosen = np.random.choice(keys, size=min(n,len(keys)), replace=False, p=[self.get_prob(key) for key in keys])
        for key in keys:
            if key not in chosen:
                self.remove(key)
    # def fill(self, n, fixer_fn):
    #     d = n - len(self)
    #     if d < 0:
    #         return
    #     self.normalize()
    #     keys = list(self._probs.keys())
    #     chosen = np.random.choice(keys, size=d, replace=True, p=[self.get_prob(key) for key in keys])
    #     for key in chosen:
    #         board = self.get_board(key).copy()
    #         if fixer_fn(board):
    #             self.add_board(board, prob = 1/(2*n))


class Nashfish(UsesStockfish, Player):
    def __init__(self):
        self.plys_passed = None
        self.action_observation_sequence = None
        self.boards = None
        self.color = None
        self.my_piece_captured_square = None
        self.observation_table = None
        self.last_sense_square = None
        self._stockfish_threads = STOCKFISH_THREADS
        self._stockfish_MB = STOCKFISH_MB
        super().__init__()
        # self.state_valuer = MaterialValuer()
        self.state_valuer = StockfishValuer(self.engine)
    @property
    def policy(self):
        return None if self.solver is None else self.solver.average_policy()
    def _get_current_observation_string(self):
        return to_dark_fen(self.board, self.observation_table, self.color)
    def _get_current_observation_tensor(self):
        allocator = Allocator()
        write_private_info_tensor(self.board, self.color, 'private', allocator, self.last_sense_square)
        logger.info({name: allocator.fields[x].data for name, x in allocator.field_pointers.items()})
        return allocator.make_tensor()

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.color = color
        self.opponent_name = opponent_name
        set_logger_config(self, opponent_name)
        logger.info(f'Game starting vs. {opponent_name}')
        logger.info('hyperparameters:')
        logger.info(f'{(BOARD_SIZE, SENSE_SIZE, DEPTH_LIMIT, CFR_ITERS, NUM_BOARDS, NUM_BOARDS_SUBGAME, STOCKFISH_SECONDS, MATE_VALUE_OUT_OF_ONE, LICHESS_EXP)=}')    
        self.seconds_left = 900
        self.action_observation_sequence = ActionObservationSequence(color)
        self.boards = Boards(starting_board=board, num_board=NUM_BOARDS, fixer_fn=lambda b, tl: self._make_board_consistent(b, self.action_observation_sequence, time_limit=tl))
        # self.boards = [board.copy() for i in range(NUM_BOARDS)]
        self.plys_passed = 0
        self.solver = None
        self.game = lib.game_wrappers.observations_as_infostates.make_game_with_observations_as_infostates(pyspiel.load_game('rbc', {'board_size': BOARD_SIZE}))
        # self.states = [self.game.new_initial_state()]

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        if self.plys_passed == 0 and self.color == chess.WHITE:
            return
        logger.info('handling opponent move result')

        ######
        if self.solver is None:
            self._solve_for_opponent()
        ######

        self.plys_passed += 1
        self.my_piece_captured_square = capture_square
        self.action_observation_sequence.add_observation(Observation(obs_type=ObsType.your_action_result,capture_square=capture_square),self.plys_passed)
        # valid_boards = []
        for board_key in list(self.boards.keys()):
            board = self.boards.get_board(board_key)
            state = self._get_state_for_board(board, last_sense_square=None, moving=1)
            if state.is_terminal():
                self.boards.remove(board_key)
                continue
            probs = self.policy.action_probabilities(state) # this will just be uniform random if `state` wasn't sampled to be part of the subgame
            # all_moves = [chess.Move.null()] + list(board.pseudo_legal_moves)
            all_moves = [chess.Move.null()] + utilities.move_actions(board)
            pychess_probs = defaultdict(float)
            for open_spiel_action, prob in probs.items():

                # pychess_requested_move = parse_san(board, all_moves, state.action_to_string(open_spiel_action), quiet=False)
                open_spiel_requested_move = state.action_to_string(open_spiel_action)
                pychess_requested_move = chess.Move.null() if open_spiel_requested_move=='pass' else chess.Move.from_uci(open_spiel_requested_move)
                if pychess_requested_move == chess.Move.null():
                    pychess_taken_move = pychess_requested_move
                else:
                    pychess_taken_move = utilities.revise_move(board, pychess_requested_move)
                    pychess_taken_move = chess.Move.null() if pychess_taken_move is None else pychess_taken_move
                pychess_probs[pychess_taken_move] += prob
            # pychess_probs = {
                
            #     for open_spiel_action, prob in probs.items()
            #     }
            if HARDCODE_VS_ATTACKER and opponent_is_attacker(self.opponent_name):
                def is_an_attacker_move(b):
                    if b.peek() == chess.Move.null():
                        return True
                    if b.ply() > 12:
                        return False
                    peek = b.pop()
                    for quick_attack in get_quick_attacks(not self.color):
                        for m in quick_attack:
                            try:
                                if requested_move_to_taken_move(b,m) == peek:
                                    b.push(peek)
                                    return True
                            except AttributeError:
                                pass
                    b.push(peek)
                    return False
                validity_fn = lambda x: self.action_observation_sequence.is_consistent_instant(x) and is_an_attacker_move(x)
            else:
                validity_fn = self.action_observation_sequence.is_consistent_instant
            child_board_keys = self.boards.expand(board_key, pychess_probs, num_children=40, validity_fn=validity_fn)
        if len(self.boards) > 0:
            baseline = 1/(len(self.boards)*5)
        for board_key in self.boards.keys():
            self.boards.set_prob(board_key, max(baseline, self.boards.get_prob(board_key)))
        self.boards.normalize()
        if len(self.boards) == 0:
            logger.warning('after handling opponent move, len(self.boards)==0 so we are trying to fix and add a board')
            board.push(chess.Move.null())
            self._make_board_consistent(board, self.action_observation_sequence, time_limit=60*15)
            self.boards.add_board(board, 1)
            # possible_moves = 
            # instead of a uniformly random move, let's sample based on the opponent's move probabilities according to the solved restricted game policy
            # board.push(random.choice(possible_moves))
            # for child_key in child_board_keys:
            #     success = self._make_board_consistent(self.boards.get_board(), self.action_observation_sequence)
            # if success:
            #     valid_boards.append(board)
            # else:
            #     print('failure :((((((((((((((((')
        logger.info('done handling opponent move result')
        # self.boards = valid_boards
        # if captured_my_piece:
        #     self.board.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        logger.info('choosing sense...')
        # openspiel does this so in order to
        # "Make sure that sense from last round does not reveal a new hidden move: allow players to perceive only results of the last sensing."
        self.observation_table = None

        # if our piece was just captured, sense where it was captured
        # if self.my_piece_captured_square:
        #     return self.my_piece_captured_square
        if self.solver is None or self.seconds_left > 160:
            logger.info('choosing min expected regret sense')
            # selection = self._get_max_entropy_sense()
            selection = self._get_min_expected_regret_sense()
        else:
            logger.info('choosing sense based on max entropy')
            sense_entropies = self._get_sense_entropies()
            state = self._get_state_for_board(self.boards.peek(),moving=0)
            cnt = 0
            while state.is_terminal():
                cnt += 1
                if cnt > 20:
                    logger.info(f"all states are terminal! {str(state)}")
                    return sense_actions[0]
                state = self._get_state_for_board(self.boards.sample_boards(1))
            probs = self.solver.average_policy().action_probabilities(state)
            for action in probs:
                logger.debug(fmt(state.action_to_string(action), probs[action], sense_entropies[action]))
                # cube entropies so that entropies matter a lot more
                probs[action] *= sense_entropies[action][1] * sense_entropies[action][1] * sense_entropies[action][1]
            probs = list(zip(*probs.items()))
            if sum(probs[1]) == 0:
                index = 0
            else:
                probs[1] = [prob/sum(probs[1]) for prob in probs[1]]
                index = random.choices(probs[0], probs[1])[0]
            selection = self._get_sense_square(index)
        self.last_sense_square = selection
        logger.info(f'sensing at {chess.square_name(selection)}')
        return selection
        # if we might capture a piece when we move, sense where the capture will occur
        # future_move = self.choose_move(move_actions, 1)
        # if future_move is not None and self.board.piece_at(future_move.to_square) is not None:
        #     return future_move.to_square

        # otherwise, just randomly choose a sense action, but don't sense on a square where our pieces are located
        # also don't sense on the edge of the board
        for square, piece in self.boards[0].piece_map().items():
            if piece.color == self.color:
                sense_actions.remove(square)
        for square in list(sense_actions):
            if not (0<chess.square_file(square)<BOARD_SIZE-1 and 0<chess.square_rank(square)<BOARD_SIZE-1):
                sense_actions.remove(square)
        logger.info([chess.square_name(sq) for sq in sense_actions])
        return random.choice(sense_actions)

        # choice = random.choice(sense_actions)
        # self.last_sense_square = choice
        # return choice

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        logger.info(f'sensed at {[chess.square_name(sr[0]) for sr in sense_result]}')
        self.action_observation_sequence.add_observation(Observation(obs_type=ObsType.sense,sense_result=sense_result),self.plys_passed)
        # valid_boards = []
        for board_key in list(self.boards.keys()):
            if not self.action_observation_sequence.is_consistent_instant(self.boards.get_board(board_key)):
                    board = self.boards.get_board(board_key).copy()
                    self.boards.maybe_remove(board_key)
                    if len(self.boards) < 10:
                        fixed = self._make_board_consistent(board, self.action_observation_sequence, time_limit=2.5)
                        if fixed and board not in self.boards:
                            self.boards.add_board(board, 1/100)
        if self.seconds_left < 65:
            self.boards.cull(600)
        elif self.seconds_left < 130:
            self.boards.cull(800)
        else:
            self.boards.cull(1050)
        logger.info(f'number of boards after sensing: {len(self.boards)}')

                
            # success = self._make_board_consistent(board, self.action_observation_sequence)
            # if success:
                # valid_boards.append(board)
            # else:
                # print('failure :((((((((((((((((')
        # self.boards = valid_boards
        # self.board.turn = not self.color
        # piece_captured = (self.my_piece_captured_square is not None)
        # for move in [chess.Move.null()] + list(self.board.pseudo_legal_moves):
        #     if self.board.is_capture(move) != piece_captured:
        #         continue
        #     self.board.push(move)
        #     if self._is_consistent(self.board, sense_result, self.my_piece_captured_square):
        #         print(f"i think the most recent move was {move}!")
        #         break
        #     self.board.pop()
        # else: # i'm allowed to use the cursed for-else bc i'm prototyping >:)
        #     print('no single move backfilled: manually changing board instead')
        #     for square, piece in sense_result:
        #         if self.board.piece_at(square) is not None and self.board.piece_at(square).color == self.color:
        #             continue
        #         if self.board.piece_at(square) != piece and piece is not None:
        #             # OPENSPIEL WONT ACCEPT FENS WITH EXTRA PIECES
        #             # SO WE DELETE AN EXISTING PIECE
        #             print(f'attempting to delete extra piece for {piece}')
        #             candidates = self.board.pieces(piece.piece_type, piece.color)
        #             if piece.piece_type == chess.PAWN:
        #                 print("a pawn  has appeared!")
        #                 for sq in candidates:
        #                     if chess.square_file(sq) == chess.square_file(square):
        #                         print(f"so i'm deleting the pawn at {chess.square_name(sq)}")
        #                         self.board.remove_piece_at(sq)
        #                         break
        #             elif piece.piece_type == chess.BISHOP:
        #                 bbsq = chess.BB_SQUARES[square]
        #                 print("a bishop has appeared!")
        #                 # didnt work
        #                 if bbsq & chess.BB_DARK_SQUARES:
        #                     candidates &= chess.BB_DARK_SQUARES
        #                 else:
        #                     candidates &= chess.BB_LIGHT_SQUARES
        #                 if len(candidates) > 0:
        #                     dlt = candidates.pop()
        #                     print(f"so i'm deleting a bishop at {chess.square_name(dlt)}")
        #                     self.board.remove_piece_at(dlt)
        #             else:
        #                 if len(candidates) > 0:
        #                     self.board.remove_piece_at(candidates.pop())
        #     for square, piece, in sense_result:
        #         if self.board.piece_at(square) is not None and self.board.piece_at(square).color == self.color:
        #             continue
        #         self.board.set_piece_at(square, piece)
        # print('done with sensing!  fen:', self.board.fen())
        # self.observation_table = compute_observation_table(self.board,self.color,self.last_sense_square,SENSE_SIZE)

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        self.seconds_left = seconds_left
        board = self.boards.peek()
        logger.info(f'{seconds_left} seconds left, and i think the board is\n{str(board)}')
        # for board, prob in self.boards.get_board_probs():
        #     print(str(board))
        #     print('')
        #     board.turn = self.color
        # print(chess.square_name(self.last_sense_square))
        # print(self._get_current_observation_string())
        # print(np.where(self._get_current_observation_tensor()))
        # board = random.choice(self.boards)

        try:
            logger.info(board.fen())
            # state = self._get_state_for_board(board, last_sense_square=self.last_sense_square)
            # states = [self._get_state_for_board(board, last_sense_square=self.last_sense_square) for board in self.boards]
            # score = self.state_valuer.get_value(None,state)
            # print(f'according to my state valuer, the score is {score}')
            # if score[self.color] > 0.85 * MATE_VALUE_OUT_OF_ONE:
            #     self.board = board.copy()
            #     return trout.TroutBot.choose_move(self, move_actions, seconds_left)
            # print(state.information_state_string())
            # probs = self._get_move_for_states(states, 1, its=(None if seconds_left>30 else 100))
            state = self._get_state_for_board(board, last_sense_square=self.last_sense_square)
            logger.info(state.information_state_string())
            if self.solver is None or True:
                # states = {self._get_state_for_board(board, last_sense_square=self.last_sense_square): prob for board, prob in self.boards.get_board_probs()}
                # states = ([self._get_state_for_board(board, last_sense_square=self.last_sense_square)], [1])
                self.boards.normalize()
                board_probs = self.boards.get_board_probs()
                if seconds_left <= 10:
                    return None
                    # return random.choice(move_actions)
                elif seconds_left < 30000:
                    logger.info(f'only {seconds_left} seconds left: delegating to stockfish.  fingers crossed!')
                    delegated_move = self._get_greedy_move(move_actions)
                    # board_prob_index = np.random.choice(range(len(board_probs)), size=1, replace=False, p=list(zip(*board_probs))[1])
                    # self.board = board_probs[board_prob_index][0].copy()
                    # logger.info(f'sampled a possible state: {self.board.fen()}')
                    # delegated_move = trout.TroutBot.choose_move(self, move_actions, seconds_left)
                    # logger.info(f'{delegated_move=}')
                    return delegated_move
                sampled_board_prob_indices = np.random.choice(range(len(board_probs)), size=min(NUM_BOARDS_SUBGAME,len(board_probs)), replace=False, p=list(zip(*board_probs))[1])
                sampled_board_probs = [board_probs[i] for i in sampled_board_prob_indices]
                states = list(zip(*sampled_board_probs))
                
                states[0] = [self._get_state_for_board(board, last_sense_square=self.last_sense_square) for board in states[0]]
                probs = self._get_move_for_states(states, 1, its=(None if seconds_left>120 else 150))
            else:
                probs = self.solver.average_policy().action_probabilities(state)
            # print(sorted({state.action_to_string(i): prob for i,prob in probs.items()}.items(),key=lambda x:x[1]))
            probs = list(zip(*probs.items()))
            # square all probs to make the better actions more likely
            probs[1] = [prob*prob if prob>=0.012 else 1e-15 for prob in probs[1]]
            probs[1] = [prob/sum(probs[1]) for prob in probs[1]]
            logger.info('probabilities squared to bias towards better moves:')
            logger.info(sorted({state.action_to_string(i): prob for i,prob in zip(*probs)}.items(),key=lambda x:x[1]))
            move = random.choices(probs[0], probs[1])[0]
            logger.info(f'chosen move: {state.action_to_string(move)}')

            # move = parse_san(board, move_actions, state.action_to_string(move))
            open_spiel_move = state.action_to_string(move)
            move = chess.Move.null() if open_spiel_move=='pass' else chess.Move.from_uci(open_spiel_move)
            if move == chess.Move.null():
                return None
            return move
        except pyspiel.SpielError as e:
            logger.info(e)
            return random.choice(move_actions + [None])

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        logger.info(f'move done! requested: {requested_move}; taken: {taken_move}')
        self.action_observation_sequence.add_observation(Observation(obs_type=ObsType.my_action_result,requested_move=requested_move,taken_move=taken_move,capture_square=capture_square),self.plys_passed)
        self.plys_passed += 1
        # valid_boards = []
        # for board in self.boards:
        #     success = self._make_board_consistent(board, self.action_observation_sequence)
        #     if success:
        #         valid_boards.append(board)
        #         if taken_move is None:
        #             board.push(chess.Move.null())
        #         else:
        #             board.push(taken_move)
        #     else:
        #         print('failure :((((((((((((((((')
        # self.boards = valid_boards
        for board_key in list(self.boards.keys()):
            if not self.action_observation_sequence.is_consistent_instant(self.boards.get_board(board_key)):
                self.boards.maybe_remove(board_key)
            else:
                board = self.boards.get_board(board_key)
                prob = self.boards.get_prob(board_key)
                self.boards.remove(board_key)
                if taken_move is None:
                    board.push(chess.Move.null())
                else:
                    board.push(taken_move)
                self.boards.add_board(board, prob)
        if self.seconds_left > 10:
            self._solve_for_opponent()
        # if taken_move is not None:
        #     self.board.turn = self.color
        #     self.board.push(taken_move)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        logger.info('\a')
        winner = 'WE WON!' if winner_color==self.color else 'WE LOST :('
        logger.info(f'GAME IS OVER: {winner}  win reason: {win_reason}')
        logger.info("I think the game's history was:")
        logger.info(self.boards.peek().move_stack)
        logger.info(f'actual history: {game_history}')
        super().handle_game_end(winner_color,win_reason,game_history)

    def _get_state_for_board(self, board, last_sense_square=None, moving=1):
        # if not moving, then sensing
        game = make_game_from_fen(board.fen())
        if moving:
            last_sense_square = 0 if last_sense_square is None else self._get_open_spiel_sense_index(last_sense_square)
            return game.new_initial_state().child(last_sense_square)
        else:
            return game.new_initial_state()
    def _get_scaled_cfr_iters(self, time_left):
        return time_left/900 * CFR_ITERS
    def _solve_for_opponent(self):
        # self._fill_boards()
        self.boards.normalize()
        board_probs = self.boards.get_board_probs()
        # sampled_board_probs = random.choices(board_probs, list(zip(*board_probs))[1], k=NUM_BOARDS_SUBGAME)
        sampled_board_prob_indices = np.random.choice(range(len(board_probs)), size=min(NUM_BOARDS_SUBGAME,len(board_probs)), replace=False, p=list(zip(*board_probs))[1])
        sampled_board_probs = [board_probs[i] for i in sampled_board_prob_indices]
        states = [(self._get_state_for_board(board, last_sense_square=None), prob) for board, prob in sampled_board_probs]
        # states = [(self._get_state_for_board(board, last_sense_square=None), prob) for board, prob in self.boards.get_board_probs()]
        states = [(state,p) for state,p in states if not state.is_terminal()]
        if len(states) == 0:
            self.solver = None
        else:
            states = list(zip(*states))
            states[1] = [p/sum(states[1]) for p in states[1]]
        # assert len(set(s.information_state_string().split()[0] for s in states)) == 1, set(s.information_state_string() for s in states)
            logger.info(f'total boards: {len(self.boards)}')
            subgame = self._make_subgame_for_states(states, 1)
            solver = self._get_solver_for_subgame(subgame, int(self._get_scaled_cfr_iters(self.seconds_left) * 0.85))
            self.solver = solver

            # print probs
            probs = solver.average_policy().action_probabilities(states[0][0])
            probs =  list(zip(*probs.items()))
            # # cube all probs to make the better actions more likely
            # probs[1] = [prob*prob*prob for prob in probs[1]]
            probs[1] = [prob/sum(probs[1]) for prob in probs[1]]
            # print('probabilities cubed to bias towards better moves:')
            logger.debug(f'estimated opponent move probabilities for {str(states[0][0])}:')
            logger.debug(sorted({states[0][0].action_to_string(i): prob for i,prob in zip(*probs)}.items(),key=lambda x:x[1]))
    def _solve_subgame_LP(self, subgame):
        val1, val2, p1, p2 = sequence_form_lp.solve_zero_sum_game(subgame)
        return p1, p2
    def _get_solver_for_subgame(self, subgame, its):
        # cfr_solver = pyspiel.OutcomeSamplingMCCFRSolver(subgame)
        # cfr_solver = pyspiel.ExternalSamplingMCCFRSolver(subgame)
        cfr_solver = outcome_sampling_mccfr.OutcomeSamplingSolver(subgame) #THERE IS A BUG PREVENTING USAGE OF PYSPIEL.OUTCOMESAMPLINGMCCFRSOLVER otherwise we'd use that: https://github.com/deepmind/open_spiel/issues/905
        logger.debug(f'CFR solving for {its} iterations')
        timer.lap('starting cfr solve')
        for i in tqdm(range(its)):
            # cfr_solver.run_iteration()
            cfr_solver.iteration()
        timer.lap('end cfr solve')
        return cfr_solver
        return cfr_solver.average_policy()
    def _make_subgame_for_states(self, states, move_number):
        logger.debug('root infostates')
        infostates = []
        for i, state in enumerate(states[0]):
            infostates.append(fmt(state.information_state_string(), ' ; ', str(state), ' ; ', states[1][i]))
        logger.debug('\n'.join(infostates))
        return make_subgame_from_states(self.game, self.state_valuer, self.color, states, move_number)
    def _get_move_for_states(self, states, move_number, its=None):
        """states: ([state,], [state_prob,])"""
        if its is None:
            its = CFR_ITERS
        timer.lap('start assert')
        assert len(set(s.information_state_string().split()[0] for s in states[0])) == 1, set(s.information_state_string() for s in states[0])
        timer.lap('start make augmented subgame')
        subgame = self._make_subgame_for_states(states, move_number)
        timer.lap('starting build cfr solver')
        solver = self._get_solver_for_subgame(subgame, its)
        return solver.average_policy().action_probabilities(states[0][0])
        # cfr_solver = external_sampling_mccfr.ExternalSamplingSolver(subgame)
        # cfr_solver = outcome_sampling_mccfr.OutcomeSamplingSolver(subgame)
        # for i in tqdm(range(its)):
        #     cfr_solver.iteration()
        # return cfr_solver.average_policy().action_probabilities(states[0])
    # def _is_consistent(self, board, sense_result, piece_captured_square):
    #     for square, piece in sense_result:
    #         if board.piece_at(square) != piece:
    #             return False
    #     return True
    def _make_board_consistent(self, board, action_observation_sequence, time_limit=3.5):
        # print(self.action_observation_sequence)
        assert board.turn == self.color
        pops = 0
        my_popped_moves = []
        logger.debug(f'are we already consistent?  ply={board.ply()}')
        success = action_observation_sequence.is_consistent_instant(board)
        search_time_start = time.time()
        deadline = search_time_start + time_limit if time_limit is not None else None
        while not success:
            if deadline is not None and time.time() > deadline:
                logger.info('failure :((((')
                return False
            if len(board.move_stack) == 0:
                logger.info(str(self.action_observation_sequence))
                return False
            if board.turn != self.color:
                my_popped_moves.append(board.pop())
            if len(board.move_stack) == 0:
                logger.info(str(self.action_observation_sequence))
                return False
            last_move = board.pop()
            pops += 1
            logger.debug(f'{pops=} attempting to make the board consistent from the starting point of')
            logger.debug(board)
            success = self._make_board_consistent_dfs(board, action_observation_sequence, my_popped_moves, last_move, deadline=deadline)
        return True
    def _make_board_consistent_dfs(self, board, action_observation_sequence, my_popped_moves, move_to_not_consider=None, deadline=None):
        if deadline is not None and time.time() > deadline:
            return False
        BRANCHING_FACTOR = 80
        pseudo_legal_moves = list(board.pseudo_legal_moves) + pseudo_legal_castles(board)
        # pseudo_legal_moves = utilities.move_actions(board)
        for move in random.sample(pseudo_legal_moves, k=min(len(pseudo_legal_moves),BRANCHING_FACTOR)) + [chess.Move.null()]:
            if move_to_not_consider is not None and move == move_to_not_consider:
                continue
            board.push(move)
            if action_observation_sequence.is_consistent_instant(board):
                # print("alright! we're partially consistent!")
                # print(board.fen())
                # print(board)
                if len(my_popped_moves) == 0:
                    return True
                board.push(my_popped_moves.pop())
                # print(f'calling recursive fn with board {board.fen()} and {my_popped_moves=}')
                success = self._make_board_consistent_dfs(board, action_observation_sequence, my_popped_moves, deadline=deadline)
                if success:
                    return True
                my_popped_moves.append(board.pop())
            board.pop()
        return False
    def _get_sense_expected_regrets(self, intra_partition_weighting='uniform', move_prediction='reality', inter_partition_weighting='not uniform'):
        """any sense partitions the set of possible boards into subsets.  For each subset, there is an action `best` with maximum EV.
        However, for some boards in the subset, there will be some regret = score of actual best move - score of `best`.
        For each subset, we can calculate the expected value of the regret from choosing `best`.
        For each partition, we can calculate the expected value of the regret from each subset's `best`.
        intra-partition weighting: since EV = sum(p(x) * x), do we use actual p(x) for self.boards, or do we use 1/n (uniform)
        inter-partition weighting: since EV = sum(p(x) * x), do we use actual p(x) from cum_probability, or do we use 1/num_of_partitions (uniform)
        move prediction: to choose `best`, do we use the weighting from intra_partition_weighting, or do we use the probabilities (which is what we will actually do)
        """
        all_moves = utilities.move_actions(self.boards.peek())
        expected_score_dict = dict()
        board_probs = self.boards.get_board_probs()
        # TODO: if we parallelize get_move_values_for_board, we don't have to sample these: we could get em all
        board_probs = random.sample(board_probs, k=min(len(board_probs), 135))
        logger.info(f'to calculate expected regrets: sampling {len(board_probs)} boards of {len(self.boards)} total')

        for board, prob in tqdm(board_probs):
            expected_score_dict[self.boards.get_key(board)] = self._get_move_values_for_board(all_moves, board)
            
        letter = lambda x: ' ' if x is None else x.symbol()
        ans = []
        for sense_square in tqdm(chess.SquareSet(chess.BB_ALL ^ (chess.BB_RANK_1 | chess.BB_FILE_A | chess.BB_FILE_H | chess.BB_RANK_8))):
            squares = []
            for sq in chess.SQUARES:
                if chess.square_distance(sq, sense_square) <= 1:
                    squares.append(sq)
            # er = self._get_expected_regrets_of_sense(sq)
            all_evs = defaultdict(lambda : defaultdict(float))
            all_evs_reality = defaultdict(lambda : defaultdict(float))
            hindsight_best_evs = defaultdict(float)
            cum_probability = defaultdict(float)
            for board, _prob in board_probs:
                if intra_partition_weighting=='uniform':
                    prob = 1/len(board_probs)
                else:
                    prob = _prob
                if move_prediction=='reality':
                    move_prediction_prob = _prob
                else:
                    move_prediction_prob = prob
                # results = defaultdict(list)
                sense_result = ''.join([letter(board.piece_at(sq)) for sq in squares])

                best_score = -1
                expected_scores = expected_score_dict[self.boards.get_key(board)]
                for move in expected_scores:
                    all_evs[sense_result][move] += prob * expected_scores[move]
                    all_evs_reality[sense_result][move] += move_prediction_prob * expected_scores[move]
                    best_score = max(best_score, expected_scores[move])
                hindsight_best_evs[sense_result] += prob * best_score
                cum_probability[sense_result] += prob
                
            regrets = 0
            denom = len(cum_probability)
            for sense_result, prob in cum_probability.items():
                if inter_partition_weighting=='uniform':
                    prob = 1/denom 
                best_move = max(all_evs_reality[sense_result].items(), key=lambda x: x[1])[0]
                best_ev = all_evs[sense_result][best_move]
                regrets += prob * (hindsight_best_evs[sense_result] - best_ev)
            # return ans
            ans.append((sense_square, regrets))
        return ans
    def _get_sense_entropies(self):
        ans = []
        for sq in chess.SquareSet(chess.BB_ALL ^ (chess.BB_RANK_1 | chess.BB_FILE_A | chess.BB_FILE_H | chess.BB_RANK_8)):
            ent = self._get_entropy_of_sense(sq)
            ans.append((sq, ent))
        return ans
    def _get_min_expected_regret_sense(self):
        expected_regrets = self._get_sense_expected_regrets()
        logger.debug('expected regrets:')
        for sense, regrets in expected_regrets:
            logger.debug(f'{chess.square_name(sense)}: {regrets}')
        return min(expected_regrets, key=lambda x: x[1])[0]
    def _get_max_entropy_sense(self):
        sq_entropies = self._get_sense_entropies()
        return max(sq_entropies, key=lambda x: x[1])[0]
    # def _get_expected_regrets_of_sense(self, sense_square, intra_partition_weighting='uniform', inter_partition_weighting='uniform'):
        
            
        
    def _get_entropy_of_sense(self, sense_square):
        # TODO: could be entropy of policy (distribution) induced by boards
        # or better yet, sensing is just considered any other move and sensing is also given
        # by solving the subgame
        # entropy = sum(-p * log_2(p) for p in ps)
        squares = []
        results = defaultdict(int)
        for sq in chess.SQUARES:
            if chess.square_distance(sq, sense_square) <= 1:
                squares.append(sq)
        letter = lambda x: ' ' if x is None else x.symbol()
        for board, prob in self.boards.get_board_probs():
            sense_result = ''.join([letter(board.piece_at(sq)) for sq in squares])
            results[sense_result] += 1
        ans = 0
        for cnt in results.values():
            p = cnt/len(self.boards)
            ans += log2(p) * -p
        # print(f'{sense_square} has {ans} entropy')
        return ans
    def _get_greedy_move(self, moves):
        # evs = {move: 0 for move in moves}
        self.boards.normalize()
        board_probs = self.boards.get_board_probs()
        num_boards = 50 if self.seconds_left > 70 else 15
        selected_board_probs = sorted(board_probs, key=lambda x: x[1], reverse=True)[:num_boards]
        mul = -1 if self.color else 1
        pretty_string = pretty_print_boards([PrettyPrintInput(board=board, probability=prob, value=mul*self.state_valuer._get_board_score(clean_board(board))) for board, prob in selected_board_probs[:10]], last_sense_square=self.last_sense_square)
        board_strings = '\n'.join([f'{board.fen()} ; {prob}' for board,prob in selected_board_probs[:10]])
        logger.debug(f'top 10 most probable states:\n{board_strings}')
        logger.info(f'top 10 most probable states:\n{pretty_string}')
        return self._get_greedy_move_for_board_probs(moves, selected_board_probs)
        # for board, prob in tqdm(selected_board_probs):
        #     for move in moves:
        #         if move is None:
        #             move = chess.Move.null()
        #         taken_move = requested_move_to_taken_move(board, move)
        #         taken_move = chess.Move.null() if taken_move is None else taken_move
        #         bcopy = board.copy()
        #         bcopy.push(taken_move)
        #         # clearing the stack because stockfish doesn't know how to deal with move stacks that have null moves
        #         bcopy.clear_stack()
        #         ev = self.state_valuer._get_board_score(bcopy)
        #         if self.color:
        #             ev *= -1
        #         evs[move] += prob * ev
        # return max(evs.items(), key=lambda x: x[1])[0]
    def _get_greedy_move_for_board_probs(self, moves, board_probs):
        move_values = self._get_move_values_for_board_probs(moves, board_probs)
        logger.debug(f'move EVs: {move_values}')
        return max(move_values.items(), key=lambda x: x[1])[0]
    def _get_move_values_for_board_probs(self, moves, board_probs):
        evs = {move: 0 for move in moves}
        for board, prob in tqdm(board_probs):
            board_evs = self._get_move_values_for_board(moves, board)
            for move in board_evs:
                evs[move] += prob * board_evs[move]
        return evs
    def _get_move_values_for_board(self, moves, board):
        evs = dict()
        for move in moves:
            if move is None:
                move = chess.Move.null()
            taken_move = requested_move_to_taken_move(board, move)
            taken_move = chess.Move.null() if taken_move is None else taken_move
            bcopy = board.copy()
            bcopy.push(taken_move)
            # clearing the stack because stockfish doesn't know how to deal with move stacks that have null moves
            bcopy.clear_stack()
            ev = self.state_valuer._get_board_score(bcopy)
            if self.color:
                ev *= -1
            evs[move] = ev
        return evs
        # votes = {move: 0 for move in moves}
        # self.boards.normalize()
        # board_probs = self.boards.get_board_probs()
        # for board, prob in sorted(board_probs, key=lambda x: x[1], reverse=True)[:50]:
        #     # if there are any ally pieces that can take king, execute one of those moves
        #     enemy_king_square = board.king(not self.color)
        #     if enemy_king_square:
        #         enemy_king_attackers = board.attackers(self.color, enemy_king_square)
        #         if enemy_king_attackers:
        #             attacker_square = enemy_king_attackers.pop()
        #             move = chess.Move(attacker_square, enemy_king_square)
        #             votes[move] += prob
        #             continue
        #     move = self.engine.play(board, chess.engine.Limit(time=0.02)).move
        #     votes[move] += prob
        # return max(votes.items(), key=lambda x: x[1])[0]
    def _get_open_spiel_sense_index(self, sense_square):
        # sense_square is center.
        all_choices = chess.SquareSet(chess.BB_ALL ^ (chess.BB_RANK_1 | chess.BB_FILE_A | chess.BB_FILE_H | chess.BB_RANK_8))
        # bot_left_corner = sense_square - 8 - 1
        return list(all_choices).index(sense_square)
    def _get_sense_square(self, open_spiel_sense_index):
        all_choices = chess.SquareSet(chess.BB_ALL ^ (chess.BB_RANK_1 | chess.BB_FILE_A | chess.BB_FILE_H | chess.BB_RANK_8))
        return list(all_choices)[open_spiel_sense_index]
    def _fill_boards(self):
        while len(self.boards) < NUM_BOARDS:
            self.boards.append(random.choice(self.boards).copy())



def to_dark_fen(board, observation_table, color):
    # translated from c++ from open_spiel's `dark_chess.cc`
    # encode the board
    fen = []
    for rank in reversed(range(BOARD_SIZE)):
        num_empty = 0
        for file in range(BOARD_SIZE):
            square = chess.square(file,rank)
            # index = square_to_index(file,rank)
            if not observation_table[chess.square_name(square)]:
                if num_empty > 0:
                    fen.append(str(num_empty))
                    num_empty = 0
                fen.append('?')
            else:
                piece = board.piece_at(square)
                if piece is None:
                    num_empty +=1
                else:
                    if num_empty > 0:
                        fen.append(str(num_empty))
                        num_empty = 0
                    fen.append(piece.symbol())
        if num_empty > 0:
            fen.append(str(num_empty))
        if rank > 0:
            fen.append('/')
    # color to play
    fen.append(f' {"w" if board.turn == chess.WHITE else "b"}')
    # by castling rights
    fen.append(' ')
    castling_rights = []
    if color == chess.WHITE:
        if board.has_kingside_castling_rights(color):
            castling_rights.append('K')
        if board.has_queenside_castling_rights(color):
            castling_rights.append('Q')
    else:
        if board.has_kingside_castling_rights(color):
            castling_rights.append('k')
        if board.has_queenside_castling_rights(color):
            castling_rights.append('q')
    if len(castling_rights) == 0:
        fen.append('-')
    else:
        fen.extend(castling_rights)
    # en passant square
    ep_square = '-'
    if board.has_pseudo_legal_en_passant():
        ep_square = board.ep_square
    fen.append(' ')
    fen.append(ep_square)
    # half-move clock
    fen.append(' ')
    fen.append(str(board.halfmove_clock))
    # full-move clock
    fen.append(' ')
    fen.append(str(board.fullmove_number))
    return ''.join(fen)

def compute_observation_table(board, color, sense_square, sense_size):
    # translated from `rbc.cc`
    observation_table = defaultdict(bool)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            square = chess.square(x,y)
            piece = board.piece_at(square)
            if piece is not None and piece.color == color:
                observation_table[chess.square_name(square)] = True
    if sense_square is None:
        return observation_table
    # inner_size = board_size - sense_size + 1
    # sense_sq = 
    rank, file = chess.square_rank(sense_square), chess.square_file(sense_square)
    rank -= 1 + sense_size//2
    file -= 1 + sense_size//2
    for x in range(max(file,0), min(file+sense_size, BOARD_SIZE)):
        for y in range(max(rank,0), min(rank+sense_size, BOARD_SIZE)):
            sq = chess.square(x,y)
            # piece = board.piece_at(sq)
            observation_table[chess.square_name(sq)] = True
    return observation_table

def write_scalar(val, mn, mx, field_name, allocator):
    out = allocator.get(field_name, (mx-mn+1,))
    out.set(val-mn, 1)

def write_binary(val, field_name, allocator):
    write_scalar(1 if val else 0, 0, 1, field_name, allocator)

def write_pieces(color, piece_type, board, sense_coords, sense_size, prefix, allocator):
    r, f = sense_coords
    out = allocator.get(prefix+'_'+chess.piece_name(piece_type)+'_pieces', (BOARD_SIZE,BOARD_SIZE))
    if sense_size is None:
        return
    for file in range(max(0,f), min(f+sense_size, BOARD_SIZE)):
        for rank in range(max(r,0), min(r+sense_size, BOARD_SIZE)):
            piece_on_board = board.piece_at(chess.square(file,rank))
            write_square = piece_on_board is not None and piece_on_board.color==color and piece_on_board.piece_type == piece_type
            out.set((file,rank), 1 if write_square else 0)

def write_private_info_tensor(board, player, prefix, allocator, sense_square):
    color = chess.WHITE if player else chess.BLACK
    other_color = not color
    # piece configuration
    for piece_type in [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
        write_pieces(color, piece_type, board, [0,0], BOARD_SIZE, prefix, allocator)
    # castling rights
    write_binary(board.has_queenside_castling_rights(color), prefix+'_left_castling', allocator)
    write_binary(board.has_kingside_castling_rights(color), prefix+'_right_castling', allocator)
    # last sensing
    r, f = chess.square_rank(sense_square), chess.square_file(sense_square)
    r -= 1 + SENSE_SIZE//2
    f -= 1 + SENSE_SIZE//2
    if sense_square is not None:
        for piece_type in [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
            # we set sense_location to None in callers instead of setting it to None here like in the c++ code
            write_pieces(other_color, piece_type, board, [r,f], SENSE_SIZE, prefix+'_sense', allocator)

