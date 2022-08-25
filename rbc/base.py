import chess.engine
import random
from reconchess import *
import os

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'

class EngineWrapper:
    def __init__(self, stockfish_path, threads=None, hash_MB=None):
        self.stockfish_path = stockfish_path
        self.threads = threads
        self.hash_MB = hash_MB
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)
        self.set_configs()
    def set_configs(self):
        if self.threads is not None:
            self.engine.configure({"Threads": self.threads})
        if self.hash_MB is not None:
            self.engine.configure({"Hash": self.hash_MB})
    def reset_engine(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path, setpgrp=True)
        self.set_configs()
    def play(self, *args, **kwargs):
        return self.engine.play(*args, **kwargs)
    def analyse(self, *args, **kwargs):
        return self.engine.analyse(*args, **kwargs)
    def quit(self, *args, **kwargs):
        return self.engine.quit(*args, **kwargs)
    

class UsesStockfish:
    """
    TroutBot uses the Stockfish chess engine to choose moves. In order to run TroutBot you'll need to download
    Stockfish from https://stockfishchess.org/download/ and create an environment variable called STOCKFISH_EXECUTABLE
    that is the path to the downloaded Stockfish executable.
    """

    def __init__(self):
        # make sure stockfish environment variable exists
        # if STOCKFISH_ENV_VAR not in os.environ:
        #     raise KeyError(
        #         'TroutBot requires an environment variable called "{}" pointing to the Stockfish executable'.format(
        #             STOCKFISH_ENV_VAR))
        


        # make sure there is actually a file
        # stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        
        # (Wang) I hardcode stockfish path here for my own personal ease of use.
        # stockfish_path = '/usr/local/bin/stockfish'
        stockfish_path = '../../../downloads/stockfish'
        if not os.path.exists(stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

        # initialize the stockfish engine
        self.engine = EngineWrapper(stockfish_path, threads=self._stockfish_threads, hash_MB=self._stockfish_MB)
        # chess.engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        print('closing!')
        try:
            # if the engine is already terminated then this call will throw an exception
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass