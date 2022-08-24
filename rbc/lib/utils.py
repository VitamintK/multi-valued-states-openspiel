from dataclasses import dataclass
import re
from typing import List
from chess import *
# FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]
# SQUARES = [
#     A1, B1, C1, D1, E1, F1, G1, H1,
#     A2, B2, C2, D2, E2, F2, G2, H2,
#     A3, B3, C3, D3, E3, F3, G3, H3,
#     A4, B4, C4, D4, E4, F4, G4, H4,
#     A5, B5, C5, D5, E5, F5, G5, H5,
#     A6, B6, C6, D6, E6, F6, G6, H6,
#     A7, B7, C7, D7, E7, F7, G7, H7,
#     A8, B8, C8, D8, E8, F8, G8, H8,
# ] = range(64)
# COLORS = [WHITE, BLACK] = [True, False]
# COLOR_NAMES = ["black", "white"]

# PieceType = int
# PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
# PIECE_SYMBOLS = [None, "p", "n", "b", "r", "q", "k"]
# PIECE_NAMES = [None, "pawn", "knight", "bishop", "rook", "queen", "king"]
# RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]
# SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]
# SAN_REGEX = re.compile(r"^([NBKRQ])?([a-h])?([1-8])?[\-x]?([a-h][1-8])(=?[nbrqkNBRQK])?[\+#]?\Z")

# BB_EMPTY = 0
# BB_ALL = 0xffff_ffff_ffff_ffff
# BB_SQUARES = [
#     BB_A1, BB_B1, BB_C1, BB_D1, BB_E1, BB_F1, BB_G1, BB_H1,
#     BB_A2, BB_B2, BB_C2, BB_D2, BB_E2, BB_F2, BB_G2, BB_H2,
#     BB_A3, BB_B3, BB_C3, BB_D3, BB_E3, BB_F3, BB_G3, BB_H3,
#     BB_A4, BB_B4, BB_C4, BB_D4, BB_E4, BB_F4, BB_G4, BB_H4,
#     BB_A5, BB_B5, BB_C5, BB_D5, BB_E5, BB_F5, BB_G5, BB_H5,
#     BB_A6, BB_B6, BB_C6, BB_D6, BB_E6, BB_F6, BB_G6, BB_H6,
#     BB_A7, BB_B7, BB_C7, BB_D7, BB_E7, BB_F7, BB_G7, BB_H7,
#     BB_A8, BB_B8, BB_C8, BB_D8, BB_E8, BB_F8, BB_G8, BB_H8,
# ] = [1 << sq for sq in SQUARES]
# BB_FILES = [
#     BB_FILE_A,
#     BB_FILE_B,
#     BB_FILE_C,
#     BB_FILE_D,
#     BB_FILE_E,
#     BB_FILE_F,
#     BB_FILE_G,
#     BB_FILE_H,
# ] = [0x0101_0101_0101_0101 << i for i in range(8)]

# BB_RANKS = [
#     BB_RANK_1,
#     BB_RANK_2,
#     BB_RANK_3,
#     BB_RANK_4,
#     BB_RANK_5,
#     BB_RANK_6,
#     BB_RANK_7,
#     BB_RANK_8,
# ] = [0xff << (8 * i) for i in range(8)]

# def square(file_index: int, rank_index: int) -> Square:
#     """Gets a square number by file and rank index."""
#     return rank_index * 8 + file_index

def parse_san(board, allowed_moves, san: str, quiet=False) -> Move:
    """
    MODIFIED FROM https://python-chess.readthedocs.io/en/latest/_modules/chess.html#Board.parse_san
    Uses the current position as the context to parse a move in standard
    algebraic notation and returns the corresponding move object.

    Ambiguous moves are rejected. Overspecified moves (including long
    algebraic notation) are accepted.

    The returned move is guaranteed to be either legal or a null move.

    :raises: :exc:`ValueError` if the SAN is invalid, illegal or ambiguous.
    """
    try:
        if san == 'pass':
            return Move.null()
        # Castling.
        try:
            if san in ["O-O", "O-O+", "O-O#", "0-0", "0-0+", "0-0#"]:
                # return next(move for move in board.generate_castling_moves() if board.is_kingside_castling(move))
                return next(move for move in allowed_moves if board.is_kingside_castling(move))
            elif san in ["O-O-O", "O-O-O+", "O-O-O#", "0-0-0", "0-0-0+", "0-0-0#"]:
                # return next(move for move in board.generate_castling_moves() if board.is_queenside_castling(move))
                return next(move for move in allowed_moves if board.is_queenside_castling(move))
        except StopIteration:
            raise ValueError(f"illegal san: {san!r} in {board.fen()}")

        # Match normal moves.
        match = SAN_REGEX.match(san)
        if not match:
            # Null moves.
            if san in ["--", "Z0", "0000", "@@@@"]:
                return Move.null()
            elif "," in san:
                raise ValueError(f"unsupported multi-leg move: {san!r}")
            else:
                raise ValueError(f"invalid san: {san!r}")

        # Get target square. Mask our own pieces to exclude castling moves.
        to_square = SQUARE_NAMES.index(match.group(4))
        to_mask = BB_SQUARES[to_square] & ~board.occupied_co[board.turn]

        # Get the promotion piece type.
        p = match.group(5)
        promotion = PIECE_SYMBOLS.index(p[-1].lower()) if p else None

        # Filter by original square.
        from_mask = BB_ALL
        if match.group(2):
            from_file = FILE_NAMES.index(match.group(2))
            from_mask &= BB_FILES[from_file]
        if match.group(3):
            from_rank = int(match.group(3)) - 1
            from_mask &= BB_RANKS[from_rank]

        # Filter by piece type.
        if match.group(1):
            piece_type = PIECE_SYMBOLS.index(match.group(1).lower())
            from_mask &= board.pieces_mask(piece_type, board.turn)
        elif match.group(2) and match.group(3):
            # Allow fully specified moves, even if they are not pawn moves,
            # including castling moves.
            move = board.find_move(square(from_file, from_rank), to_square, promotion)
            if move.promotion == promotion:
                return move
            else:
                raise ValueError(f"missing promotion piece type: {san!r} in {board.fen()}")
        else:
            if not match.group(2) and not match.group(3):
                from_mask &= BB_FILES[FILE_NAMES.index(match.group(4)[0])]
            if match.group(3) and not match.group(2):
                from_mask &= BB_FILES[FILE_NAMES.index(match.group(4)[0])]
            from_mask &= board.pawns

        # Match legal moves.
        matched_move = None
        # for move in board.generate_legal_moves(from_mask, to_mask):
        for move in allowed_moves:
            if move.to_square != to_square:
                continue
            if not (BB_SQUARES[move.from_square] & from_mask):
                continue
            if move.promotion != promotion:
                continue

            if matched_move:
                raise ValueError(f"ambiguous san: {san!r} in {board.fen()}. both {matched_move} and {move}")

            matched_move = move

        if not matched_move:
            # print(f'unmatched move: {san} in {board.fen()}')
            if quiet:
                return Move.null()
            else:
                print(allowed_moves)
                raise ValueError(f"illegal san: {san!r} in {board.fen()}")

        return matched_move
    except Exception as e:
        if quiet:
            return Move.null()
        else:
            raise e

CASTLES = [[Move.from_uci('e8g8'), Move.from_uci('e8c8')], [Move.from_uci('e1g1'), Move.from_uci('e1c1')]]
import reconchess.utilities

def pseudo_legal_castles(board):
    player_castles = CASTLES[board.turn]
    return [castle for castle in player_castles if board.is_pseudo_legal(castle)]

def clean_board(board):
    board_copy = board.copy()
    board_copy.clear_stack()
    return board_copy

@dataclass
class PrettyPrintInput:
    board: Board
    value: float
    probability: float


def pretty_print_boards(boards: List[PrettyPrintInput], last_sense_square=None):
    board_string_rows = [str(board.board).split('\n') for board in boards]
    if last_sense_square is not None:
        sense_file = square_file(last_sense_square)
        sense_rank = 7 - square_rank(last_sense_square)
        for board in board_string_rows:
            for rank in range(sense_rank-1, sense_rank+2):
                board[rank] = board[rank][:2*(sense_file-1)] + '\033[1;33m' + board[rank][2*(sense_file-1):2*(sense_file+2)] + '\033[0;39m' + board[rank][2*(sense_file+2):]
    transpose = zip(*board_string_rows)
    ans = '\n'.join('| |'.join(block for block in row) for row in transpose)
    additional_lines = [[], []]
    WIDTH = 15
    for board in boards:
        additional_lines[0].append(f'{board.value:^{WIDTH}.02}')
        additional_lines[1].append(f'p={board.probability:^{WIDTH-2}.02}')
    additional_lines = '\n'.join(['   '.join(line) for line in additional_lines])
    return ans + '\n' + additional_lines


#######################################################################################
# copied from attacker_bot
QUICK_ATTACKS = [
    # queen-side knight attacks
    [Move(B1, C3), Move(C3, B5), Move(B5, D6),
     Move(D6, E8)],
    [Move(B1, C3), Move(C3, E4), Move(E4, F6),
     Move(F6, E8)],

    # king-side knight attacks
    [Move(G1, H3), Move(H3, F4), Move(F4, H5),
     Move(H5, F6), Move(F6, E8)],

    # four move mates
    [Move(E2, E4), Move(F1, C4), Move(D1, H5), Move(
        C4, F7), Move(F7, E8), Move(H5, E8)],
]
def flipped_move(move):
    def flipped(sq):
        return square(square_file(sq), 7 - square_rank(sq))

    return Move(from_square=flipped(move.from_square), to_square=flipped(move.to_square),
                      promotion=move.promotion, drop=move.drop)
def get_quick_attacks(color):
    move_sequences = QUICK_ATTACKS
    if color == BLACK:
        move_sequences = [list(map(flipped_move, move_sequence)) for move_sequence in move_sequences]
    return move_sequences
##########################################################################################
def opponent_is_attacker(name):
    return name in ['attacker', 'AttackerBot']