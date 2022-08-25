import math
pieces = set('kqrbnpKQRBNP')
piece_values = {
    'k': 2000,
    'q':9,
    'r':5,
    'b':3,
    'n':3,
    'p':1,
}
# starting_value_sum = sum(piece_values[piece] for piece in 'pppppppprrnnbbkq')
# piece_values = {k:v/starting_value_sum for k,v in piece_values.items()}

def get_centipawns_from_fen(fen: str):
    """returns black centipawn advantage"""
    fen = fen.split()[0]
    ans = 0
    for token in fen:
        if token not in pieces:
            continue
        lowtok = token.lower()
        value = piece_values[lowtok] * 100
        ans += -value if token.isupper() else value
    return ans
def centipawns_to_expectation(centipawns):
    return 1 / (1 + math.exp(-0.0012 * centipawns))
def get_piece_value_from_fen(fen: str):
    """returns black (player 0) piece value - white (player 1) piece value
    scaled to (-1,1)
    """
    black_centipawns = get_centipawns_from_fen(fen)
    return centipawns_to_expectation(black_centipawns) * 2 - 1