###############################################################################
# This file contains helper functions and the heuristic functions
# for our AI agents to play the Mancala game.
#
# CSC 384 Fall 2023 Assignment 2
# version 1.0
###############################################################################

import sys

###############################################################################
### DO NOT MODIFY THE CODE BELOW

### Global Constants ###
TOP = 0
BOTTOM = 1

### Errors ###
class InvalidMoveError(RuntimeError):
    pass

class AiTimeoutError(RuntimeError):
    pass

### Functions ###
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_opponent(player):
    if player == BOTTOM:
        return TOP
    return BOTTOM

### DO NOT MODIFY THE CODE ABOVE
###############################################################################


def heuristic_basic(board, player):
    """
    Compute the heuristic value of the current board for the current player 
    based on the basic heuristic function.

    :param board: the current board.
    :param player: the current player.
    :return: an estimated utility of the current board for the current player.
    """

    player_score = board.mancalas[player]
    opp_score = board.mancalas[get_opponent(player)]
    # consider the case when game is end
    # if is_end(board):
    #     if any(board.pockets[player]):
    #         player_score += next(x for x in board.pockets[player] if x != 0)
    #     elif any(board.pockets[get_opponent(player)]):
    #         opp_score += next(x for x in board.pockets[get_opponent(player)] if x != 0)
    return player_score - opp_score


def heuristic_advanced(board, player): 
    """
    Compute the heuristic value of the current board for the current player
    based on the advanced heuristic function.

    :param board: the current board object.
    :param player: the current player.
    :return: an estimated heuristic value of the current board for the current player.
    """

    # Strategies to follow:
    # Minimize Opponent's Non-Empty Pockets
    # Maximize even distribution

    def store_diff(board, player):
        return heuristic_basic(board, player)

    def non_empty_pockets(board, player):
        player_non_empty = sum(1 for pocket in board[player].pockets if pocket > 0)
        opp_non_empty = sum(1 for pocket in board[get_opponent(player)].pockets if pocket > 0)
        return player_non_empty - opp_non_empty

    def even_distribute(board, player):
        mean = sum(board[player].pockets) / len(board[player].pockets)
        variance = sum((pocket - mean) ** 2 for pocket in board[player].pockets) / len(board[player].pockets)
        return -variance

    w1, w2, w3 = 10, 4, 2
    score = w1*store_diff(board, player) + w2*non_empty_pockets(board, player) + w3*even_distribute(board, player)
    return score


def is_end(board):
    return all(val == 0 for val in board.pockets[TOP]) or all(val == 0 for val in board.pockets[BOTTOM])
