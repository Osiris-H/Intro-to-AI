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
    
    raise NotImplementedError


def is_end(board):
    return all(val == 0 for val in board.pockets[TOP]) or all(val == 0 for val in board.pockets[BOTTOM])
