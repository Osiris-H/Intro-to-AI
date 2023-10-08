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

    # player should either TOP or BOTTOM
    return board.mancalas[player] - board.mancalas[get_opponent(player)]


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
