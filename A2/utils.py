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

    def potential_captures(board, player):
        curr_pocket = board.pockets[player]
        opp_pocket = board.pockets[get_opponent(player)]
        score = 0
        count = 0
        for i, value in enumerate(curr_pocket):
            if player == TOP:
                if value != 0 and i-value >= 0 and curr_pocket[i-value] == 0:
                    count += 1
                    score += opp_pocket[i-value]
            else:
                if value != 0 and i+value <= len(curr_pocket)-1 and curr_pocket[i+value] == 0:
                    count += 1
                    score += opp_pocket[i+value]
        if score != 0:
            score /= count
        return score

    def empty_pockets(board, player):
        player_empty = sum(1 for value in board.pockets[player] if value == 0)
        opp_empty = sum(1 for value in board.pockets[get_opponent(player)] if value == 0)
        return opp_empty - player_empty

    def game_phase(board):
        store_val = sum(board.mancalas)
        pocket_val = sum(sum(pocket) for pocket in board.pockets)
        total = store_val + pocket_val
        if pocket_val >= 3 * total / 4:
            return 0  # opening
        elif pocket_val >= total / 4:
            return 1  # mid-game
        else:
            return 2  # end-game

    def even_distribute(board, player):
        mean = sum(board.pockets[player]) / len(board.pockets[player])
        variance = sum((pocket - mean) ** 2 for pocket in board.pockets[player]) / len(board.pockets[player])
        return -variance

    stage = game_phase(board)
    # stage = 2
    if stage == 0:
        w1, w2, w3 = 1, 0.5, 1
    elif stage == 1:
        w1, w2, w3 = 1, 1, 1.5
    else:
        w1, w2, w3 = 1, 1, 2
    # total = w1 + w2 + w3
    # w1, w2, w3 = w1 / total, w2 / total, w3 / total
    score = w1*store_diff(board, player) + w2*empty_pockets(board, player) + w3*potential_captures(board, player)
    return score


def is_end(board):
    return all(val == 0 for val in board.pockets[TOP]) or all(val == 0 for val in board.pockets[BOTTOM])
