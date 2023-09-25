############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 1 Starter Code
## v1.1
##
## Changes: 
## v1.1: removed the hfn paramete from dfs. Updated solve_puzzle() accordingly.
############################################################

from typing import List
import heapq
from heapq import heappush, heappop
import time
import argparse
import math # for infinity

from board import *

def is_goal(state):
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """

    box_pos = state.board.boxes
    dest = state.board.storage

    return set(box_pos) == set(dest)


def get_path(state):
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """

    states = []
    cur = state
    while True:
        states.insert(0, cur)
        if cur.depth == 0:
            break
        else:
            cur = cur.parent

    return states


def get_successors(state):
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """

    def sanity_check(position):
        return position[0] <= state.board.width and position[1] <= state.board.height

    def check_box_pos(pos):
        if any(pos in container for container in (state.board.obstacles, state.board.boxes, state.board.robots)):
            return False
        return True

    states = []
    robots = state.board.robots
    boxes = state.board.boxes
    for robot in robots:
        up = (0, 1)
        down = (0, -1)
        left = (-1, 0)
        right = (1, 0)
        moves = [up, down, left, right]
        for move in moves:
            robot_pos = tuple(x + y for x, y in zip(robot, move))
            if not sanity_check(robot_pos):
                print("Position of robot out of bound.")
                continue
            if robot_pos in state.board.obstacles:
                continue
            elif robot_pos in state.board.robots:
                continue
            elif robot_pos in boxes:
                box_pos = tuple(x + y for x, y in zip(robot_pos, move))
                if check_box_pos(box_pos):
                    bot_id = robots.index(robot)
                    box_id = boxes.index(robot_pos)
                    new_bots = [
                        robot_pos if i == bot_id else old_pos for i, old_pos in enumerate(robots)
                    ]
                    new_boxes = [
                        box_pos if i == box_id else old_pos for i, old_pos in enumerate(boxes)
                    ]
                    new_board = Board(state.board.name, state.board.width, state.board.height, new_bots, new_boxes,
                                      state.board.storage, state.board.obstacles)
                    new_state = State(new_board, state.hfn, state.f, state.depth + 1, state)
                    states.append(new_state)
                else:
                    continue
            else:
                idx = robots.index(robot)
                new_bots = [
                    robot_pos if i == idx else old_pos for i, old_pos in enumerate(robots)
                ]
                new_board = Board(state.board.name, state.board.width, state.board.height, new_bots, state.board.boxes,
                                  state.board.storage, state.board.obstacles)
                new_state = State(new_board, state.hfn, state.f, state.depth + 1, state)
                states.append(new_state)

    return states


def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    raise NotImplementedError


def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic (a function that consumes a Board and produces a numeric heuristic value)
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    raise NotImplementedError


def heuristic_basic(board):
    """
    Returns the heuristic value for the given board
    based on the Manhattan Distance Heuristic function.

    Returns the sum of the Manhattan distances between each box 
    and its closest storage point.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    raise NotImplementedError


def heuristic_advanced(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    raise NotImplementedError


def solve_puzzle(board: Board, algorithm: str, hfn):
    """
    Solve the given puzzle using the given type of algorithm.

    :param algorithm: the search algorithm
    :type algorithm: str
    :param hfn: The heuristic function
    :type hfn: Optional[Heuristic]

    :return: the path from the initial state to the goal state
    :rtype: List[State]
    """

    print("Initial board")
    board.display()

    time_start = time.time()

    if algorithm == 'a_star':
        print("Executing A* search")
        path, step = a_star(board, hfn)
    elif algorithm == 'dfs':
        print("Executing DFS")
        path, step = dfs(board)
    else:
        raise NotImplementedError

    time_end = time.time()
    time_elapsed = time_end - time_start

    if not path:

        print('No solution for this puzzle')
        return []

    else:

        print('Goal state found: ')
        path[-1].board.display()

        print('Solution is: ')

        counter = 0
        while counter < len(path):
            print(counter + 1)
            path[counter].board.display()
            print()
            counter += 1

        print('Solution cost: {}'.format(step))
        print('Time taken: {:.2f}s'.format(time_elapsed))

        return path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The file that contains the solution to the puzzle."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=['a_star', 'dfs'],
        help="The searching algorithm."
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        required=False,
        default=None,
        choices=['zero', 'basic', 'advanced'],
        help="The heuristic used for any heuristic search."
    )
    args = parser.parse_args()

    # set the heuristic function
    heuristic = heuristic_zero
    if args.heuristic == 'basic':
        heuristic = heuristic_basic
    elif args.heuristic == 'advanced':
        heuristic = heuristic_advanced

    # read the boards from the file
    board = read_from_file(args.inputfile)

    # solve the puzzles
    path = solve_puzzle(board, args.algorithm, heuristic)

    # save solution in output file
    outputfile = open(args.outputfile, "w")
    counter = 1
    for state in path:
        print(counter, file=outputfile)
        print(state.board, file=outputfile)
        counter += 1
    outputfile.close()