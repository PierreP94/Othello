import sys
import math
import numpy as np
import random
import itertools

from datetime import datetime

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

_id = int(input())  # id of your player.
board_size = int(input())
class obs :
    board = []
    mark = _id

class config :
    columns = board_size

# Calculates score if agent drops piece in selected column
def score_move(grid, move, mark, config, nsteps):
    next_grid, taken = play_piece(grid, move, mark, config)
    score = minimax(next_grid, nsteps-1, False, mark, config)
    return score

def is_terminal_node(grid, config):
    return np.sum(grid == '.') == 0

def minimax(node, depth, maximizingPlayer, mark, config):
    is_terminal = is_terminal_node(node, config)
    if depth == 0 or is_terminal:
        value = get_heuristic(node, mark, config)
    elif maximizingPlayer:
        v_moves = valid_moves(node, mark, config)
        value = -np.Inf
        for move in v_moves:
            child, child_taken = play_piece(node, move, mark, config)
            value = max(value, minimax(child, depth-1, False, mark, config))
            print(v_moves, file=sys.stderr, flush=True)
    else:
        op_v_moves = valid_moves(node, 1-mark, config)
        value = np.Inf
        for move in op_v_moves:
            child, child_taken = play_piece(node, move, 1-mark, config)
            value = min(value, minimax(child, depth-1, True, mark, config))
    return value

# Calculates number of pieces in borders
def count_borders(grid, mark, config):
    up = grid[0,1:-1]
    down = grid[config.columns-1,1:]
    left = grid[1:-1,0]
    right = grid[1:-1,config.columns-1]
    borders = np.concatenate((up, down, left, right))
    count = np.sum(borders == str(mark))
    return count

# Calculates number of pieces in corners
def count_corners(grid, mark, config):
    up_left = grid[0,0]
    up_right = grid[0,-1]
    down_left = grid[-1,0]
    down_right = grid[-1,-1]
    corners = np.array([up_left, down_left, up_right, down_right])
    count = np.sum(corners == str(mark))
    return count

# Helper function for listing vald moves
def valid_moves(grid, mark, config):
    return [move
            for move in itertools.product(range(config.columns), range(confg.columns))
            if grid[move] == '.' and play_piece(grid, move, mark, config)[1] > 0]


# Helper for knowing how much are taken per window
def taken_in_move(window, mark, config):
    count = 0
    nb_taken = 0
    for win in window[1:]:
        if win == '.' : break
        elif win != str(mark) and win != '.' :
            count += 1
        elif win == str(mark) :
            nb_taken = count
            break
    return nb_taken

# Helper for knowing result of a move
def play_piece(grid, move, mark, config) :
    x = move[1]
    y = move[0]
    next_grid = grid.copy()
    taken_this_move = 0
    # horizontal +
    window = list(grid[y, x::])
    if len(window) > 2 and window[1] == str(1 - mark):
        taken = taken_in_move(window, mark, config)
        if taken > 0:
            next_grid[y, x:x+taken] = mark
            taken_this_move += taken
    # horizontal -
    window = list(grid[y, x::-1])
    if len(window) > 2 :
        if window[1] == str(1 - mark) :
            taken = taken_in_move(window, mark, config)
            if taken > 0:
                next_grid[y, x:x-taken:-1] = mark
                taken_this_move += taken
    # vertical +
    window = list(grid[y::, x])
    if len(window) > 2 :
        if window[1] == str(1 - mark) :
            taken = taken_in_move(window, mark, config)
            if taken > 0:
                next_grid[y:y+taken,x] = mark
                taken_this_move += taken
    # vertical -
    window = list(grid[y::-1,x ])
    if len(window) > 2 :
        if window[1] == str(1 - mark):
            taken = taken_in_move(window, mark, config)
            if taken > 0:
                next_grid[y:y-taken+1:-1,x] = mark
                taken_this_move += taken
    # diagonal up right

    for (a, b), min_ for zip(itertools.product([1, -1], [1, -1]),
                             [[config.columns - y - 1, config.columns - x - 1]
                              [config.columns - y - 1, x]
                              [y, config.columns - x - 1]
                              [y, x]]):
        diag = [[y + a * i for i in range(min_)],
                [x + b * i for i in range(min_)]]
        window = grid[diag]
        if len(window) > 2 and window[1] == str(1 - mark) :
            taken = taken_in_move(window, mark, config)
            if taken > 0:
                next_grid[diag[taken]] = mark
                taken_this_move += taken

    return next_grid, taken_this_move

# Helper function for score_move: calculates value of heuristic for grid
def get_heuristic(grid, mark, config):
    corners = count_corners(grid, mark, config)
    borders = count_borders(grid, mark, config)
    op_cant_play = not bool(valid_moves(grid, 1-mark, config))
    op_corners = count_corners(grid, 1-mark, config)
    op_borders = count_borders(grid, 1-mark, config)
    cant_play = not bool(valid_moves(grid, mark, config))
    score = (grid == "mark").sum() + 10*(borders-op_borders) + 100*(corners-op_corners) + 1000*(cant_play - op_cant_play)
    return score

def convert_move(move) :
    y = move[0] + 1
    x = move[1]
    return "abcdefgh"[x] + str(y)

N_STEPS = 2
def agent(obs, config):
    # Convert the board to a 2D grid ()
    # Get list of valid moves
    v_moves = valid_moves(grid, obs.mark, config)
    print("v_moves: ", file=sys.stderr, flush=True)
    print(v_moves, file=sys.stderr, flush=True)

    # Use the heuristic to assign a score to each possible board in the next turn
    scores = dict(zip(v_moves, [score_move(grid, move, obs.mark, config, N_STEPS) for move in v_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    print("scores: ", file=sys.stderr, flush=True)
    print(scores, file=sys.stderr, flush=True)

    max_moves = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    print("max_moves :", file=sys.stderr, flush=True)

    print(max_moves, file=sys.stderr, flush=True)

    return random.choice(max_moves)

def main():
    # game loop
    while True:
        obs.board = []
        for i in range(board_size):
            line = input()  # rows from top to bottom (viewer perspective).
            obs.board.extend(line)
        action_count = int(input())  # number of legal actions for this turn.
        grid = np.asarray(obs.board).reshape(config.columns, config.columns)


        for i in range(action_count):
            action = input()  # the action
        print(np.sum(grid == '.'), file=sys.stderr, flush=True)

        temps1 = datetime.now()
        move_to_play = agent(obs, config)
        order = convert_move(move_to_play)
        print(move_to_play, file=sys.stderr, flush=True)
        temps2 = datetime.now()
        delay = temps2 - temps1
        print(delay, file=sys.stderr, flush=True)


        # Write an action using print
        # To debug: print("Debug messages...", file=sys.stderr, flush=True)


        # a-h1-8
        print(order)


if __name__ == '__main__':
    main()
