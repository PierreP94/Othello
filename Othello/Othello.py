import sys
import numpy as np
import random
import itertools

from datetime import datetime

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

_id = int(input())  # id of your player.
board_size = int(input())


class Obs:
    board = []
    """Id of player."""
    mark = _id


class Config:
    size = board_size - 1


# Calculates score if agent drops piece in selected column
def score_move(grid, move, mark, nsteps):
    alpha = -np.Inf
    beta = np.Inf  
    next_grid = play_piece(grid, move, mark)[0]

    score = alpha_beta_nega(next_grid, nsteps - 1, mark, False, alpha, beta)
    return score


def is_terminal_node(grid):
    return np.sum(grid == '.') == 0


def diag_bound(move):
    """Allow to get boundaries for diagonals following position"""
    x = move[1]
    y = move[0]
    if y + x <= Config.size:
        boundary_ur = x + y
        boundary_dl = x + y
        if x - y >= 0:
            """top part"""
            boundary_ul = x - y
            boundary_dr = Config.size - x + y
            boundaries = [[boundary_dr, Config.size],
                          [boundary_dl, 0],
                          [0, boundary_ur],
                          [0, boundary_ul]]
        else:
            """left part"""
            boundary_ul = y - x
            boundary_dr = Config.size - y + x
            boundaries = [[Config.size, boundary_dr],
                          [boundary_dl, 0],
                          [0, boundary_ur],
                          [boundary_ul, 0]]
    else:
        boundary_ur = x + y - Config.size
        boundary_dl = x + y - Config.size
        if x - y >= 0:
            """right part"""
            boundary_ul = x - y
            boundary_dr = Config.size - x + y
            boundaries = [[boundary_dr, Config.size],
                          [Config.size, boundary_dl],
                          [boundary_ur, Config.size],
                          [0, boundary_ul]]
        else:
            """down part"""
            boundary_ul = y - x
            boundary_dr = Config.size - y + x
            boundaries = [[Config.size, boundary_dr],
                          [Config.size, boundary_dl],
                          [boundary_ur, Config.size],
                          [boundary_ul, 0]]
    return boundaries


def alpha_beta_nega(node, depth, mark, maximizing, alpha, beta):

    if depth == 0 or is_terminal_node(node):
        value = get_heuristic(node, mark)
        return value
    elif maximizing:
        alpha_node = -np.Inf
        for move in valid_moves(node, mark):
            child = play_piece(node, move, mark)[0]
            value = alpha_beta_nega(child, depth - 1, mark, False, alpha, beta)
            alpha_node = max(alpha_node, value)
            alpha = max(alpha, alpha_node)
            if alpha >= beta:
                break
        return alpha
    else:
        beta_node = np.Inf
        for move in valid_moves(node, 1-mark):
            child = play_piece(node, move, 1-mark)[0]
            value = alpha_beta_nega(child, depth - 1, mark, True, alpha, beta)
            beta_node = min(beta_node, value)
            beta = min(beta, beta_node)

            if alpha >= beta:
                break

        return beta


# Calculates number of pieces in borders
def count_borders(grid, mark):
    up = grid[0, 1:-1]
    down = grid[Config.size, 1:-1]
    left = grid[1:-1, 0]
    right = grid[1:-1, Config.size]
    borders = np.concatenate((up, down, left, right))
    count = np.sum(borders == str(mark))
    return count


def count_bef_borders(grid, mark):
    """Calculate number of pieces in the line before borders"""
    win = np.array([])
    if '.' in grid[0, 2:-2]:
        win = np.append(win, grid[1, 2:-2])
    if '.' in grid[-1, 2:-2]:
        win = np.append(win, grid[-2, 2:-2])
    if '.' in grid[2:-2, 0]:
        win = np.append(win, grid[2:-2, 1])
    if '.' in grid[2:-2, -1]:
        win = np.append(win, grid[2:-2, -2])
    return np.sum(win == str(mark))


def count_bef_corners(grid, mark):
    """Calculate number of pieces before corners"""
    win = np.array([])
    op = 1 - mark
    if grid[0, 0] == '.':
        if str(op) in grid[0, 2:] or '.' in grid[0, 2:]:
            win = np.append(win, grid[0, 1])
        if str(op) in grid[2:, 0] or '.' in grid[2:, 0]:
            win = np.append(win, grid[1, 0])
        win = np.append(win, grid[1, 1])
    if grid[0, -1] == '.':
        if str(op) in grid[0, :-2] or '.' in grid[0, :-2]:
            win = np.append(win, grid[0, -2])
        if str(op) in grid[2:, -1] or '.' in grid[2:, -1]:
            win = np.append(win, grid[1, -1])
        win = np.append(win, grid[1, -2])
    if grid[-1, 0] == '.':
        if str(op) in grid[-1, 2:] or '.' in grid[-1, 2:]:
            win = np.append(win, grid[-1, 1])
        if str(op) in grid[:-2, 0] or '.' in grid[:-2, 0]:
            win = np.append(win, grid[0, -2])
        win = np.append(win, grid[-2, 1])
    if grid[-1, -1] == '.':
        if str(op) in grid[-1, :-2] or '.' in grid[-1, :-2]:
            win = np.append(win, grid[-1, -2])
        if str(op) in grid[:-2, -1] or '.' in grid[:-2, -1]:
            win = np.append(win, grid[-2, -1])
        win = np.append(win, grid[-2, -2])
    return np.sum(win == str(mark))


# Calculates number of pieces in corners
def count_corners(grid, mark):
    up_left = grid[0, 0]
    up_right = grid[0, Config.size]
    down_left = grid[Config.size, 0]
    down_right = grid[Config.size, Config.size]
    corners = np.array([up_left, down_left, up_right, down_right])
    return np.sum(corners == str(mark))


# Helper function for listing valid moves
def valid_moves(grid, mark):
    return [move for move in itertools.product(range(Config.size + 1),
                                               range(Config.size + 1))
            if grid[move] == '.' and play_piece(grid, move, mark)[1] > 0]


# Helper for knowing how much are taken per window
def taken_in_move(window, mark):
    count = 0
    nb_taken = 0
    for win in window[1:]:
        if win == '.':
            break
        elif win != str(mark) and win != '.':
            count += 1
        elif win == str(mark):
            nb_taken = count
            break
    return nb_taken


# Helper for knowing result of a move
def play_piece(grid, move, mark):
    x = move[1]
    y = move[0]
    next_grid = grid.copy()
    taken_this_move = 0
    next_grid[y, x] = mark
    # horizontal +
    for a in [1, -1]:
        win_hor = next_grid[y, x::a]
        win_ver = next_grid[y::a, x]
        if len(win_hor) > 2 and win_hor[1] == str(1 - mark):
            taken = taken_in_move(win_hor, mark)
            if taken > 0:
                next_grid[y, x:x + a * (taken+1):a] = mark
                taken_this_move += taken

        if len(win_ver) > 2 and win_ver[1] == str(1 - mark):
            taken = taken_in_move(win_ver, mark)
            if taken > 0:
                next_grid[y:y + a * (taken+1):a, x] = mark
                taken_this_move += taken

    # diagonals
    boundaries = diag_bound(move)
    for (a, b), (bound_row, bound_col) in zip(itertools.product([1, -1],
                                              [1, -1]), boundaries):
        diag = np.array([[row for row in range(y, bound_row + a, a)],
                        [col for col in range(x, bound_col + b, b)]])
        window = next_grid[tuple(diag)]
        if len(window) > 2 and window[1] == str(1 - mark):
            taken = taken_in_move(window, mark)
            if taken > 0:
                next_grid[tuple(diag[:, :taken + 1])] = mark
                taken_this_move += taken
    return next_grid, taken_this_move


# Helper function for score_move: calculates value of heuristic for grid
def get_heuristic(grid, mark):
    list_grid = grid.ravel()
    corners = count_corners(grid, mark)
    borders = count_borders(grid, mark)
    bef_borders = count_bef_borders(grid, mark)
    bef_corners = count_bef_corners(grid, mark)
    score = (np.sum(list_grid == str(mark))
             - 100 * bef_borders
             + 1000 * borders
             - 100000 * bef_corners
             + 10000000 * corners)
    return score


def convert_move(move):
    y = move[0] + 1
    x = move[1]
    return "abcdefgh"[x] + str(y)


def agent(obs):
    # Convert the board to a 2D grid ()
    # Get list of valid moves
    v_moves = valid_moves(grid, obs.mark)
    print("v_moves: ", file=sys.stderr, flush=True)
    print(v_moves, file=sys.stderr, flush=True)

    if len(v_moves) == 4:
        n_steps = 3
    elif len(v_moves) <= 3:
        n_steps = 4
    else:
        n_steps = 2
    """ Use the heuristic to assign a score
     to each possible board in the next turn """
    scores = dict(zip(v_moves, [score_move(grid, move, obs.mark, n_steps)
                                for move in v_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    print("scores: ", file=sys.stderr, flush=True)
    print(scores, file=sys.stderr, flush=True)
    print(max(scores.values()), file=sys.stderr, flush=True)

    max_moves = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    print("max_moves :", file=sys.stderr, flush=True)

    print(max_moves, file=sys.stderr, flush=True)
    return random.choice(max_moves)


# def main():
# game loop
while True:
    Obs.board = []
    for i in range(board_size):
        line = input()  # rows from top to bottom (viewer perspective).
        Obs.board.extend(line)
    action_count = int(input())  # number of legal actions for this turn.
    grid = np.asarray(Obs.board).reshape(Config.size + 1, Config.size + 1)
    print(grid, file=sys.stderr, flush=True)

    for i in range(action_count):
        action = input()  # the action
    temps1 = datetime.now()
    if action_count > 0:
        move_to_play = agent(Obs)
        order = convert_move(move_to_play)
        print(move_to_play, file=sys.stderr, flush=True)
        temps2 = datetime.now()
        delay = temps2 - temps1
        print("delay : " + str(delay), file=sys.stderr, flush=True)

        # Write an action using print
        # To debug: print("Debug messages...", file=sys.stderr, flush=True)

        # a-h1-8
        print(order)

# if __name__ == '__main__':
#    main()

