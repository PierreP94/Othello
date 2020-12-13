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
    next_grid, taken = play_piece(grid, move, mark)
    score = alpha_beta(next_grid, nsteps - 1, mark, -np.Inf, np.Inf)
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
            boundaries = [[boundary_dr, -1],
                          [boundary_dl, 0],
                          [0, boundary_ur],
                          [0, boundary_ul]]
        else:
            """left part"""
            boundary_ul = y - x
            boundary_dr = Config.size - y + x
            boundaries = [[-1, boundary_dr],
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
            boundaries = [[boundary_dr, -1],
                          [-1, boundary_dl],
                          [boundary_ur, -1],
                          [0, boundary_ul]]
        else:
            """down part"""
            boundary_ul = y - x
            boundary_dr = Config.size - y + x
            boundaries = [[-1, boundary_dr],
                          [-1, boundary_dl],
                          [boundary_ur, -1],
                          [boundary_ul, 0]]
    return boundaries


def alpha_beta(node, depth, mark, alpha, beta):
    is_terminal = is_terminal_node(node)
    if depth == 0 or is_terminal:
        value = get_heuristic(node, mark)
    else:
        value = -np.Inf
        for move in valid_moves(grid, mark):
            child, child_taken = play_piece(node, move, mark)
            value = max(value, alpha_beta(child, depth - 1, 1 - mark, - alpha,
                        - beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    print("alpha: ", file=sys.stderr, flush=True)
    print(alpha, file=sys.stderr, flush=True)
    print("beta: ", file=sys.stderr, flush=True)
    print(beta, file=sys.stderr, flush=True)
    return value


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
    up = grid[1, 2:-2]
    down = grid[Config.size - 1, 2:-2]
    left = grid[2:-2, 1]
    right = grid[2:-2, Config.size - 1]
    bef_borders = np.array(list(up) + list(down) + list(left) + list(right))
    count = np.sum(bef_borders == str(mark))
    """print("bef_borders: ", file=sys.stderr, flush=True)

    print(bef_borders, file=sys.stderr, flush=True)
    """

    return count


def count_bef_corners(grid, mark):
    """Calculate number of pieces before corners"""
    win = list()
    if grid[0, 0] == '.':
        if str(1-mark) in grid[0, 2:] or '.' in grid[0, 2:]:
            win.append(grid[0, 1])
        if str(1-mark) in grid[2:, 0] or '.' in grid[2:, 0]:
            win.append(grid[1, 0])
        win.append(grid[1, 1])
    if grid[0, -1] == '.':
        if str(1-mark) in grid[0, :-2] or '.' in grid[0, :-2]:
            win.append(grid[0, -2])
        if str(1-mark) in grid[2:, -1] or '.' in grid[2:, -1]:
            win.append(grid[1, -1])
        win.append(grid[1, -2])
    if grid[-1, 0] == '.':
        if str(1-mark) in grid[-1, 2:] or '.' in grid[-1, 2:]:
            win.append(grid[-1, 1])
        if str(1-mark) in grid[:-2, 0] or '.' in grid[:-2, 0]:
            win.append(grid[0, -2])
        win.append(grid[-2, 1])
    if grid[-1, -1] == '.':
        if str(1-mark) in grid[-1, :-2] or '.' in grid[-1, :-2]:
            win.append(grid[-1, -2])
        if str(1-mark) in grid[:-2, -1] or '.' in grid[:-2, -1]:
            win.append(grid[-2, -1])
        win.append(grid[-2, -2])
    count = np.sum(win == str(mark))
    return count


# Calculates number of pieces in corners
def count_corners(grid, mark):
    up_left = grid[0, 0]
    up_right = grid[0, Config.size]
    down_left = grid[Config.size, 0]
    down_right = grid[Config.size, Config.size]
    corners = np.array([up_left, down_left, up_right, down_right])
    count = np.sum(corners == str(mark))
    return count


# Helper function for listing vald moves
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
    if grid[x, y] == '.':
        next_grid[y, x] = mark
        # horizontal +
        for a in [1, -1]:
            win_hor = grid[y, x::a]
            win_ver = grid[y::a, x]
            if len(win_hor) > 2 and win_hor[1] == str(1 - mark):
                taken = taken_in_move(win_hor, mark)
                if taken > 0:
                    next_grid[y, x:x + a * taken] = mark
                    taken_this_move += taken
            if len(win_ver) > 2 and win_ver[1] == str(1 - mark):
                taken = taken_in_move(win_ver, mark)
                if taken > 0:
                    next_grid[y:y + a * taken, x] = mark
                    taken_this_move += taken
        # diagonals
        boundaries = diag_bound(move)
        for (a, b), (bound_row, bound_col) in zip(itertools.product([1, -1], [1, -1]), boundaries):
            diag = np.array([[row for row in range(y, bound_row + a, a)],
                             [col for col in range(x, bound_col + b, b)]])
            window = grid[tuple(diag)]
            if len(window) > 2 and window[1] == str(1 - mark):
                taken = taken_in_move(window, mark)
                if taken > 0:
                    next_grid[tuple(diag[:, :taken + 1])] = mark
                    taken_this_move += taken
    return next_grid, taken_this_move


# Helper function for score_move: calculates value of heuristic for grid
def get_heuristic(grid, mark):
    op = 1 - mark
    corners = count_corners(grid, mark)
    borders = count_borders(grid, mark)
    bef_borders = count_bef_borders(grid, mark)
    bef_corners = count_bef_corners(grid, mark)
    cant_play = not bool(valid_moves(grid, mark))
    op_bef_borders = count_bef_borders(grid, op)
    op_bef_corners = count_bef_corners(grid, op)
    op_cant_play = not bool(valid_moves(grid, op))
    op_corners = count_corners(grid, op)
    op_borders = count_borders(grid, op)
    score = ((grid == "mark").sum() - (grid == str(1 - mark)).sum()
             + 10 * op_bef_borders
             - 1000 * bef_borders
             + 100000 * borders
             - 5000 * op_borders
             + 500000 * op_bef_corners
             - 10000000 * bef_corners
             + 1000000000 * corners
             - 1000000000 * op_corners
             + 100000000000 * op_cant_play
             - 10000000000 * cant_play)
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
    n_steps = 4
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


