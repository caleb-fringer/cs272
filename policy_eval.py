from enum import Enum


class Action(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


def policy_eval(policy, board, theta=0.001):
    k = 0
    while True:
        delta = 0
        for i in range(len(board)):
            for j in range(len(board[i])):
                s = (i, j)
                old_v = v(s, board)

                successors = [(*map_state_action(s, a), a) for a in Action]
                new_v = sum(policy(a, s)*(r+v(s_prime, board))
                            for s_prime, r, a in successors)

                delta = max(delta, abs(new_v - old_v))
                board[i][j] = new_v
        k += 1
        if delta < theta:
            print(f"Policy evaluation concluded after {k} iterations.")
            return


def equally_likely_policy(a, s):
    if s == (0, 0) or s == (3, 3):
        return 0
    else:
        return 0.25


def v(s, board):
    row, col = s
    return board[row][col]


def q(s, a, board):
    row, col = s
    successor, r = map_state_action(s, a)
    return r + v(successor, board)


def clamp(pos, dir):
    return max(0, min(pos + dir, 3))


def map_state_action(s, a):
    '''
    Map a state, action pair to a s_prime, r.
    Accounts for terminal states and bounds checks
    '''
    # Terminal states
    if s == (0, 0) or s == (3, 3):
        return s, 0

    # Update state
    s_prime = tuple(map(clamp, s, a.value))

    return s_prime, -1


def print_board(board):
    for row in board:
        print(" ".join(f"{val:8.2f}" for val in row))


board = [[0]*4 for i in range(4)]
policy_eval(equally_likely_policy, board, theta=0.0001)
print_board(board)

# Exercise 4.1
q((2, 3), Action.DOWN, board)  # -1
q((1, 3), Action.DOWN, board)  # -14.945296722711777
