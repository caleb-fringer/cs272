import numpy as np
from pettingzoo.utils import AgentSelector
from os import path
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import AECEnv
from enum import Enum

class Direction(Enum):
    FR = 0
    FL = 1
    BR = 2 
    BL = 3 
         
    @property
    def vector(self):
        lookup = [
            (-1, 1),
            (-1,-1),
            ( 1, 1),
            ( 1,-1),
        ]
        return np.array(lookup[self.value])

def calculate_legal_action_mask(board):
    '''
    Calculates the legal action mask for a board.
    '''
    fwd = np.array([np.zeros((6,6)), board]).max(axis=0) > 0
    back = np.array([np.ones((6,6)), board]).max(axis=0) - 1 > 0
    return np.array((fwd, back))

def pos_to_coord(pos):
    '''
    Convert a position (the count of a black square, counted left to right, top to
    bottom) into the actual co-ordinates of a square on the 6x6 gameboard.
    '''
    row = pos // 3
    col = 2*(pos % 3)
    # Offset to the right 1 for even rows.
    if row % 2 == 0:
        col += 1
    return (row, col)

def coord_to_pos(row, col):
    '''
    Inverse of pos_to_cord
    '''
    return row * 3 + col

def flip_board(board):
    '''
    Given a board, rotate it 180 degrees and multiply values by -1.
    This ensures that the current agent always sees the board with
    1 representing their pieces, and forward being in the decreasing
    row dimension.
    '''
    J = np.identity(6)[:, ::-1]
    return -1 * J @ board @ J

class CheckersEnv(AECEnv):
    metadata = {
        "name": "checkers_environment_v0",
    }

    def __init__(self):
        self.possible_agents = ["black", "red"]
        self.timestep = None
        self._agent_selector = None
        self.current_agent = None
        self.observations = {}

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self._agent_selector = AgentSelector(self.possible_agents)
        self.current_agent = self._agent_selector.next()
        self.observations["board"] = np.array([[ 0,-1, 0,-1, 0,-1],
                                               [-1, 0,-1, 0,-1, 0],
                                               [ 0, 0, 0, 0, 0, 0],
                                               [ 0, 0, 0, 0, 0, 0],
                                               [ 0, 1, 0, 1, 0, 1],
                                               [ 1, 0, 1, 0, 1, 0]])
        self.observations["legal_action_mask"] = calculate_legal_action_mask(self.observations["board"])
        return self.observations

    def step(self, action):
        agent = self.current_agent
        pos, direction = action
        coords = pos_to_coord(pos)
        destination = tuple(np.array(coords) + np.array(direction.vector))
        if self.current_agent == "red":
            destination *= -1
        self.observations["board"][destination] = self.observations["board"][coords]
        self.observations["board"][coords] = 0

        self.observations["legal_action_mask"] = calculate_legal_action_mask(self.observations["board"])
        self.current_agent = self._agent_selector.next()
        # TODO: Determine if its the next players turn
        return self.observations["board"]

    def render(self):
        print("-"*13)
        for row in self.observations["board"]:  # ty:ignore[not-iterable]
            squares = map(str, row)
            line = "|".join(squares).replace("-1", "R").replace("1", "B").replace("0", " ")
            print(f"|{line}|")
            print("-"*13)

    def observation_space(self, agent):
        return MultiDiscrete([6,6,3])

    def action_space(self, agent):
        '''
        Action is a tuple of (pos, direction), where pos is the number of the
        black square (counted from top to bottom, left to right), and direction
        is an enum describing which direction to move.

        Use parse_position(pos) to get the true board co-ordinates
        '''
        return MultiDiscrete([18,4])

