from board import Board
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

class CheckersEnv(AECEnv):
    metadata = {
        "name": "checkers_environment_v0",
    }

    def __init__(self):
        self.possible_agents = ["black", "red"]
        self.timestep = None
        self._agent_selector = None
        self.current_agent = None
        self.board = None
        self.legal_action_mask = None
        self.observations = {}

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self._agent_selector = AgentSelector(self.possible_agents)
        self.current_agent = self._agent_selector.next()
        self.board = Board()
        self.legal_action_mask = calculate_legal_action_mask(self.board.get_board())
        return self.get_observations()

    def step(self, action):
        agent = self.current_agent
        board = self.board
        pos, direction = action

        # Convert pos ({pos|0<=pos<18}) to 6x6 board coords
        src_coords = pos_to_coord(pos)

        direction_vec = np.array(direction.vector)

        # Always use directions relative to the current player's perspective
        if agent == "red":
            direction_vec *= -1

        # Calculate destination square
        destination_vec = np.array(src_coords) + direction_vec
        destination = tuple(destination_vec)

        # TODO: CRITICAL BUG. This doesn't work unless I negate the board for red.
        is_capture = board[destination] < 0

        board.move(src_coords, tuple(direction_vec)) 

        # Handle capture by moving an additional square in that direction
        if is_capture:
            board.move(destination, tuple(direction_vec)) 
        
        # TODO: Add check to see if the current player must move again (via
        # capture), or if we can move to the next player.
        self.current_agent = self._agent_selector.next()
        return self.get_observations()

    def render(self):
        self.board.render() 

    def get_observations(self):
        '''
        Get current set of observations as a (5,6,6) array where channel 0 is
        the board state and channels 1-4 are the legal action masks of moves
        FR,FL,BR,BL in that order.
        '''
        board = self.board.get_board().reshape((1,6,6))
        mask = calculate_legal_action_mask(board, player=self.current_agent)
        self.legal_action_mask = mask
        observations = np.concatenate([board,mask], axis=0)
        return observations

    def observation_space(self, agent):
        return MultiDiscrete([6,6,5])

    def action_space(self, agent):
        '''
        Action is a tuple of (pos, direction), where pos is the number of the
        black square (counted from top to bottom, left to right), and direction
        is an enum describing which direction to move.

        Use parse_position(pos) to get the true board co-ordinates
        '''
        return MultiDiscrete([18,4])
