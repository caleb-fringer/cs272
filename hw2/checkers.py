from board import Board
import numpy as np
from pettingzoo.utils import AgentSelector
from os import path
from gymnasium.spaces import MultiDiscrete, Dict, MultiBinary
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

def calculate_legal_action_mask(board, player="black"):
    '''
    Calculates a (4, 6, 6) mask of legal moves for the given player.
    Channels correspond to the Direction enum relative to the player:
    0: Forward-Right (FR)
    1: Forward-Left  (FL)
    2: Backward-Right(BR)
    3: Backward-Left (BL)
    '''
    mask = np.zeros((4, 6, 6), dtype=np.int8)
    
    # 1. Determine player-specific channels and relative movement vectors
    pawn_c, king_c = (1,3) if player == "red" else (0,2)

    # 2. Determine valid destinations
    # A destination is valid if it is empty OR occupied by an enemy.
    # This is mathematically equivalent to: NOT occupied by a friendly piece.
    friendly_pieces = board[pawn_c] | board[king_c]
    valid_destinations = 1 - friendly_pieces 
    
    # 3. Calculate masks for each direction
    for direction in Direction:
        dir_idx = direction.value
        (dr, dc) = -1*direction.vector if player == "red" else direction.vector
            
        # Identify which pieces are allowed to move in this direction
        # Kings can move any direction; Pawns can only move Forward (channels 0 and 1)
        capable_pieces = np.copy(board[king_c])
        if dir_idx in (0, 1):
            capable_pieces |= board[pawn_c]
            
        # Define matrix slices based on the direction vector
        # (This automatically handles the board boundaries)
        src_r = slice(1, None) if dr < 0 else slice(None, -1)
        dst_r = slice(None, -1) if dr < 0 else slice(1, None)
        
        src_c = slice(1, None) if dc < 0 else slice(None, -1)
        dst_c = slice(None, -1) if dc < 0 else slice(1, None)
        
        # A move is valid if a capable piece exists at the source AND 
        # the corresponding destination is valid
        valid_moves = capable_pieces[src_r, src_c] & valid_destinations[dst_r, dst_c]
        
        # Map the valid moves back into the source locations on the mask
        mask[dir_idx, src_r, src_c] = valid_moves
        
    return mask

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

        # Check if the move results in a capture
        b = board.get_board()
        enemy_pawns_chan, enemy_kings_chan = (0,2) if agent == "red" else (1,3)
        enemy_locations = b[enemy_pawns_chan] | b[enemy_kings_chan]
        is_capture = enemy_locations[destination] == 1

        board.move(src_coords, tuple(direction_vec)) 

        # Handle capture by moving an additional square in that direction
        if is_capture:
            board.move(destination, tuple(direction_vec)) 
            # Update destination so that the pawn promotion check is correct
            destination_vec += direction_vec
            destination = tuple(destination_vec)
        
        # Handle pawn promotion
        dest_row = destination[0]
        should_promote = dest_row == 5 if agent == "red" else dest_row == 0
        if should_promote:
            board[destination] = np.array([0,0,0,1]) if agent == "red" else np.array([0,0,1,0])

        # TODO: Add check to see if the current player must move again (via
        # capture), or if we can move to the next player.
        self.current_agent = self._agent_selector.next()
        return self.get_observations()

    def render(self):
        self.board.render() 

    def get_observations(self):
        '''
        Get current set of observations & corresponding legal_action_mask,
        each as a (4,6,6) MultiBinary matrix
        '''
        board = self.board.get_board()
        mask = calculate_legal_action_mask(board, player=self.current_agent)
        self.legal_action_mask = mask
        return {
            "observations": board,
            "legal_action_mask": mask,
        }

    def observation_space(self, agent):
        return Dict({
            "observations": MultiBinary([4,6,6]),
            "legal_action_mask": MultiBinary([4,6,6])
        })

    def action_space(self, agent):
        '''
        Action is a tuple of (pos, direction), where pos is the number of the
        black square (counted from top to bottom, left to right), and direction
        is an enum describing which direction to move.

        Use parse_position(pos) to get the true board co-ordinates
        '''
        return MultiDiscrete([18,4])
