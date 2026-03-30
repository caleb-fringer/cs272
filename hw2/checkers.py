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

def calculate_legal_action_mask(board, player="black", active_piece=None):
    '''
    Calculates a (8, 6, 6) mask of legal moves for the given player.
    Channels 0-3: Regular 1-step moves (FR, FL, BR, BL)
    Channels 4-7: Capture 2-step moves (FR, FL, BR, BL)
    '''
    mask = np.zeros((8, 6, 6), dtype=np.int8)
    
    # 1. Determine player-specific channels
    pawn_c, king_c = (1,3) if player == "red" else (0,2)
    enemy_pawn_c, enemy_king_c = (0,2) if player == "red" else (1,3)

    # 2. Map occupied spaces
    friendly_pieces = board[pawn_c] | board[king_c]
    enemy_pieces = board[enemy_pawn_c] | board[enemy_king_c]
    empty_squares = 1 - (friendly_pieces | enemy_pieces) 
    
    # 3. Calculate masks for each direction
    for direction in Direction:
        dir_idx = direction.value
        (dr, dc) = -1*direction.vector if player == "red" else direction.vector
            
        # Identify which pieces are allowed to move in this direction
        # Kings can move any direction; Pawns can only move Forward (channels 0 and 1)
        capable_pieces = np.copy(board[king_c])
        if dir_idx in (0, 1):
            capable_pieces |= board[pawn_c]
            
        # If we are in the middle of a multi-jump sequence, ONLY the active piece can move
        if active_piece is not None:
            active_mask = np.zeros((6, 6), dtype=np.int8)
            active_mask[active_piece] = 1
            capable_pieces &= active_mask

        # --- REGULAR MOVES (1-Step) ---
        src_r_1 = slice(1, None) if dr < 0 else slice(None, -1)
        dst_r_1 = slice(None, -1) if dr < 0 else slice(1, None)
        src_c_1 = slice(1, None) if dc < 0 else slice(None, -1)
        dst_c_1 = slice(None, -1) if dc < 0 else slice(1, None)
        
        valid_moves = capable_pieces[src_r_1, src_c_1] & empty_squares[dst_r_1, dst_c_1]
        mask[dir_idx, src_r_1, src_c_1] = valid_moves
        
        # --- CAPTURE MOVES (2-Step) ---
        # Define 2-step source, intermediate (enemy), and destination (empty) slices
        if dr < 0:
            src_r_2, mid_r, dst_r_2 = slice(2, None), slice(1, -1), slice(None, -2)
        else:
            src_r_2, mid_r, dst_r_2 = slice(None, -2), slice(1, -1), slice(2, None)
            
        if dc < 0:
            src_c_2, mid_c, dst_c_2 = slice(2, None), slice(1, -1), slice(None, -2)
        else:
            src_c_2, mid_c, dst_c_2 = slice(None, -2), slice(1, -1), slice(2, None)
            
        # A capture requires: Capable piece at Source, Enemy at Mid, Empty at Dest
        valid_captures = (capable_pieces[src_r_2, src_c_2] & 
                          enemy_pieces[mid_r, mid_c] & 
                          empty_squares[dst_r_2, dst_c_2])
        mask[dir_idx + 4, src_r_2, src_c_2] = valid_captures
        
    # 4. Enforce Forced Captures: If ANY capture is possible, clear all regular moves
    if np.any(mask[4:8]):
        mask[0:4] = 0
        
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
        pos, action_channel = action # action_channel is 0-7

        # Convert pos ({pos|0<=pos<18}) to 6x6 board coords
        src_coords = pos_to_coord(pos)

        # 2. PERFORM LEGAL MOVE
        is_capture = action_channel >= 4
        dir_idx = action_channel % 4
        direction = Direction(dir_idx)
        direction_vec = np.array(direction.vector)

        # Always use directions relative to the current player's perspective
        
        if agent == "red":
            direction_vec *= -1

        if not is_capture:
            # Move 1 step
            board.move(src_coords, tuple(direction_vec))
            destination = tuple(np.array(src_coords) + direction_vec)
        else:
            # Move 2 steps for jump
            board.move(src_coords, tuple(direction_vec * 2))
            destination = tuple(np.array(src_coords) + direction_vec * 2)
            
            # Remove the captured enemy piece
            captured_pos = tuple(np.array(src_coords) + direction_vec)
            board[captured_pos] = 0

        # 3. HANDLE PAWN PROMOTION
        dest_row = destination[0]
        promoted = False
        should_promote = (dest_row == 5 and agent == "red") or (dest_row == 0 and agent == "black")
        
        if should_promote:
            current_piece = board[destination]
            # Verify piece is a pawn before promoting
            if agent == "red" and current_piece[1] == 1:
                board[destination] = np.array([0,0,0,1])
                promoted = True
            elif agent == "black" and current_piece[0] == 1:
                board[destination] = np.array([0,0,1,0])
                promoted = True

        # 5. HANDLE MULTI-JUMPS & TURN PROGRESSION
        # Standard Checkers Rules: A turn does not end if a piece can jump again, 
        # UNLESS that piece just promoted to a King, which ends the turn immediately.
        if is_capture and not promoted:
            # Re-evaluate mask strictly for the piece that just jumped
            new_mask = calculate_legal_action_mask(b, player=agent, active_piece=destination)
            
            if np.any(new_mask[4:8]):
                # Captures are available! Update mask, DO NOT switch agent
                self.legal_action_mask = new_mask
                return self.get_observations()

        # If we reach here, the turn is over
        self.current_agent = self._agent_selector.next()
        self.legal_action_mask = calculate_legal_action_mask(board.get_board(), player=self.current_agent)
        
        return self.get_observations()

    def render(self):
        self.board.render() 

    def get_observations(self):
        '''
        Get current set of observations & corresponding legal_action_mask.
        The mask is calculated in step() or reset() to accommodate mid-turn multi-jumps.
        '''
        return {
            "observations": self.board.get_board(),
            "legal_action_mask": self.legal_action_mask,
        }

    def observation_space(self, agent):
        return Dict({
            "observations": MultiBinary([4, 6, 6]),
            "legal_action_mask": MultiBinary([8, 6, 6]) # Updated to 8 Channels
        })

    def action_space(self, agent):
        '''
        pos: Square number (0 to 17)
        direction: 0-3 for regular moves (FR, FL, BR, BL), 4-7 for capture moves.
        '''
        return MultiDiscrete([18, 8])
