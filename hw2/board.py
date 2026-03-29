import numpy as np

class Board():
    def __init__(self):
        pawn_row = np.array([[0,1]*3], dtype=np.int8)

        black_pawns = np.concat([
            np.zeros((4,6), dtype=np.bool),
            pawn_row, 
            np.roll(pawn_row,1)
        ], dtype=np.int8)

        red_pawns = np.roll(black_pawns, 2, axis=0)
        self._board = np.concat([
            black_pawns,
            red_pawns,
            # Black kings
            np.zeros_like(black_pawns),
            # Red kings
            np.zeros_like(red_pawns),
        ]).reshape((4,6,6))
             
    def __getitem__(self, position):
        '''
        Get the 4-channel depth vector at position.
        Throws an IndexError if the position is out of range.
        '''
        row, col = position
        if not (0 <= row < self._board.shape[1]) or not (0 <= col < self._board.shape[2]):
            raise IndexError(f"Position {position} is out of bounds")
        return self._board[:, row, col]

    def __setitem__(self, position, value):
        '''
        Sets the 4-channel depth vector at position.
        '''
        row, col = position
        if not (0 <= row < self._board.shape[1]) or not (0 <= col < self._board.shape[2]):
            raise IndexError(f"Position {position} is out of bounds")
        self._board[:, row, col] = value
        
    def __eq__(self, other):
        '''
        Returns whether the two boards are equal.
        '''
        if not isinstance(other, Board):
            return False
        return np.array_equal(self._board, other.get_board())

    def get_board(self):
        '''
        Return an immutable copy of the board.
        '''
        result = self._board.copy()
        result.flags.writeable = False
        return result
    
    def render(self):
        '''
        Renders the board, using the 4 channels to determine the piece.
        '''
        print("-" * 19)
        for row in range(self._board.shape[1]):
            line_chars = []
            for col in range(self._board.shape[2]):
                cell = self._board[:, row, col]
                # Check which channel is active
                if cell[0] == 1:
                    line_chars.append("B ")
                elif cell[1] == 1:
                    line_chars.append("R ")
                elif cell[2] == 1:
                    line_chars.append("BK")
                elif cell[3] == 1:
                    line_chars.append("RK")
                else:
                    line_chars.append("  ")
            
            line = "|".join(line_chars)
            print(f"|{line}|")
            print("-" * 19)
       
    def move(self, src, dir):
        '''
        Move the piece at src (row, col) in the direction of dir (dir_row, dir_col).
        Moves the entire channel vector. Overwrites the destination.
        '''
        row, col = src
        dir_row, dir_col = dir
        
        if not (0 <= row < self._board.shape[1]) or not (0 <= col < self._board.shape[2]):
            raise IndexError(f"Source position {src} out of bounds")
        
        dest_row, dest_col = row + dir_row, col + dir_col
        if not (0 <= dest_row < self._board.shape[1]) or not (0 <= dest_col < self._board.shape[2]):
            raise IndexError(f"Destination position {(dest_row, dest_col)} out of bounds")

        # Move the depth vector to destination, zero out the source
        self._board[:, dest_row, dest_col] = self._board[:, row, col]
        self._board[:, row, col] = 0

