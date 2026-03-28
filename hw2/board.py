import numpy as np

class Board():
    def __init__(self):
        self._board = np.array([[ 0,-1, 0,-1, 0,-1],
                                [-1, 0,-1, 0,-1, 0],
                                [ 0, 0, 0, 0, 0, 0],
                                [ 0, 0, 0, 0, 0, 0],
                                [ 0, 1, 0, 1, 0, 1],
                                [ 1, 0, 1, 0, 1, 0]],
                               dtype=np.int8)

    def __getitem__(self, position):
        '''
        Get the item at position using the current board orientation.
        Throws an IndexError if the position is out of range
        '''
        row, col = position
        board = self._board
        if not (0 <= row < board.shape[0]) or not (0 <= col < board.shape[1]):
            raise IndexError(f"Position {position} is out of bounds")
        return board[position]

    def __setitem__(self, position, value):
        row, col = position
        board = self._board
        if row < 0 or col < 0 or row <= len(board) or col <= len(board[0]):
            raise IndexError(f"Position {position} is out of bounds")
        board[position] = value
        
    def __equals__(self, other):
        '''
        Returns whether the two boards are elementwise equal.
        '''
        return np.equal(self._board, other.get_board())

    def get_board(self):
        '''
        Return an immutable copy of the board.
        '''
        result = self._board.copy()
        result.flags.writeable = False
        return result

    
    def render(self):
        '''
        Renders the board, always using black's perspective as the orientation.
        '''
        print("-"*13)
        
        for row in self._board:
            squares = map(str, row)
            line = "|".join(squares).replace("-1", "R").replace("1", "B").replace("0", " ")
            print(f"|{line}|")
            print("-"*13)
       
    def move(self, src, dir):
        '''
        Move the piece at (row, col) in the direction of (dir_row, dir_col), performing bounds
        checks. If another piece is in the destination square, the piece is 
        overwritten without regard to the legality of the move.
        '''
        row, col = src
        dir_row, dir_col = dir
        board = self._board
        if not (0 <= row < board.shape[0]) or not (0 <= col < board.shape[1]):
            raise IndexError(f"Source position ({row,col}) out of bounds")
        
        dest_row, dest_col = row + dir_row, col + dir_col
        if not (0 <= dest_row < len(board)) or not (0 <= dest_col < len(board[0])):
            raise IndexError(f"Destination position ({dest_row,dest_col}) out of bounds")

        board[(dest_row, dest_col)] = board[(row, col)]
        board[(row,col)] = 0
