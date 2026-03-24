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
        self._is_flipped = False

    def __getitem__(self, position):
        '''
        Get the item at position using the current board orientation.
        Throws an IndexError if the position is out of range
        '''
        row, col = position
        board = self._board
        if not (0 <= row < len(board)) or not (0 <= col < len(board[0])):
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
        Returns whether the two boards are elementwise equal in the standard
        orientation. Performs flips to bring boards into the same orientation
        before comparison, restoring the orientation afterwars if necessary
        '''
        should_flip_self = self._is_flipped
        should_flip_other = other.get_orientation() == "red"

        if should_flip_self:
            self.flip_board()
        if should_flip_other:
            other.flip_board()

        is_equal = np.equal(self._board, other.get_board())

        # Restore original orientations
        if should_flip_self:
            self.flip_board()
        if should_flip_other:
            other.flip_board()

        return is_equal

    def get_board(self):
        '''
        Return an immutable copy of the board.
        '''
        result = self._board.copy()
        result.flags.writeable = False
        return result

    def get_orientation(self):
        return "red" if self._is_flipped else "black"

    def flip_board(self):
        '''
        Given a board, rotate it 180 degrees and multiply values by -1.
        This ensures that the current agent always sees the board with
        1 representing their pieces, and forward being in the decreasing
        row dimension.
        '''
        J = np.identity(6)[:, ::-1]
        self._is_flipped = True
        self._board = -1 * J @ self._board @ J
        return self._board

    def render(self):
        '''
        Renders the board, always using black's perspective as the orientation.
        '''
        print("-"*13)
        should_flip = self._is_flipped
        if should_flip:
            self.flip_board()

        for row in self._board:  # ty:ignore[not-iterable]
            squares = map(str, row)
            line = "|".join(squares).replace("-1", "R").replace("1", "B").replace("0", " ")
            print(f"|{line}|")
            print("-"*13)

        # Restore the original orientation
        if should_flip:
            self.flip_board()

    def move(self, row, col, dir_row, dir_col):
        '''
        Move the piece at (row, col) in the direction of (dir_row, dir_col), performing bounds
        checks. If another piece is in the destination square, the piece is 
        overwritten without regard to the legality of the move.
        '''
        board = self._board
        if not (0 <= row < len(board)) or not (0 <= col < len(board[0])):
            raise IndexError(f"Source position ({row,col}) out of bounds")
        
        dest_row, dest_col = row + dir_row, col + dir_col
        if not (0 <= dest_row < len(board)) or not (0 <= dest_col < len(board[0])):
            raise IndexError(f"Destination position ({dest_row,dest_col}) out of bounds")

        board[(dest_row, dest_col)] = board[(row, col)]
        board[(row,col)] = 0
