import unittest
import numpy as np
from board import Board

class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board()

    def test_initial_setup(self):
        """Check if pieces are in correct starting positions."""
        # Top-left (0,0) should be empty, (0,1) should be enemy (-1)
        self.assertEqual(self.board[0, 1], -1)
        # Bottom-right (5,5) should be empty, (5,4) should be friendly (1)
        self.assertEqual(self.board[5, 4], 1)

    def test_bounds_checking_get_set(self):
        """Verify that out-of-bounds access raises IndexError."""
        invalid_coords = [(-1, 0), (0, -1), (6, 0), (0, 6), (10, 10)]
        for r, c in invalid_coords:
            with self.subTest(coord=(r, c)):
                with self.assertRaises(IndexError):
                    _ = self.board[r, c]
                with self.assertRaises(IndexError):
                    self.board[r, c] = 1

    def test_move_valid(self):
        """Test a basic valid move and state update."""
        # Move piece from (5,0) to (4,1)
        # Note: (5,0) is a '1' in your init, (4,1) is '1'. 
        # Let's move (5,0) to an empty square (3,0) for testing
        self.board.move(5, 0, -2, 0)
        self.assertEqual(self.board[3, 0], 1)
        self.assertEqual(self.board[5, 0], 0)

    def test_move_out_of_bounds(self):
        """Verify move raises IndexError if destination is off-board."""
        with self.assertRaises(IndexError):
            self.board.move(0, 1, -1, 0) # Moving enemy off top edge

    def test_immutability_of_get_board(self):
        """Ensure the copy returned by get_board cannot be modified."""
        snapshot = self.board.get_board()
        with self.assertRaises(ValueError):
            snapshot[0, 0] = 99

    def test_flip_board_logic(self):
        """Verify that flipping rotates the board and inverts values."""
        # Save a value from the bottom-left corner
        original_val = self.board[5, 0] # Should be 1
        
        # In your current code, flip_board returns the board but doesn't save it.
        # Assuming the bug is fixed, we test if orientation changes:
        flipped_array = self.board.flip_board()
        
        # After flip, the old (5,0) is now at (0,5) and sign-flipped
        # (1 becomes -1)
        self.assertEqual(flipped_array[0, 5], -original_val)
        self.assertEqual(self.board.get_orientation(), "red")

if __name__ == "__main__":
    unittest.main()
