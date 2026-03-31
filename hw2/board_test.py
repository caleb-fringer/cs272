import unittest
import numpy as np
from board import Board

class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board()

    def test_initialization(self):
        # Check shapes
        self.assertEqual(self.board.get_board().shape, (4, 6, 6))
        
        # Check a known Red Pawn position (Channel 1)
        np.testing.assert_array_equal(self.board[0, 1], [0, 1, 0, 0])
        
        # Check a known Black Pawn position (Channel 0)
        np.testing.assert_array_equal(self.board[5, 0], [1, 0, 0, 0])
        
        # Check an empty middle square
        np.testing.assert_array_equal(self.board[3, 3], [0, 0, 0, 0])

    def test_getitem_out_of_bounds(self):
        with self.assertRaises(IndexError):
            _ = self.board[-1, 0]
        with self.assertRaises(IndexError):
            _ = self.board[6, 6]

    def test_setitem(self):
        # Make a Red King (Channel 3)
        self.board[2, 2] = [0, 0, 0, 1]
        np.testing.assert_array_equal(self.board[2, 2], [0, 0, 0, 1])

    def test_setitem_out_of_bounds(self):
        with self.assertRaises(IndexError):
            self.board[6, 0] = [1, 0, 0, 0]

    def test_equality(self):
        board2 = Board()
        self.assertTrue(self.board == board2)
        
        board2[2, 2] = [1, 0, 0, 0]
        self.assertFalse(self.board == board2)

    def test_move_valid(self):
        # Move Black Pawn from (4, 1) Up-Right to (3, 2)
        self.board.move((4, 1), (-1, 1))
        
        # Source should now be empty
        np.testing.assert_array_equal(self.board[4, 1], [0, 0, 0, 0])
        # Dest should now have a Black Pawn
        np.testing.assert_array_equal(self.board[3, 2], [1, 0, 0, 0])

    def test_move_out_of_bounds_source(self):
        with self.assertRaises(IndexError):
            self.board.move((6, 0), (1, 1))

    def test_move_out_of_bounds_dest(self):
        # Try to move off the top of the board
        with self.assertRaises(IndexError):
            self.board.move((0, 1), (-1, -1))

if __name__ == '__main__':
    unittest.main()
