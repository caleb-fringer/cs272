from collections import defaultdict
import random
from typing import List, Tuple

class StudentAgent:
    def __init__(self):
        """
        Initialize your internal state here.
        """

    def get_action(self, x: int, y: int, history: List[Tuple[int, int, int, int, int, float]]) -> int:
        """
        Decide the next action to take.

        Args:
            (x, y): defines the agent's current state
            history (List): A list of past episodes. Each episode is a list of steps. 
                            Each step is (old_x, old_y, action, new_x, new_y, reward).

        Returns:
            int: The action to take (0, 1, 2, or 3).
        """

        # TODO: Implement your logic here.
        # This is just a random walker.
        print(history)

        action = random.randint(0, 3)

        return action  # do not change
