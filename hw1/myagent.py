from datetime import datetime
from numpy import average
from collections import defaultdict
import random
from typing import List, Tuple

class StudentAgent:
    def __init__(self, epsilon=1, seed=1337):
        """
        Initialize your internal state here.
        """
        # Initialize (state,action) values to 0
        self._q = defaultdict(lambda: {action: 0 for action in range(4)})
        # Tracks (state,action) returns from the history
        self._returns = defaultdict(list)
        # Epsilon-greedy for choosing random (non-optimal action)
        self._epsilon = epsilon
        # For tracking changes to history
        self._history = []
        # Create history file
        self._history_file = f"history_{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}.txt"
        with open(self._history_file, "x") as f:
            f.write(f"Beginning run w/ parameters epsilon={self._epsilon}, seed={seed}\n")
        # Used to signal when to start annealling epsilon
        self._goal_count = 0
        random.seed(seed)

    def choose_action(self, state):
        '''
        This implements the epsilon-greedy policy where, for the given state,
        we greedily choose an action w/ optimal q-value with probability
        1-epsilon, and choose a non-greedy exploration action w/ probability
        epsilon.

        Ties between actions are broken randomly.
        '''
        actions = self._q[state]
        # Highest value of q.
        optimal_q = max(actions.items(), key=lambda item: item[1])[1]

        # Filter greedy actions
        greedy_actions = [action for action in actions if actions[action] >= optimal_q] # >= optimal_q in case of floating point jankiness
        non_greedy_actions = [action for action in actions]
        # Bernoulli trial to decide if we should explore w/ probability epsilon
        should_explore = random.binomialvariate(1,self._epsilon)

        if should_explore:
            return random.choice(non_greedy_actions)
        else:
            return random.choice(greedy_actions)

    def update_history(self, episode):
        '''
        Add the most recent episode's data to the history.
        '''
        self._history.append(episode)
        with open(self._history_file, "a") as f:
            f.write(f"{episode}\n")
        # Track first visits so each q value is updated at most once
        first_visits = set()
        # Total return
        episode_return = sum(step[-1] for step in episode)

        # Cumulative rewards from time 0 to t
        G = 0

        if episode_return > -400:
            # Give a large positive reward for reaching the goal state.
            # This should help to distinguish runs that reach terminal
            # states from runs that simply run out of moves.
            G += 500
            self._goal_count += 1

        for x, y, a, next_x, next_y, r in episode:
            s = (x,y)
            if (s, a) not in first_visits:
                first_visits.add((s,a))
                # Calculate return from this step onwards
                sampled_return = episode_return - G
                # Append to history
                self._returns[(s,a)].append(sampled_return)
                # Efficiently compute the incremental average of returns
                prev_q = self._q[s][a]
                k = len(self._returns[(s,a)])
                self._q[s][a] = prev_q + (1/k)*(sampled_return - prev_q)
            # Increment cumulative reward for the next step.
            G += r

        # Anneal epsilon after we have reached the terminal state 5 times
        if self._goal_count >= 5:
            # Need to maintain a non-zero epsilon to guarantee convergence
            self._epsilon = max(self._epsilon*0.99, 0.01)


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
        if len(history) > len(self._history):
            print("Updating history")
            self.update_history(history[-1])

        action = self.choose_action((x,y))

        return action  # do not change
