import os
import pickle
from datetime import datetime
from numpy import average
from collections import defaultdict
import random
from typing import List, Tuple
import logging

class EpisodeFilter(logging.Filter):
    '''
    Creates a filter that can inject agent state (episode number) into logs.
    '''
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def filter(self, record):
        # Dynamically grab the episode from the agent's private member
        record.episode = self.agent._episode
        return True

# Setup Agent Trace
agent_trace = logging.getLogger("agent_trace")
agent_trace.setLevel(logging.DEBUG)

# Setup History Log
history_logger = logging.getLogger("history")
history_logger.setLevel(logging.INFO)

class StudentAgent:
    def __init__(self, epsilon=1, seed=1337, decay_base=0.99, decay_rate=1/16):
        """
        Initialize your internal state here.
        """
        # Initialize (state,action) values to 0
        self._q = defaultdict(self._init_q)
        # Tracks (state,action) returns from the history
        self._returns = defaultdict(list)
        # Epsilon-greedy for choosing random (non-optimal action)
        self._epsilon = epsilon
        # For tracking changes to history
        self._history = []
        # Keep track of episode count
        self._episode = 0
        # Used to signal when to start annealling epsilon
        self._goal_count = 0
        random.seed(seed)
        # Configure logging to have access to internal state
        self._setup_logging()
        # Epsilon decay params
        self._decay_factor = decay_base ** decay_rate

    def _init_q(self):
        '''
        Creates an initial q-value dict w/ a value of 0 for every action.
        Used by defaultdict, but must be a named function to allow pickling.
        '''
        return {action: 0 for action in range(4)}

    def _setup_logging(self):
        '''
        Configures the loggers to use this instance's state.
        '''
        # Read the session directory from the environment, default to current dir
        session_dir = os.environ.get("SESSION_DIR", ".")
        
        agent_trace.handlers = []
        epi_filter = EpisodeFilter(self)
        agent_trace.addFilter(epi_filter)

        # Write DIRECTLY to the session directory
        agent_fh = logging.FileHandler(os.path.join(session_dir, "decisions.log"))
        agent_fh.setFormatter(logging.Formatter("Episode %(episode)d | %(message)s"))
        agent_trace.addHandler(agent_fh)

        history_fh = logging.FileHandler(os.path.join(session_dir, "history.log"))
        history_fh.setFormatter(logging.Formatter("%(message)s"))
        history_logger.addHandler(history_fh)

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
            action = random.choice(non_greedy_actions)
            agent_trace.debug(f"Exploring: Selected action {action} randomly.")
        else:
            action = random.choice(greedy_actions)
            agent_trace.debug(f"Exploiting: Selected greedy action {action} randomly.")
        return action

    def update_history(self, episode):
        '''
        Add the most recent episode's data to the history.
        '''
        self._episode += 1

        # Log history
        self._history.append(episode)
        history_logger.info(episode)

        # Track first visits so each q value is updated at most once
        first_visits = set()
        # Total return
        episode_return = sum(step[-1] for step in episode)

        # Cumulative rewards from time 0 to t
        G = 0

        if episode_return > -400:
            self._goal_count += 1
            agent_trace.info(f"Goal reached! Total goals: {self._goal_count}")

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
                self._q[s][a] = prev_q + self._learning_rate*(sampled_return - prev_q)
            # Increment cumulative reward for the next step.
            G += r

        # Anneal epsilon after we have reached the terminal state 5 times
        if self._goal_count >= 10:
            # Need to maintain a non-zero epsilon to guarantee convergence
            self._epsilon = max(self._epsilon*self._decay_factor, 0.01)
            agent_trace.info(f"Annealing: New epsilon is {self._epsilon:.4f}")

    def dump_state(self, filepath):
        '''
        Dumps the q-value table for future learning.
        '''
        if self._q is None:
            return

        with open(filepath, 'wb') as f:
            pickle.dump(self._q, f)

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
            self.update_history(history[-1])

        action = self.choose_action((x,y))

        return action  # do not change
