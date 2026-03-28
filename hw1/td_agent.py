import pickle
import os
from collections import defaultdict
import requests
import time
from typing import List, Tuple, Dict
import random
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


SERVER_URL = "https://mfgame-52355607629.us-west2.run.app"

# (x, y, action, next_x, next_y, reward)
TrajectoryStep = Tuple[int, int, int, int, int, float]

class Agent:
    def __init__(self, server_url: str = SERVER_URL, start_x: int = 0, start_y: int = 40):
        self.server_url = server_url
        self.start_x = start_x
        self.start_y = start_y
        self.x = 0
        self.y = 0
        self.history: List[List[TrajectoryStep]] = []
        self._q = defaultdict(lambda: { action: 0 for action in range(4) })
        self._epsilon = 0.1
        self._alpha = 0.1
        self._gamma = 1
        self._decay = 0.995
        self._goal_count = 0

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
        
    def move(self, action: int) -> Tuple[int, int, float, bool]:
        """
        Sends a move request to the server.
        """
        payload = {
            "x": self.x,
            "y": self.y,
            "action": action
        }
        try:
            resp = requests.post(f"{self.server_url}/move", json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            nx, ny = data["new_x"], data["new_y"]
            reward = data["reward"]
            done = data["done"]
            
            # Update state
            self.x = nx
            self.y = ny
            
            return nx, ny, reward, done
            
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with server: {e}")
            raise

    def choose_action(self, state):
        action_space = self._q[state]
        optimal_q = max(action_space.items(), key=lambda item: item[1])[1]

        # Filter greedy actions
        greedy_actions = [action for action in action_space if action_space[action] >= optimal_q] # >= optimal_q in case of floating point jankiness
        non_greedy_actions = [action for action in action_space]

        explore = random.random() < self._epsilon
        if explore:
            action = random.choice(non_greedy_actions)
            agent_trace.debug(f"Exploring: Selected action {action} randomly.")
            return action
        else:
            
            action = random.choice(greedy_actions)
            agent_trace.debug(f"Exploiting: Selected greedy action {action}.")
            return max(action_space.items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state, next_action):
        q = self._q[state][action]
        q_next = self._q[next_state][next_action]
        self._q[state][action] = q + self._alpha * (reward + self._gamma * q_next - q)

    def run_episode(self):
        current_history: List[TrajectoryStep] = []
        total_reward = 0
        step_count = 0
        self.x = self.start_x
        self.y = self.start_y
        print(f"Starting episode at ({self.x}, {self.y})")

        action = self.choose_action((self.x, self.y))

        done = False
        while not done and step_count <= 400:
            step_count += 1
            new_x, new_y, reward, done = self.move(action)

            print(f"Step {step_count}: Action {action} -> ({new_x}, {new_y}), R={reward}")

            total_reward += reward
            current_history.append((self.x,self.y,action,new_x,new_y,reward))

            if done:
                self._goal_count += 1
                agent_trace.info(f"Goal reached! Total goals: {self._goal_count}")
                print("Goal reached!")
                break

            next_action = self.choose_action(action)
            self.update((self.x, self.y), action, reward, (new_x, new_y), next_action)
            self.x = new_x
            self.y = new_y
            action = next_action

        print(f"Episode finished. Total reward: {total_reward}")

        agent_trace.info(f"Annealing: New epsilon is {self._epsilon:.4f}")
        self._epsilon = max(self._epsilon*self._decay, 0.01)
        self.history.append(current_history)
        return step_count, total_reward

    def dump_state(self, filepath):
        '''
        Dumps the q-value table for future learning.
        '''
        if self._q is None:
            return

        with open(filepath, 'wb') as f:
            pickle.dump(self._q, f)

