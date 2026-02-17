import requests
import time
from typing import List, Tuple, Dict

SERVER_URL = "https://mfgame-52355607629.us-west2.run.app"

# (x, y, action, next_x, next_y, reward)
TrajectoryStep = Tuple[int, int, int, int, int, float]

class GameClient:
    def __init__(self, server_url: str = SERVER_URL, start_x: int = 0, start_y: int = 4):
        self.server_url = server_url
        self.x = start_x
        self.y = start_y
        self.history: List[List[TrajectoryStep]] = []

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

    def run_episode(self, agent, max_steps: int = 400):
        """
        Runs a single episode using the provided agent.
        """
        self.x = 0 
        self.y = 40 
        current_episode: List[TrajectoryStep] = []
        
        total_reward = 0.0
        print(f"Starting episode at ({self.x}, {self.y})")
        
        for step in range(max_steps):
            # Agent decides action based on current state and maybe history
            old_x = self.x
            old_y = self.y
            action = agent.get_action(self.x, self.y, self.history)

            nx, ny, reward, done = self.move(action)
            
            # Record step in current episode
            current_episode.append((old_x, old_y, action, nx, ny, reward))
            
            total_reward += reward
            
            print(f"Step {step}: Action {action} -> ({nx}, {ny}), R={reward}")
            
            if done:
                print("Goal reached!")
                break
        
        self.history.append(current_episode)
        print(f"Episode finished. Total Reward: {total_reward}")
        return current_episode, total_reward

if __name__ == "__main__":
    from myagent import StudentAgent
    
    client = GameClient()
    agent = StudentAgent()

    num_epi = 20_000
    total_rewards = []
    for _ in range(num_epi):
        _, tr = client.run_episode(agent)
        total_rewards.append(tr)
        if len(total_rewards) >= 20:
            if input("Do you still want to continue? Y or N: ").lower() != "y":break
    
    print(f"Final score: {sum(total_rewards[-20:]) / 20}")
