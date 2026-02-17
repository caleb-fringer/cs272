from runner import GameClient
import numpy as np

if __name__ == "__main__":
    from myagent import StudentAgent
    
    client = GameClient()
    agent = StudentAgent()

    num_epi = 20_000
    # Seed this with a value so the first running avg is not NaN
    total_rewards = [-400] 
    while np.average(total_rewards[-20:]) < -50 and len(total_rewards) < num_epi:
        _, tr = client.run_episode(agent)
        total_rewards.append(tr)

    
    print(f"Final score: {sum(total_rewards[-20:]) / 20}")
