import torch
import numpy as np
from mycheckersenv import CheckersEnv
from myagent import ActorCritic

# --- Helpers ---
def decode_action(action_idx):
    """Converts 0-143 index back to MultiDiscrete [pos, action, direction]"""
    direction = action_idx % 4
    remainder = action_idx // 4
    action_type = remainder % 2
    pos = remainder // 2
    return [pos, action_type, direction]

def flatten_mask(mask_8x6x6):
    """
    Maps the (8, 6, 6) legal action mask to the flat 144 action space 
    using the pos_to_coord logic from the environment.
    """
    flat_mask = np.zeros(144, dtype=np.int8)
    for i in range(144):
        direction = i % 4
        action_type = (i // 4) % 2
        pos = i // 8
        
        # Match action_channel logic from env step()
        action_channel = action_type * 4 + direction
        
        # Match pos_to_coord logic
        row = pos // 3
        col = 2 * (pos % 3)
        if row % 2 == 0:
            col += 1
            
        flat_mask[i] = mask_8x6x6[action_channel, row, col]
    return flat_mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ActorCritic().to(device)
model.load_state_dict(torch.load("2026_03_30-07_10_47_PM-checkers-agent.pth", device))
env = CheckersEnv()
env.reset()

model.eval()

t=0
for agent in env.agent_iter():
    t += 1
    print(f"Timestep {t}, {agent}'s move:")
    observation, reward, termination, truncation, info = env.last()
    is_terminal = termination or truncation

    if is_terminal:
        print("Terminating!!!")
        env.step(None)
        break

    obs_array = observation["observations"]
    obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(device)
    
    mask_8x6x6 = env.legal_action_mask[agent]
    flat_mask = flatten_mask(mask_8x6x6)
    mask_tensor = torch.FloatTensor(flat_mask).unsqueeze(0).to(device)

    dist, _ = model(obs_tensor, mask=mask_tensor)
    action_idx = dist.sample() 

    action = decode_action(action_idx.item())
    env.step(action)
    env.render()
