import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone: Input is (4, 6, 6) (Masked action tensor)
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Critic: v(s,w)
        self.critic = nn.Sequential(
            nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Policy: pi(a|s, theta)
        self.actor = nn.Linear(2304, 144)

    def forward(self, obs, mask=None):
        features = self.backbone(obs)
        value = self.critic(features)
        logits = self.actor(features)
        
        if mask is not None:
            # Mask out illegal actions with a large negative number
            # Future idea: see how much longer learning takes without masking
            # (can it learn to differentiate between legal and illegal actions?)
            logits = logits.masked_fill(mask == 0, -1e9)
            
        dist = Categorical(logits=logits)
        return dist, value

def flatten_mask(mask_8x6x6):
    ''' 
    The legal action mask is dimension (8,6,6), but the action space uses
    a smaller 18x2x4 discrete action space. In order to use the action_mask,
    need to reduce it to an vector of shape (144,) by skipping over invalid
    squares and flattening the output.
    '''
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

def decode_action(action_idx):
    '''
    Converts 0-143 action index back to MultiDiscrete [pos, action, direction]
    '''
    direction = action_idx % 4
    remainder = action_idx // 4
    action_type = remainder % 2
    pos = remainder // 2
    return [pos, action_type, direction]

if __name__ == "__main__":
    # Test the implementation
    from mycheckersenv import ActionType, Direction, CheckersEnv

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    env = CheckersEnv()
    env.reset()
    observation = env.observe("black")
    state = torch.FloatTensor(observation["observations"]).unsqueeze(0).to(device)

    flat_mask = flatten_mask(observation["action_mask"])
    mask_tensor = torch.FloatTensor(flat_mask).unsqueeze(0).to(device)

    model = ActorCritic().to(device)
    dist, value = model(state, mask=mask_tensor)
    action = decode_action(dist.sample().item())

    print(f"Action: {action}, Value: {value.item():.2f}")
    env.step(action)
