import torch
from torch import nn

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1),
            nn.Flatten()
        ) 

        self.actor_head = nn.Sequential(
            nn.Linear(in_features=2304, out_features=18*2*4),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(in_features=2304, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1),
        )

    def forward(self, state):
        features = self.backbone(state)
        return self.actor_head(features), self.critic_head(features)

def decode_action(action_idx):
    """
    Converts a flat action index (0-143) into the 
    MultiDiscrete format [pos, action, direction].
    """
    # If action_idx is a tensor, we want to ensure it's on CPU/Python int for the env
    if isinstance(action_idx, torch.Tensor):
        action_idx = action_idx.item()

    direction = action_idx % 4
    remainder = action_idx // 4
    
    action = remainder % 2
    pos = remainder // 2
    
    # Returning as a list/array compatible with PettingZoo's step()
    return [pos, action, direction]

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    from mycheckersenv import ActionType, Direction, CheckersEnv
    env = CheckersEnv()
    env.reset()
    observations = env.observe("black")
    state = torch.concat([torch.from_numpy(tensor) for tensor in observations.values()], dim=0).float().to(device).unsqueeze(0)

    model = ActorCritic().to(device)
    logits, value = model(state)
    policy = torch.distributions.Categorical(logits=logits)
    action_idx = policy.sample().item()
    action = decode_action(action_idx)

    print(f"Action: {action}, Value: {value.item():.2f}")
    env.step(action)
