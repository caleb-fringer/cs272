import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1),
            nn.Flatten()
        ) 

        self.actor_head = nn.Sequential(
            nn.Linear(in_features=36, out_features=18*2*4),
            nn.Softmax()
        )

        self.critic_head = nn.Sequential(
            nn.Linear(in_features=36, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
        )

    def forward(self, state):
        features = self.backbone(state)
        return self.actor_head(state), self.critic_head(features)

if __name__ == "__main__":
    from mycheckersenv import ActionType, Direction, CheckersEnv
    env = CheckersEnv()
    env.reset()
    observations = env.observe("black").copy()
    state = torch.concat([torch.from_numpy(tensor) for tensor in observations.values()], dim=0)

    model = ActorCritic()
    action, value = model(state)
    print(f"Action: {ActionType(action)}, Value: {value:.2f}")
