import sys
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from myagent import ActorCritic, flatten_mask, decode_action
from mycheckersenv import CheckersEnv, pos_to_coord, ActionType, Direction

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model and hyperparameters
model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

env = CheckersEnv()
gamma = 0.99
num_episodes = 1000

# --- Train ---
cumulative_rewards = {agent: 0 for agent in env.possible_agents}
for episode in range(num_episodes):
    env.reset()
    
    # Initialize I = 1 for both agents
    I_factor = {agent: 1.0 for agent in env.possible_agents}
    
    # Storage for the previous step (S and A) to compute TD error
    prev_states = {}
    prev_masks = {}
    prev_actions = {}
    prev_I = {}
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        is_terminal = termination or truncation

        cumulative_rewards[agent] += reward
        
        # Convert observation_space representation to tensors for the 
        # ActorCritic model
        if not is_terminal:
            obs_array = observation["observations"]
            obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(device)
            
            mask_8x6x6 = observation["action_mask"]
            flat_mask = flatten_mask(mask_8x6x6)
            mask_tensor = torch.FloatTensor(flat_mask).unsqueeze(0).to(device)
        else:
            obs_tensor, mask_tensor = None, None

        # --- Learning Updates ---
        # If the agent has taken an action previously, we can perform gradient
        # updates
        if agent in prev_states:
            p_state = prev_states[agent]
            p_mask = prev_masks[agent]
            p_action = prev_actions[agent]
            p_I = prev_I[agent]
            
            # Estimate v(S')
            if is_terminal:
                next_v = torch.tensor([0.0], device=device) # if S' is terminal, v(S') = 0
            else:
                # We don't want to calculate the gradient of v(S') until the next step
                # S = the current observation's state.
                with torch.no_grad():
                    _, next_v = model(obs_tensor, mask=mask_tensor)
                    next_v = next_v.squeeze(0)
            
            # Now that S' is known for S = p_state, we can compute the 
            # gradients of v(S) and the logits of the policy pi(A|S)
            dist, value = model(p_state, mask=p_mask)
            value = value.squeeze(0)
            log_prob = dist.log_prob(p_action)
            
            # Calculate TD error for actor_loss. Actor doesn't depend on the
            # weights w of the value function (critic), so detach gradients for
            # this calculation
            delta = reward + gamma * next_v - value.detach()
            
            # Calculate critic update step
            # Equivalent to minimizing MSE between value and (reward + gamma * next_v)
            critic_loss = F.mse_loss(value, reward + gamma * next_v)
            
            # Calculate the actor update step
            # PyTorch optimizers minimize, so we negate the loss to perform gradient ascent
            actor_loss = - (p_I * delta * log_prob)
            
            loss = actor_loss + 0.5 * critic_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Take action ---
        if is_terminal:
            env.step(None)
        else:
            # Sample A from the policy (non-deterministic)
            with torch.no_grad():
                dist, _ = model(obs_tensor, mask=mask_tensor)
                action = dist.sample()
            
            # Store data for the next turn's update
            prev_states[agent] = obs_tensor
            prev_masks[agent] = mask_tensor
            prev_actions[agent] = action
            prev_I[agent] = I_factor[agent]
            
            # Decay I
            I_factor[agent] *= gamma
            
            # Take action A
            env_action = decode_action(action.item())
            env.step(env_action)
            
    if episode % 10 == 0:
        print(f"Episode {episode} completed.")

model_path = f"{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}-checkers-agent.pth"
torch.save(model.state_dict(), model_path)
print("Model saved to " + model_path)


# Redirect stdout
filepath = f"{datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")}-game.log"
with open(filepath, "w") as f:
    # Run test game, saving output to file
    env.reset()
    # Disable gradients
    model.eval()
    sys.stdout = f
    t=0
    for agent in env.agent_iter():
        t += 1
        observation, reward, termination, truncation, info = env.last()
        is_terminal = termination or truncation

        cumulative_rewards[agent] += reward

        if is_terminal:
            rewards = env.rewards
            if rewards["black"] == rewards["red"]:
                print("Game ended in a tie!")
            else:
                winner = max(rewards, key=lambda player: rewards[player])
                print(f"Winner: {winner}!")
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
        pos, action_type, dir = action
        coords = pos_to_coord(pos)
        print(f"Timestep {t}, {agent} takes action {coords, ActionType(action_type), Direction(dir)}")

        env.step(action)
        env.render()
    print(f"Cumulative rewards: {cumulative_rewards}")

sys.stdout = sys.__stdout__
print(f"Test game log saved to " + filepath)
