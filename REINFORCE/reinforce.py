import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.layer(x)

    # Action sampling
    def select_action(self, state):
        # Need to pick actions AND remember how likely they are bc REINFORCE update
        # is from log π(a_t | s_t)

        # Get logits
        state_t = torch.as_tensor(state, dtype=torch.float32)
        logits = self(state_t) # self is the network, this just calls forward

        # Softmax logits internally with torch.distributions.Categorical
        dist = Categorical(logits=logits)

        # Get action as an int
        action = dist.sample()

        # Get log prob
        log_prob = dist.log_prob(action)

        # We want action as an int but log prob as a tensor for backprop reasons
        # but can't .item() action above or log_prob calc breaks
        return (action.item(), log_prob)

def compute_returns(rewards, gamma):
    # Actions get credited for everything that comes after them
    # G_t for each timestep is how muhc reward followed this action
    returns = []
    G = 0

    # We compute it backwards 
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G) # Prepend because we are going backward
    
    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize
    returns = (returns - returns.mean())/(returns.std() + 1e-8) # 1e-8 to avoid division by zero
    return returns