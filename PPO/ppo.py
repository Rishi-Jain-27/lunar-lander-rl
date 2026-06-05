import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
            )
        self.actor = nn.Linear(hidden_dim, action_dim) # logits
        self.critic = nn.Linear(hidden_dim, 1) # value
    
    def forward(self, x):
        x = self.layer(x)

        # Bring value down to scalar
        return (self.actor(x), self.critic(x).squeeze(-1))

    # For the first time PPO touches the policy network
    # which is rollout
    def select_action(self, state):
        # Get logits
        state_t = torch.as_tensor(state, dtype=torch.float32)
        logits, value = self(state_t)  # self is the network, this just calls forward

        # Softmax logits internally with torch.distributions.Categorical
        dist = Categorical(logits=logits)

        # Get action as an int
        action = dist.sample()

        # Get log prob
        log_prob = dist.log_prob(action)

        # We want action as an int but log prob as a tensor for backprop reasons
        # but can't .item() action above or log_prob calc breaks
        return (action.item(), log_prob, value)
    
    # For the second time PPO touches the policy network
    # which is during the update
    def evaluate_actions(self, states, actions):
        # Re-score the picks that select_action made
        states_t = torch.as_tensor(states, dtype=torch.float32)
        logits, values = self(states_t)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(torch.as_tensor(actions, dtype=torch.long))
        entropy = dist.entropy()
        return (new_log_probs, entropy, values)

# Turn one batch of experiences into two training signals
def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    advantages = []
    values = values + [last_value] # so V(t+1) always exists even for the last real step

    gae = 0.0 # running accumulator.

    # We walk backwards thru time
    for t in reversed(range(len(rewards))): # T = len(rewards)
        # mask zeros delta and gae when dones[t] is true
        # if the episode continued at step t, mask is 1.
        # if not, mask is 0.
        # When episode ends, the next state is from a new
        # unrelated episode, mask erases the fake connection there
        mask  = 1.0 - float(dones[t])

        # reward gained + discounted value of landing - value expected
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]

        # Blend in the advantage from all the following steps too
        # GAE recursion
        gae = delta + gamma * gae_lambda * mask * gae

        # Prepend the result bc we iterate backwards
        advantages.insert(0, gae)
    
    # convert to tensors
    advantages = torch.as_tensor(advantages, dtype=torch.float32)
    values_t = torch.as_tensor(values[:-1], dtype=torch.float32)
    
    # Return is how wrong the critic was (advantage) + what the critic guessed (value)
    returns = advantages + values_t

    # normalize
    advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

    return (advantages, returns)

