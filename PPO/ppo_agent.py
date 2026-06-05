# Import environment
import gymnasium as gym
from gymnasium.spaces import Discrete

# Import ML libraries
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer

# Import stuff from other files
from ppo import ActorCritic, compute_gae

# Import yaml for hyperparams
import yaml

# Itertools for indefinite looping
import itertools

# os for directories
import os

# for randomness
import random

# MPL for plotting
import matplotlib.pyplot as plt

# datetime for datetime
from datetime import datetime, timedelta

# argparse for CLI training
import argparse

# for printing date and time
DATE_FORMAT = "%y-%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PPOAgent:
    def __init__(self, hyperparameter_set):
        # Get hyperparameters
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set

        # Set params
        self.env_id = hyperparameters['env_id']
        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.learning_rate = hyperparameters['learning_rate'] # adam step size
        self.gamma = hyperparameters['gamma'] # discount factor for future rewards
        self.hidden_dim = hyperparameters['hidden_dim'] # width of hidden layers
        self.stop_on_reward = hyperparameters['stop_on_reward'] # early-stopping target
        self.gae_lambda = hyperparameters['gae_lambda'] # Generalized Advantage Estimation smoothing, trades bias vs variance in advantage estimates
        self.clip_ratio = hyperparameters['clip_ratio'] # PPO clipping epsilon
        self.n_steps = hyperparameters['n_steps'] # environment steps collected per update (one rollout)
        self.n_epochs = hyperparameters['n_epochs'] # How many times to iterate over the rollout before getting fresh data
        self.minibatch_size = hyperparameters['minibatch_size'] # minibatch size to split rollout into for each gradient step (2048/64 = 32 gradient updates)
        self.value_coef = hyperparameters['value_coef'] # weight on the value-function
        self.entropy_coef = hyperparameters['entropy_coef'] # weight on the entropy bonus, encouraging exploitation by rewarding randomness
        self.max_grad_norm = hyperparameters['max_grad_norm'] # gradient clipping threshold

        # Path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
    
    def collect_rollout(self, env, policy_network, state):
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        values = []
        dones = []
        completed_ep_rewards = []
        current_ep_reward = 0
        for t in range(self.n_steps): # just play 2048 steps, not a set number of episodes
            # Get action, log prob, and value
            action, log_prob, value = policy_network.select_action(state)

            # Get next state, reward, terminated, and truncated
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # find done

            # Record
            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob.detach())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)

            # Increase accumulator
            current_ep_reward += reward

            if done:
                completed_ep_rewards.append(current_ep_reward)
                current_ep_reward = 0
                state, _ = env.reset()
            else:
                state = next_state

        # Get last value. 0 if episode ended at last step
        last_value = 0.0 if done else policy_network(torch.as_tensor(state))[1].item()
        
        # Convert states, actions, and old log probs for tensors
        # because they go to optimizer
        states = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions = torch.as_tensor(np.array(actions), dtype=torch.long)
        old_log_probs = torch.stack(old_log_probs) # stack into (n_steps,) tensor
        return (states, actions, old_log_probs, rewards, values, dones, last_value, completed_ep_rewards, state)
        
    def optimize(self, policy_network, optimizer, states, actions, old_log_probs, advantages, returns):
        for epoch in range(self.n_epochs):
            # Squeeze multiple gradient steps out of one batch of environment interaction
            perm = torch.randperm(self.n_steps)
            # Walk through the rollout in chunks
            for start in range(self.n_steps // self.minibatch_size):
                # Split things up into minibatches
                mb_idx = perm[start * self.minibatch_size : (start + 1) * self.minibatch_size]
                states_mb = states[mb_idx]
                actions_mb = actions[mb_idx]
                old_log_probs_mb = old_log_probs[mb_idx]
                advantages_mb = advantages[mb_idx]
                returns_mb = returns[mb_idx]

                # Re-run the current network on the states and actions
                new_log_probs, entropy, values = policy_network.evaluate_actions(states_mb, actions_mb)

                # Importance-sampling ratio — numerically stable by subtracting in log space
                ratio = torch.exp(new_log_probs - old_log_probs_mb)

                # Push up actions with positive advantages, push down with negative
                surr1 = ratio * advantages_mb

                # Same but with ratio clamped — the trust region
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_mb

                # Takes the smaller of the two surrogates per sample, negates bc optimizers want to minimize
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic loss: MSE between critics predictions and returns (regression targets from GAE)
                value_loss = torch.mean((values - returns_mb)**2)

                # Calc loss
                loss = policy_loss + value_loss * self.value_coef - self.entropy_coef * entropy.mean()
                
                # Zero grad + backprop
                optimizer.zero_grad()
                loss.backward()

                # Clip grad to guard against outlier minibatches
                torch.nn.utils.clip_grad_norm_(policy_network.parameters(), self.max_grad_norm)
                
                # Apply clipped gradients
                optimizer.step()
        
    def train(self):
        # Build env
        env = gym.make(self.env_id,
                       render_mode=None,
                       **self.env_make_params)

        # Create ActorCritic network & optimizer
        assert env.observation_space.shape is not None
        num_states = env.observation_space.shape[0] # should be 8

        assert isinstance(env.action_space, Discrete)
        num_actions = env.action_space.n # should be 4

        actor_critic_network = ActorCritic(num_states, num_actions, self.hidden_dim)
        optimizer = torch.optim.Adam(params=actor_critic_network.parameters(),
                                          lr=self.learning_rate)

        # Create rewards tracking variables
        rewards_per_episode = []
        best_mean_reward = float('-inf')

        # Begin logging
        start_time = datetime.now()
        last_graph_update_time = start_time
        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')
        
        # Loop infinitely
        state, _ = env.reset()
        for i in itertools.count():
            # Each iteration is one rollout
            # Rollout is of n_steps, which may contain several finished episodes or zero episodes or end midway
            # fixed step is good for a stable batch size.
            # and GAE doesn't need episodes to align, we can cut off the rollout mid-episode and be fine

            # 1. Collect rollout
            states, actions, old_log_probs, rewards, values, dones, last_value, completed_ep_rewards, state = self.collect_rollout(env, actor_critic_network, state)

            # 2. Compute gae
            advantages, returns = compute_gae(rewards, values, dones, last_value, self.gamma, self.gae_lambda)

            # 3. Optimize
            self.optimize(actor_critic_network, optimizer, states, actions, old_log_probs, advantages, returns)
            
            # Rollout returns a list of rewards for episodes that finished during the n_steps.
            # Extend flattens them into the history
            if completed_ep_rewards == []:
                continue # in case an episode just doesn't finish at all
            else:
                rewards_per_episode.extend(completed_ep_rewards)
                mean_reward = np.mean(rewards_per_episode[-100:])
            
            # Update best mean reward and log and save model
            if mean_reward > best_mean_reward:
                # log
                log_message = (f"{datetime.now().strftime(DATE_FORMAT)}: Episode {i} | New best mean reward: {mean_reward:.1f}")
                print(log_message)

                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')
                
                best_mean_reward = mean_reward
                torch.save(actor_critic_network.state_dict(), self.MODEL_FILE)
            
            # Update the graph every ~10 seconds
            if datetime.now() - last_graph_update_time > timedelta(seconds=10):
                self.save_graph(rewards_per_episode)
                last_graph_update_time = datetime.now()
            
            # Check for stop on rewards condition
            # >= because mean_reward won't be guaranteed exactly self.stop_on_reward
            # use len(rewards_per_episode) check to avoid outliers from ending training early
            if mean_reward >= self.stop_on_reward and len(rewards_per_episode) >= 100:
                # Log a solved message
                log_message = "Solved! (reached stop_on_reward)"
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')
                
                break
    
    def run(self):
        # Build the environment
        env = gym.make(self.env_id,
                       render_mode="human",
                       **self.env_make_params)
        
        # Load the network
        assert env.observation_space.shape is not None
        num_states = env.observation_space.shape[0] # should be 8

        assert isinstance(env.action_space, Discrete)
        num_actions = env.action_space.n # should be 4

        actor_critic_network = ActorCritic(num_states, num_actions, self.hidden_dim)
        actor_critic_network.load_state_dict(torch.load(self.MODEL_FILE, weights_only=True))

        # Activate inference settings
        actor_critic_network.eval()
        with torch.no_grad():
            for i in itertools.count(): # test infinitely (until control C stop)
                # Rollout but without gathering any training-relevant data
                state, _ = env.reset()
                done = False
                while not done:
                    action, _, _ = actor_critic_network.select_action(state)
                    state, _, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

    def save_graph(self, rewards_per_episode):
        mean_rewards = np.zeros(len(rewards_per_episode))

        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99) : x + 1])
        
        fig = plt.figure(1)
        plt.xlabel('Episodes')
        plt.ylabel('Mean reward of last 100 eps')
        plt.plot(mean_rewards)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig) # so figures don't pile up 

if __name__ == '__main__':
    # Parser for CLI inputs
    parser = argparse.ArgumentParser(description="Train or test?")
    parser.add_argument('hyperparameters', help='Enter the name of the set of hyperparameters to test/train')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    ppo = PPOAgent(hyperparameter_set=args.hyperparameters)

    if args.train:
        ppo.train()
    else:
        ppo.run()
