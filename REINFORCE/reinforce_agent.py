# Import environment
import gymnasium as gym
from gymnasium.spaces import Discrete

# Import ML libraries
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer

# Import stuff from other files
from REINFORCE.reinforce import PolicyNetwork, compute_returns

# Import yaml for hyperparams
import yaml

# Itertools for indefinite looping
import itertools

# os for directories
import os

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

class Agent:
    def __init__(self, hyperparameter_set):
        """
        Initialize all hyperparameters.
        """

        # Get hyperparameters
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set

        # Environment params
        self.env_id = hyperparameters['env_id']
        self.env_make_params = hyperparameters.get('env_make_params', {})

        # Training relevant params
        self.learning_rate = hyperparameters['learning_rate']
        self.gamma = hyperparameters['gamma']
        self.hidden_dim = hyperparameters['hidden_dim']
        self.stop_on_reward = hyperparameters['stop_on_reward']

        # Path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def train(self, render=False):
        # Build the environment
        env = gym.make(self.env_id,
                       render_mode="human" if render else None,
                       **self.env_make_params)

        # Create the policy network & optimizer
        assert env.observation_space.shape is not None
        num_states = env.observation_space.shape[0] # should be 8

        assert isinstance(env.action_space, Discrete)
        num_actions = env.action_space.n # should be 4

        policy_network = PolicyNetwork(num_states, num_actions, self.hidden_dim)
        self.optimizer = torch.optim.Adam(params=policy_network.parameters(),
                                     lr=self.learning_rate)

        rewards_per_episode = []
        best_mean_reward = float('-inf')

        # Begin logging
        start_time = datetime.now()
        last_graph_update_time = start_time
        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')
        for i in itertools.count():
            state, _ = env.reset()
            log_probs, rewards = [], []
            done = False

            while not done:
                action, log_prob = policy_network.select_action(state)

                state, reward, terminated, truncated, _ = env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                done = terminated or truncated
            # Optimize
            returns = compute_returns(rewards, self.gamma)
            self.optimize(log_probs, returns)

            episode_reward = sum(rewards)
            print(f"Episode {i}: total reward {episode_reward:.1f}, steps {len(rewards)}")
            
            rewards_per_episode.append(episode_reward)
            mean_reward = np.mean(rewards_per_episode[-100:])
            
            # Update best mean reward and log and save model
            if mean_reward > best_mean_reward:
                # log
                log_message = (f"{datetime.now().strftime(DATE_FORMAT)}: Episode {i} | New best mean reward: {mean_reward:.1f}")
                print(log_message)

                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')
                
                best_mean_reward = mean_reward
                torch.save(policy_network.state_dict(), self.MODEL_FILE)
            
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
        
        # Load policy network
        assert env.observation_space.shape is not None
        num_states = env.observation_space.shape[0] # should be 8

        assert isinstance(env.action_space, Discrete)
        num_actions = env.action_space.n # should be 4

        policy_network = PolicyNetwork(num_states, num_actions, self.hidden_dim)
        policy_network.load_state_dict(torch.load(self.MODEL_FILE, weights_only=True))
        
        # Activate inference settings
        policy_network.eval()
        with torch.no_grad():
            for i in itertools.count():
                state, _ = env.reset()
                done = False
                while not done:
                    action, _ = policy_network.select_action(state)

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
    
    def optimize(self, log_probs, returns):
        # Calc loss
        log_probs = torch.stack(log_probs) # stack the list of tensors into one tensor
        loss = -(log_probs * returns).sum()

        # Optimize!
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    # Parser for CLI inputs
    parser = argparse.ArgumentParser(description="Train or test?")
    parser.add_argument('hyperparameters', help='Enter the name of the set of hyperparameters to test/train')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    reinforce = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        reinforce.train()
    else:
        reinforce.run()