# lunar-lander-rl
This is a PyTorch implementation of Proximal Policy Optimization (PPO) and REward Increment = Nonnegative Factor × Offset Reinforcement × Characteristic Eligibility with normalized returns (REINFORCE). Both algorithms learn to play **Lunar Lander** from the [Lunar Lander Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment.
This implementation includes a configurable training pipeline.

## Demos
| PPO | REINFORCE |
|-------------|----------|
| ![PPO](demos/ppo_demo.mov) | ![REINFORCE](demos/reinforce_demo.mov) |

Trained agents and reward curves are saved under `runs/` in the respective `PPO/` and `REINFORCE/` directories.

## Features
- **PPO** and **REINFORCE** implementation from scratch, including normalized returns on REINFORCE.
- **Hyperparameters** configurable per each experiment.
- **Automatic logging**, **best-model checkpointing**, and **live reward/epsilon plots**.

## Project Layout

```
lunar-lander-rl/
├── PPO/
│   ├── ppo.py                # ActorCritic network + GAE advantage computation
│   ├── ppo_agent.py          # PPOAgent: training loop, CLI entry point
│   ├── hyperparameters.yml   # Named hyperparameter sets
│   └── runs/                 # Logs, reward plots, and saved checkpoints
├── REINFORCE/
│   ├── reinforce.py          # PolicyNetwork + normalized-return computation
│   ├── reinforce_agent.py    # Agent: training loop, CLI entry point
│   ├── hyperparameters.yml   # Named hyperparameter sets
│   └── runs/                 # Logs, reward plots, and saved checkpoints
├── demos/                    # Recorded gameplay of trained agents
├── requirements.txt
└── README.md
```

## Setup
Note: requires **Python 3.11** 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train an agent using a named hyperparameter set from `hyperparameters.yml`:

```bash
cd ALGORITHM_DIR
python ALGORITHM_agent.py HYPERPARAM_SET_NAME --train
```

Watch a trained agent play (loads `runs/<set>.pt`, renders to screen):

```bash
cd ALGORITHM_DIR
python agent.py HYPERPARAM_SET_NAME
```

While training, the script writes:
- `runs/<set>.log`. The timestamped log of new best rewards.
- `runs/<set>.pt`. The best model weights so far.
- `runs/<set>.png`. Curves of mean-reward and epsilon decay.

## Hyperparameter sets
Defined in `hyperparameters.yml` for each algorithm's directory. Add a new key to define your own experiment.

## Future Extensions
For PPO:
- Vectorized envs.
- Clipped value loss.
