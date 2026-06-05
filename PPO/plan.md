# Plan: Add PPO to lunar-lander-rl

## Context

The repo currently solves `LunarLander-v3` (Gymnasium 1.3.0, torch 2.12.0) with **REINFORCE**:
- [reinforce.py](reinforce.py) — `PolicyNetwork` + `compute_returns`
- [agent.py](agent.py) — `Agent` class (`train`/`run`/`save_graph`/`optimize`) + argparse CLI
- [hyperparameters.yml](hyperparameters.yml) — named hyperparameter sets; artifacts saved to `runs/{set}.{log,pt,png}`

The goal is to add **PPO (Proximal Policy Optimization)** as a second algorithm that solves the same task, mirroring the existing structure and conventions. PPO is lower-variance and more sample-stable than REINFORCE, so it should produce a smoother reward curve. **The existing REINFORCE code must remain completely untouched.**

Confirmed decisions: **shared-trunk Actor-Critic** network, and **new parallel files** (no shared `utils.py`; small duplication of `save_graph`/`run` is accepted to keep REINFORCE untouched).

## Approach

Create two new files mirroring the REINFORCE pair, and append one hyperparameter block. REINFORCE files are reference-only — do not edit them.

### 1. New file: `ppo.py` (mirrors [reinforce.py](reinforce.py))

`ActorCritic(nn.Module)` — shared trunk, two heads:
- `__init__(state_dim, action_dim, hidden_dim)`: shared MLP `Linear(state_dim,hidden)→ReLU→Linear(hidden,hidden)→ReLU`, then `actor = Linear(hidden, action_dim)` (logits) and `critic = Linear(hidden, 1)` (value).
- `forward(x)` → `(logits, value.squeeze(-1))`.
- `select_action(state)` — used during rollout; samples from `Categorical(logits=...)`, returns `(action.item(), log_prob, value)`. Mirrors REINFORCE's `select_action` but adds the critic value.
- `evaluate_actions(states, actions)` — used during the update; returns `(new_log_probs, entropy, values)` for stored `(state, action)` batches under the current policy.

`compute_gae(rewards, values, dones, last_value, gamma, gae_lambda)` — the analogue of `compute_returns`, computed backwards (matches existing style):
```
values = values + [last_value]
for t in reversed(range(T)):
    mask  = 1.0 - float(dones[t])
    delta = rewards[t] + gamma * values[t+1] * mask - values[t]
    gae   = delta + gamma * gae_lambda * mask * gae
    advantages[t] = gae
returns    = advantages + values[:-1]
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # normalize
return advantages, returns
```
`returns` (un-normalized) are the value-head targets; `advantages` (normalized) weight the policy loss. `dones` masks prevent advantage bleeding across episode boundaries.

### 2. New file: `ppo_agent.py` (mirrors [agent.py](agent.py))

`PPOAgent` class — same `__init__` pattern (load YAML, set env params + paths `runs/{set}.{log,pt,png}`) plus the new PPO hyperparameters. Methods:

- **`collect_rollout(env, policy, state)`** — steps the **single env** for `n_steps` (NOT whole episodes), filling six lists: `states, actions, old_log_probs (detached), rewards, values (detached), dones`. Resets env when an episode ends. Tracks a `current_ep_reward` accumulator; on each `done`, appends it to a `completed_ep_rewards` list and resets to 0. After the loop, **bootstrap** `last_value` = critic value of the carried-over `state` (0 if last step was terminal). Returns the buffer lists, `last_value`, `completed_ep_rewards`, and the carried `state` (so the next rollout continues seamlessly).

- **`optimize(states, actions, old_log_probs, advantages, returns)`** — `n_epochs` passes over shuffled minibatches of size `minibatch_size`. Per minibatch:
  ```
  new_log_probs, entropy, values = policy.evaluate_actions(states_mb, actions_mb)
  ratio       = exp(new_log_probs - old_log_probs_mb)
  surr1       = ratio * adv_mb
  surr2       = clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv_mb
  policy_loss = -min(surr1, surr2).mean()
  value_loss  = ((values - returns_mb) ** 2).mean()        # plain MSE
  loss        = policy_loss + value_coef*value_loss - entropy_coef*entropy.mean()
  zero_grad → backward → clip_grad_norm_(max_grad_norm) → step
  ```

- **`train(render=False)`** — build env + `ActorCritic` (state_dim=8, action_dim=4) + Adam; infinite `itertools.count()` loop. Each iteration = one rollout: `collect_rollout` → `compute_gae` → `optimize`, then `rewards_per_episode.extend(completed_ep_rewards)`. The rest is **verbatim from [agent.py:121-150](agent.py#L121-L150)**: rolling-100 `mean_reward`, save `state_dict` to `MODEL_FILE` on new best, `save_graph` every ~10s, stop when `mean_reward >= stop_on_reward and len(rewards_per_episode) >= 100`. (Guard: `continue` if no episode completed in a rollout yet.)

- **`run()`** — near-copy of [agent.py:152-179](agent.py#L152-L179): load `state_dict` into `ActorCritic`, `eval()` + `no_grad()`, render episodes. Unpack `action, _, _ = policy.select_action(state)` (3-tuple now).

- **`save_graph(rewards_per_episode)`** — copied verbatim from [agent.py:182-193](agent.py#L182-L193) (rolling-100 mean plot).

- **CLI block** — identical to [agent.py:205-217](agent.py#L205-L217): `python ppo_agent.py PPOlunarlander1 --train` to train, omit `--train` to eval.

### 3. Edit: `hyperparameters.yml` — append `PPOlunarlander1`

Append a new block (do not modify the existing REINFORCE block). Same env block as REINFORCE, plus:
```yaml
PPOlunarlander1:
  env_id: LunarLander-v3
  env_make_params:
    continuous: False
    gravity: -10.0
    enable_wind: False
    wind_power: 10.0
    turbulence_power: 1.5
  learning_rate: 0.0003     # 3e-4, standard PPO LR
  gamma: 0.99
  hidden_dim: 128
  stop_on_reward: 200
  gae_lambda: 0.95
  clip_ratio: 0.2
  n_steps: 2048             # env steps per update
  n_epochs: 10              # passes per rollout
  minibatch_size: 64
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
```

## Critical files
- `/Users/rishijain/Documents/lunar-lander-rl/ppo.py` — **new** (`ActorCritic`, `compute_gae`)
- `/Users/rishijain/Documents/lunar-lander-rl/ppo_agent.py` — **new** (`PPOAgent` + CLI)
- [hyperparameters.yml](hyperparameters.yml) — **edit** (append `PPOlunarlander1` block)
- [agent.py](agent.py), [reinforce.py](reinforce.py) — **reference only, must remain unmodified**

## Verification
1. Train: `python ppo_agent.py PPOlunarlander1 --train`
   - `runs/PPOlunarlander1.log` gets "Training starting..." then upward-trending "New best mean reward" lines.
   - `runs/PPOlunarlander1.png` rolling-100 mean climbs from ~−200/−400 → crosses 0 → toward +200, smoother than the REINFORCE curve.
   - Converges to `mean_reward >= 200` (low hundreds of thousands of env steps for single-env PPO), logs "Solved!", exits, leaves a `.pt`.
2. Evaluate: `python ppo_agent.py PPOlunarlander1` — renders episodes with the trained policy landing successfully.
3. Sanity checks while debugging: `ratio ≈ 1.0` on the first epoch of each update (catches detach/indexing bugs); normalized advantages have ~0 mean / ~1 std; losses finite. If reward collapses, suspect LR too high or entropy_coef too low.
4. Confirm REINFORCE is unaffected: `python agent.py REINFORCElunarlander1` still runs.

## Deferred (not in scope, flagged for later)
- Vectorized envs (faster wall-clock; omitted for simplicity).
- Clipped value loss (minor stabilizer; start with plain MSE).
- Shared `utils.py` (only worth it if a 3rd algorithm is added later).
