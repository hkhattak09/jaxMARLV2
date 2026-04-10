# Hanabi env notes (JaxMARL upstream) - Stage 0

## Files reviewed

- jaxmarl/environments/hanabi/hanabi.py
- jaxmarl/environments/hanabi/hanabi_game.py

## Environment contract

- Class: HanabiEnv (inherits HanabiGame)
- Agents: configurable, constrained to 2..5
- Default num_agents: 2
- Default hand_size: 5 for 2-3 players, 4 for 4-5 players

## Observation and action dimensions

- Per-agent observation size is env.obs_size (set in hanabi.py)
- For default 2-player settings, obs_size is 658
- Action dimension is env.num_moves where:
  - num_moves = hand_size * 2 + (num_agents - 1) * (num_colors + num_ranks) + 1
  - default 2-player => 5*2 + 1*(5+5) + 1 = 21

## Rewards and episode length

- Reward is dense and score-based:
  - +1 when fireworks increase
  - bomb-0 adjustment when lives hit zero
- Episode score is tracked in state.score and is bounded by total fireworks (max 25 with defaults)
- Termination when any of:
  - all fireworks completed (game_won)
  - out of lives
  - last round completed after deck exhaustion
- There is no fixed hardcoded horizon in these files; effective length is game-dynamics-driven.

## World-state getter

- HanabiEnv does not expose a built-in world_state() method in jaxmarl/environments/hanabi/.
- MAPPO scripts construct world-state via a wrapper:
  - upstream baseline: HanabiWorldStateWrapper in baselines/MAPPO/mappo_rnn_hanabi.py
  - local implementation: HanabiWorldStateWrapper in smax_ctm/train_mappo_ctm_hanabi.py

## Upstream MAPPO/IPPO Hanabi baseline script and reviewer-check params

Reviewed upstream files:

- baselines/MAPPO/mappo_rnn_hanabi.py
- baselines/MAPPO/config/mappo_homogenous_rnn_hanabi.yaml
- baselines/IPPO/ippo_rnn_hanabi.py
- baselines/IPPO/config/ippo_rnn_hanabi.yaml

Key hyperparameters used in both RNN configs:

- GRU_HIDDEN_DIM: 128
- FC_DIM_SIZE: 128
- ENT_COEF: 0.01
- LR: 5e-4
- NUM_ENVS: 1024
- NUM_STEPS: 128

Episode return range reviewers typically expect:

- Metric logged in scripts is returned_episode_returns (team return).
- Hanabi score range under default rules is 0..25.
- In practice, strong MAPPO/GRU Hanabi runs are expected to converge near the top of this range (roughly low-20s up to near-25), with exact values seed/config dependent.
