"""
Training script for GA02 (PyTorch DQN) on snake-rl.

- Defaults to version v17.1 (so you normally don't need to pass --version)
- Lets you tune gamma (discount) and epsilon schedule from the CLI
- Does NOT modify rewards inside game_environment.py (keeps the +1/-1 per food/no-food logic)

This file orchestrates the full DQN training loop:
- parses CLI hyperparameters
- loads JSON config for the chosen version
- creates training and evaluation environments
- runs experience collection + optimization steps
- periodically evaluates, logs metrics and saves checkpoints
"""

import os
import time
import json
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import play_game2
from game_environment import SnakeNumpy
from agent import DeepQLearningAgent


# ---------------- CLI ----------------
def parse_args():
    """
    Parse command line arguments for the training run.

    Exposes:
    - model version (points to model_config/<version>.json)
    - training length / logging frequency
    - number of parallel environments for train/eval
    - warmup frames before starting SGD updates
    - RL hyperparameters (gamma, epsilon schedule)
    - optimization hyperparameters (batch size)
    - a 'fast' flag to run a smaller sanity-check experiment
    """
    p = argparse.ArgumentParser(description="Train DQN on snake-rl (GA02, PyTorch)")

    # Model config version (JSON in model_config/)
    p.add_argument(
        "--version",
        default="v17.1",
        help="model_config/<version>.json (default: v17.1)",
    )

    # Training length / logging frequency
    p.add_argument(
        "--episodes",
        type=int,
        default=50000,
        help="training iterations (outer loop steps, default 50000)",
    )
    p.add_argument(
        "--log-freq",
        type=int,
        default=500,
        help="iterations between eval + checkpoint (default 500)",
    )

    # Parallelism and warmup
    p.add_argument(
        "--games",
        type=int,
        default=64,
        help="parallel games for training (SnakeNumpy, default 64)",
    )
    p.add_argument(
        "--eval-games",
        type=int,
        default=10,
        help="parallel games for evaluation (default 10)",
    )
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=512 * 64,
        help="frames to collect before training (default 512*64)",
    )

    # RL hyperparameters
    p.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor γ (default from model_config or 0.99)",
    )
    p.add_argument(
        "--eps-start",
        type=float,
        default=1.0,
        help="initial epsilon (default 1.0)",
    )
    p.add_argument(
        "--eps-end",
        type=float,
        default=0.01,
        help="final epsilon floor (default 0.01)",
    )
    p.add_argument(
        "--eps-decay",
        type=float,
        default=0.97,
        help="multiplicative epsilon decay applied at each log step (default 0.97)",
    )

    # Optimisation
    p.add_argument(
        "--batch",
        type=int,
        default=64,
        help="DQN minibatch size (default 64)",
    )

    # Tiny quick run (for sanity tests)
    p.add_argument(
        "--fast",
        action="store_true",
        help="small shake-out run (episodes=2000, log-freq=100, games=32, eval-games=4, warmup-frames=2048)",
    )

    return p.parse_args()


# ---------------- Helpers ----------------
def load_model_config(version: str):
    """
    Load a model configuration JSON for the given version string.

    The JSON file typically specifies:
    - board_size
    - number of stacked frames
    - max time limit
    - action space size
    - obstacle flag
    - replay buffer size
    - (optionally) default RL hyperparameters such as gamma
    """
    with open(f"model_config/{version}.json", "r") as f:
        return json.load(f)


# ---------------- Main ----------------
def main():
    # Parse CLI arguments at startup
    args = parse_args()

    # --- Load config from JSON ---
    # This reads environment and agent settings that correspond
    # to a particular experimental version (e.g. v17.1).
    cfg = load_model_config(args.version)
    version = args.version

    # Unpack core configuration parameters from JSON
    board_size     = cfg["board_size"]
    frames         = cfg["frames"]
    max_time_limit = cfg["max_time_limit"]
    n_actions      = cfg["n_actions"]
    obstacles      = bool(cfg["obstacles"])
    buffer_size    = cfg["buffer_size"]
    gamma_default  = 0.99  # fallback gamma if not provided in config

    # --- Defaults / CLI overrides ---
    # For each hyperparameter we choose a default, but allow CLI to override.
    # This makes the script easy to run with defaults, but still flexible.
    episodes      = 100_000 if args.episodes      is None else args.episodes
    log_frequency = 500     if args.log_freq      is None else args.log_freq
    games_eval    = 8       if args.eval_games    is None else args.eval_games
    n_games_train = 64      if args.games         is None else args.games
    warmup_frames = 512 * 64 if args.warmup_frames is None else args.warmup_frames
    gamma         = args.gamma if args.gamma is not None else cfg.get("gamma", gamma_default)

    # Fast mode for quick tests
    # Reduces the scale of the experiment so we can check that
    # the training loop and logging work end-to-end.
    if args.fast:
        episodes      = 2_000
        log_frequency = 100
        n_games_train = 32
        games_eval    = 4
        warmup_frames = 2_048

    # --- Build agent ---
    # Instantiate the PyTorch DQN agent that we implemented for GA02.
    # All remaining training logic will call into this object.
    agent = DeepQLearningAgent(
        board_size=board_size,
        frames=frames,
        n_actions=n_actions,
        buffer_size=buffer_size,
        version=version,
        gamma=gamma,
    )
    print(f"Agent: DeepQLearningAgent | version={version} | gamma={gamma:.4f}")

    # Epsilon schedule
    # ε is only decayed at each logging step, not every iteration.
    epsilon     = float(args.eps_start)
    epsilon_end = float(args.eps_end)
    eps_decay   = float(args.eps_decay)   # applied per LOG STEP
    reward_type = "current"              # matches original repo reward shaping
    sample_actions = False               # standard ε-greedy, no extra sampling tricks

    # --- Optional warmup: fill replay buffer a bit before training ---
    # Before we start gradient updates we can populate the replay buffer
    # with some initial transitions. This makes the first SGD steps more stable.
    if warmup_frames > 0:
        # Choose number of parallel games for warmup based on target frame count.
        games = max(64, min(512, warmup_frames // 64))  # keep it reasonable

        # Create a vectorized environment for collecting warmup data.
        env_warm = SnakeNumpy(
            board_size=board_size,
            frames=frames,
            max_time_limit=max_time_limit,
            games=games,
            frame_mode=True,
            obstacles=obstacles,
            version=version,
        )

        t0 = time.time()
        # Collect experience only (record=True) using ε-greedy policy.
        _ = play_game2(
            env_warm,
            agent,
            n_actions,
            n_games=games,
            record=True,
            epsilon=epsilon,   # explore
            verbose=True,
            reset_seed=False,
            frame_mode=True,
            total_frames=warmup_frames,
        )
        dt = time.time() - t0
        print(f"[warmup] Collected {warmup_frames} frames in {dt:.2f}s")

    # --- Training & eval environments ---
    # Training environment: many parallel games to maximize sample throughput.
    env = SnakeNumpy(
        board_size=board_size,
        frames=frames,
        max_time_limit=max_time_limit,
        games=n_games_train,
        frame_mode=True,
        obstacles=obstacles,
        version=version,
    )

    # Evaluation environment: separate copy to avoid interference from training.
    env_eval = SnakeNumpy(
        board_size=board_size,
        frames=frames,
        max_time_limit=max_time_limit,
        games=games_eval,
        frame_mode=True,
        obstacles=obstacles,
        version=version,
    )

    # --- Training loop ---
    # We track basic metrics so that we can:
    # - monitor learning progress over time
    # - select the best checkpoint later
    model_logs = {
        "iteration":   [],
        "reward_mean": [],
        "length_mean": [],
        "games":       [],
        "loss":        [],
    }

    print(
        f"Start training → episodes={episodes}, log-freq={log_frequency}, "
        f"train-games={n_games_train}, eval-games={games_eval}, batch={args.batch}, "
        f"eps=({epsilon} → {epsilon_end}, x{eps_decay} per log step)"
    )

    try:
        # Each 'it' is one outer iteration: collect experience + one optimization step.
        for it in tqdm(range(1, episodes + 1)):
            # 1) Collect transitions using frame_mode SnakeNumpy
            #    This step runs n_games_train parallel games and writes transitions
            #    into the agent's replay buffer.
            _ = play_game2(
                env,
                agent,
                n_actions,
                epsilon=epsilon,            # ε-greedy exploration
                n_games=n_games_train,
                record=True,
                sample_actions=sample_actions,
                reward_type=reward_type,
                frame_mode=True,
                total_frames=n_games_train, # one step per parallel game
                stateful=True,
            )

            # 2) One optimization step
            #    The agent pulls a minibatch from the replay buffer and performs
            #    a single DQN update (backprop + optimizer.step()).
            loss = agent.train_agent(
                batch_size=args.batch,
                num_games=n_games_train,
                reward_clip=True,
            )

            # 3) Periodic evaluation + logs + checkpoint
            #    Every log_frequency iterations we:
            #      - run a greedy evaluation (ε=-1)
            #      - log mean reward, episode length and loss
            #      - save a checkpoint and update target network
            #      - decay ε for the next training phase
            if it % log_frequency == 0:
                rewards, lengths, total_games = play_game2(
                    env_eval,
                    agent,
                    n_actions,
                    n_games=games_eval,
                    epsilon=-1,          # greedy evaluation (no exploration)
                    record=False,
                    sample_actions=False,
                    frame_mode=True,
                    total_frames=-1,
                    total_games=games_eval,
                )

                # Aggregate metrics over evaluation games
                model_logs["iteration"].append(it)
                model_logs["reward_mean"].append(round(int(rewards) / total_games, 2))
                model_logs["length_mean"].append(round(int(lengths) / total_games, 2))
                model_logs["games"].append(total_games)
                model_logs["loss"].append(loss)

                # Save logs to disk so that training progress can be inspected
                # and the best checkpoint can be selected for evaluation.
                os.makedirs("model_logs", exist_ok=True)
                log_df = pd.DataFrame(model_logs)[
                    ["iteration", "reward_mean", "length_mean", "games", "loss"]
                ]
                log_df.to_csv(f"model_logs/{version}.csv", index=False)

                # Target network update + checkpoint
                # This implements the classic DQN "target network" trick:
                # the target net is periodically synced to stabilize training.
                agent.update_target_net()
                out_dir = f"models/{version}"
                os.makedirs(out_dir, exist_ok=True)
                agent.save_model(file_path=out_dir, iteration=it)

                # Epsilon decay only at log steps
                # This keeps exploration high between log events and decays
                # it smoothly as training progresses.
                epsilon = max(epsilon * eps_decay, epsilon_end)

    except KeyboardInterrupt:
        # Allow manual interruption without losing progress:
        # we still sync the target net and write a final checkpoint.
        print("\nTraining interrupted by user. Saving final checkpoint...")
        out_dir = f"models/{version}"
        os.makedirs(out_dir, exist_ok=True)
        agent.update_target_net()
        agent.save_model(file_path=out_dir, iteration=0)

    print("✅ Done.")


if __name__ == "__main__":
    # Standard Python entry point to start training when run as a script.
    main()
