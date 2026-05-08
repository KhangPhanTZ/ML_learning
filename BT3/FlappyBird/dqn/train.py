"""
Train the DQN agent on FlappyBirdEnv.

Usage:
    python train.py                       # default 2000 episodes, headless
    python train.py --episodes 5000       # train longer
    python train.py --crash-penalty -10   # stronger crash signal (allowed by spec)
    python train.py --render              # watch training (slow)

Saves the best-scoring checkpoint to ./best_dqn_model.pth (whatever yields
the highest pipes-passed in a single episode).
"""

from __future__ import annotations

import argparse
import random as pyrandom
import time
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent
from game import FlappyBirdEnv

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "best_dqn_model.pth"


def parse_args():
    p = argparse.ArgumentParser(description="DQN training for Flappy Bird (BT3).")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=10_000,
                   help="Hard cap on env steps per episode (prevents runaway perfect runs).")
    p.add_argument("--render", action="store_true", help="Render the game during training (slow).")
    p.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--memory-size", type=int, default=50_000)
    p.add_argument("--eps-decay-steps", type=int, default=30_000)
    p.add_argument("--target-update-steps", type=int, default=1_000)
    p.add_argument("--crash-penalty", type=float, default=-1.0,
                   help="Reward on crash. Spec allows 'or heavier penalty'; try -10 if learning is slow.")
    p.add_argument("--target-score", type=int, default=200,
                   help="Stop training early when avg pipes-passed over the last 20 eps >= this.")
    p.add_argument("--max-time-sec", type=float, default=None,
                   help="Hard wall-clock cap (seconds). Stops cleanly between episodes so the "
                        "best checkpoint on disk is never corrupted by an external SIGTERM.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from --model-path if it exists (continue training from a checkpoint).")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def set_seed(seed: int) -> None:
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")
    if device.type == "cuda":
        # Helpful for the user's GTX 1650 setup.
        print(f"[train] gpu    = {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    env = FlappyBirdEnv(render_mode=args.render, crash_penalty=args.crash_penalty)
    agent = DQNAgent(
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        eps_decay_steps=args.eps_decay_steps,
        target_update_steps=args.target_update_steps,
        device=device,
    )

    if args.resume and Path(args.model_path).exists():
        agent.load(args.model_path, load_optimizer=True)
        print(f"[train] resumed from {args.model_path} (train_steps={agent.train_steps})")

    best_score = -1
    recent_scores: list[int] = []
    t0 = time.time()

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        info = {"score": 0, "frames": 0}
        last_loss: float | None = None

        for _ in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, reward, next_state, float(done))

            loss = agent.train_step()
            if loss is not None:
                last_loss = loss

            state = next_state
            total_reward += reward
            steps += 1

            if args.render:
                env.render()
            if done:
                break

        score = info["score"]
        recent_scores.append(score)
        if len(recent_scores) > 20:
            recent_scores.pop(0)
        avg20 = sum(recent_scores) / len(recent_scores)

        improved = score > best_score
        if improved:
            best_score = score
            agent.save(args.model_path)

        if ep % args.log_every == 0 or improved:
            loss_str = f"{last_loss:.4f}" if last_loss is not None else "  -- "
            tag = " *NEW BEST*" if improved else ""
            print(
                f"[ep {ep:5d}] reward={total_reward:8.2f}  pipes={score:4d}  "
                f"steps={steps:5d}  eps={agent.epsilon:.3f}  loss={loss_str}  "
                f"best={best_score:4d}  avg20={avg20:6.1f}  "
                f"t={time.time() - t0:6.1f}s{tag}"
            )

        if avg20 >= args.target_score and len(recent_scores) == 20:
            print(f"[train] target reached: avg20={avg20:.1f} >= {args.target_score}; stopping.")
            break

        if args.max_time_sec is not None and (time.time() - t0) >= args.max_time_sec:
            elapsed = time.time() - t0
            print(f"[train] time budget reached: elapsed={elapsed:.1f}s >= {args.max_time_sec}s; stopping.")
            break

    print(f"[train] done. best score = {best_score}; checkpoint saved to {args.model_path}")
    env.close()


if __name__ == "__main__":
    main()
