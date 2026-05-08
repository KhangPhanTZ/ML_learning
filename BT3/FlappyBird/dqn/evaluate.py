"""
Evaluate the trained DQN by visually rendering the game.

Loads ./best_dqn_model.pth, sets epsilon = 0 (pure exploitation), renders the
game with the original sprites, and verifies the agent clears the target
number of pipes (default: 50, per BT3 spec).

Usage:
    python evaluate.py
    python evaluate.py --episodes 5 --target-pipes 50 --fps 30
    python evaluate.py --model-path ./some_other.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from agent import DQNAgent
from game import FlappyBirdEnv

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "best_dqn_model.pth"


def parse_args():
    p = argparse.ArgumentParser(description="DQN evaluation for Flappy Bird (BT3).")
    p.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--target-pipes", type=int, default=50,
                   help="Required pipes-passed to consider an episode a 'pass' per BT3 spec.")
    p.add_argument("--max-steps", type=int, default=200_000,
                   help="Safety cap so a perfect agent doesn't loop forever.")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device     = {device}")
    print(f"[eval] model      = {model_path}")
    print(f"[eval] target     = clear at least {args.target_pipes} pipes (epsilon = 0)")

    env = FlappyBirdEnv(render_mode=True, fps=args.fps)
    agent = DQNAgent(device=device)
    agent.load(str(model_path))
    agent.policy_net.eval()

    successes = 0
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        info = {"score": 0, "frames": 0}
        for _ in range(args.max_steps):
            action = agent.select_action(state, greedy=True)  # pure exploitation, eps = 0
            state, reward, done, info = env.step(action)
            total_reward += reward
            env.render()
            if done:
                break

        score = info["score"]
        ok = score >= args.target_pipes
        successes += int(ok)
        tag = "PASS" if ok else "FAIL"
        print(f"[ep {ep}] pipes={score:4d}  reward={total_reward:8.2f}  frames={info['frames']:5d}  [{tag}]")

    print(f"[eval] {successes}/{args.episodes} episodes cleared >= {args.target_pipes} pipes.")
    env.close()


if __name__ == "__main__":
    main()
