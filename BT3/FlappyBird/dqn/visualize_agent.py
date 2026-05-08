"""
visualize_agent.py
==================

Standalone script that loads a trained Flappy Bird DQN checkpoint
(`best_dqn_model.pth`) and runs the agent VISUALLY in pygame at a steady
frame-rate so you can record it on video.

This file is self-contained: it does NOT import from game.py / model.py /
agent.py. Drop it into the same folder as `best_dqn_model.pth` (and the
sibling `Flappy-bird-python/assets/` sprite folder) and run:

    python visualize_agent.py
    python visualize_agent.py /path/to/best_dqn_model.pth   # custom path
    python visualize_agent.py --max-pipes 80                # force reset after 80 pipes
    python visualize_agent.py --episodes 10 --max-pipes 60  # 10 different maps then quit

Each episode is reseeded so the pipe layout differs from run to run,
making it easy to visualise the agent generalising across many "maps".
Press SPACE to skip to the next map, ESC / close window to quit.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn

# =====================================================================
# Constants. These MUST match the values used during training so the
# loaded policy is interpreted in the same world it learned on.
# =====================================================================
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GROUND_HEIGHT = 100
PLAYABLE_HEIGHT = SCREEN_HEIGHT - GROUND_HEIGHT  # 500 px - the same divisor used to normalise the state at train time.

GAME_SPEED = 15        # pipe scroll speed (px / step)
GRAVITY = 2.5          # added to bird's vertical velocity every step
JUMP_SPEED = -20.0     # vertical velocity right after a flap (negative = up)

PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150
PIPE_X_START = 800
PIPE_SPAWN_DISTANCE = SCREEN_WIDTH
PIPE_SIZE_MIN = 100
PIPE_SIZE_MAX = 300

BIRD_X = SCREEN_WIDTH // 6   # 66 - bird's fixed x position
BIRD_WIDTH = 34
BIRD_HEIGHT = 24

# Render frame-rate (also = physics step-rate, since we do one physics
# update per rendered frame). 30 FPS is comfortable for video recording.
FPS = 30

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = HERE / "best_dqn_model.pth"
ASSETS_DIR = HERE.parent / "Flappy-bird-python" / "assets" / "sprites"


# =====================================================================
# Network architecture (must match exactly what was trained).
#   Input:   3 features [bird_y, top_pipe_y, bottom_pipe_y]   (normalised)
#   Hidden:  Linear(3, 24) -> ReLU -> Linear(24, 24) -> ReLU
#   Output:  Linear(24, 2) (raw Q-values for [NOOP, FLAP])
# Layer attribute names (`fc1`, `fc2`, `out`) must match the names used
# at training time so torch can load the state_dict by key.
# =====================================================================
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out = nn.Linear(24, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# =====================================================================
# Pipe pair geometry. Mirrors the reference game's get_random_pipes():
#   bottom_pipe_top_y = SCREEN_HEIGHT - size       (top edge of lower pipe)
#   top_pipe_bottom_y = SCREEN_HEIGHT - size - GAP (bottom edge of upper pipe)
# So the bird's y must satisfy:
#   top_pipe_bottom_y < bird_y < bottom_pipe_top_y
# to remain inside the gap.
# =====================================================================
class PipePair:
    __slots__ = ("x", "top_pipe_bottom_y", "bottom_pipe_top_y", "passed")

    def __init__(self, x: float, top_y: float, bot_y: float):
        self.x = x
        self.top_pipe_bottom_y = top_y
        self.bottom_pipe_top_y = bot_y
        self.passed = False


def spawn_pipe_pair(x: float) -> PipePair:
    size = random.randint(PIPE_SIZE_MIN, PIPE_SIZE_MAX)
    bottom_top = SCREEN_HEIGHT - size           # 600 - size
    top_bottom = SCREEN_HEIGHT - size - PIPE_GAP  # 450 - size
    return PipePair(x, top_bottom, bottom_top)


def next_pipe_ahead(pipes: list[PipePair]) -> PipePair:
    """Return the closest pipe pair the bird has NOT yet fully cleared.

    A pipe pair is "ahead" of the bird while its right edge has not
    crossed the bird's left edge. We pick the first such pair (lowest x);
    once the bird passes a pair, this function naturally returns the next
    one in the list.
    """
    for p in pipes:
        if p.x + PIPE_WIDTH >= BIRD_X:
            return p
    # Safety: should never happen because we always keep two pipe pairs
    # alive on screen and recycle them before they're fully passed.
    return pipes[-1]


def extract_state(bird_y: float, pipes: list[PipePair]) -> np.ndarray:
    """Build the 3-feature input tensor in the SAME order/scale as training."""
    p = next_pipe_ahead(pipes)
    return np.array(
        [
            bird_y / PLAYABLE_HEIGHT,
            p.top_pipe_bottom_y / PLAYABLE_HEIGHT,
            p.bottom_pipe_top_y / PLAYABLE_HEIGHT,
        ],
        dtype=np.float32,
    )


def check_collision(bird_y: float, pipes: list[PipePair]) -> bool:
    # Ceiling.
    if bird_y < 0:
        return True
    # Ground.
    if bird_y + BIRD_HEIGHT >= PLAYABLE_HEIGHT:
        return True

    bird_left = BIRD_X
    bird_right = BIRD_X + BIRD_WIDTH
    bird_top = bird_y
    bird_bot = bird_y + BIRD_HEIGHT

    for p in pipes:
        pipe_left = p.x
        pipe_right = p.x + PIPE_WIDTH
        # No horizontal overlap with this pair -> can't collide with it.
        if bird_right < pipe_left or bird_left > pipe_right:
            continue
        # Horizontal overlap: bird must fit inside the gap or it crashed.
        if bird_top < p.top_pipe_bottom_y or bird_bot > p.bottom_pipe_top_y:
            return True
    return False


# =====================================================================
# Loading helpers
# =====================================================================
def load_model(path: Path, device: torch.device) -> DQN:
    if not path.exists():
        sys.exit(f"[error] checkpoint not found: {path}")

    # weights_only=False so we can load checkpoints saved as a dict
    # ({'policy_state_dict': ..., 'optimizer_state_dict': ..., 'train_steps': ...}).
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        state_dict = ckpt["policy_state_dict"]                       # saved via DQNAgent.save()
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]                        # alternative naming
    elif isinstance(ckpt, dict) and any(k.startswith("fc1") for k in ckpt):
        state_dict = ckpt                                            # raw state_dict
    else:
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__
        sys.exit(f"[error] unrecognised checkpoint format. Top-level keys: {keys}")

    model = DQN().to(device)
    model.load_state_dict(state_dict)
    model.eval()  # disable dropout / batchnorm modes (none here, but good practice)
    return model


def load_sprites():
    if not ASSETS_DIR.exists():
        sys.exit(
            f"[error] sprite directory not found: {ASSETS_DIR}\n"
            f"        expected the LeonMarqs reference repo at "
            f"{ASSETS_DIR.parent.parent}"
        )
    bg = pygame.image.load(str(ASSETS_DIR / "background-day.png")).convert()
    bg = pygame.transform.scale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

    bird = pygame.image.load(str(ASSETS_DIR / "bluebird-midflap.png")).convert_alpha()

    pipe = pygame.image.load(str(ASSETS_DIR / "pipe-green.png")).convert_alpha()
    pipe = pygame.transform.scale(pipe, (PIPE_WIDTH, PIPE_HEIGHT))
    pipe_top = pygame.transform.flip(pipe, False, True)

    ground = pygame.image.load(str(ASSETS_DIR / "base.png")).convert_alpha()
    ground = pygame.transform.scale(ground, (SCREEN_WIDTH, GROUND_HEIGHT))

    return {"bg": bg, "bird": bird, "pipe": pipe, "pipe_top": pipe_top, "ground": ground}


# =====================================================================
# Main loop
# =====================================================================
def parse_args(argv: list[str]) -> tuple[Path, int, int, int]:
    """Tiny ad-hoc parser so the script stays dependency-free.

    Returns (model_path, max_pipes, n_episodes, base_seed).
    max_pipes <= 0 means "no cap" (run until the bird crashes).
    n_episodes <= 0 means "loop forever".
    """
    model_path = DEFAULT_MODEL_PATH
    max_pipes = 60
    n_episodes = 0
    base_seed = 2000

    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--max-pipes" and i + 1 < len(argv):
            max_pipes = int(argv[i + 1]); i += 2
        elif a == "--episodes" and i + 1 < len(argv):
            n_episodes = int(argv[i + 1]); i += 2
        elif a == "--seed" and i + 1 < len(argv):
            base_seed = int(argv[i + 1]); i += 2
        elif a in ("-h", "--help"):
            print(__doc__); sys.exit(0)
        else:
            model_path = Path(a); i += 1
    return model_path, max_pipes, n_episodes, base_seed


def main():
    model_path, max_pipes, n_episodes, base_seed = parse_args(sys.argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird DQN - Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16, bold=True)
    big_font = pygame.font.SysFont("monospace", 28, bold=True)
    sprites = load_sprites()

    model = load_model(model_path, device)
    print(f"[viz] device   = {device}")
    if device.type == "cuda":
        print(f"[viz] gpu      = {torch.cuda.get_device_name(0)}")
    print(f"[viz] model    = {model_path}")
    cap_str = f"cap={max_pipes} pipes" if max_pipes > 0 else "no cap"
    ep_str = f"{n_episodes} episodes" if n_episodes > 0 else "infinite episodes"
    print(f"[viz] running  = {ep_str}, {cap_str}, base_seed={base_seed}, {FPS} FPS")
    print(f"[viz] keys     = SPACE: skip map  |  ESC: quit\n")

    episode = 0
    best_score = 0

    try:
        while n_episodes <= 0 or episode < n_episodes:
            episode += 1

            # ----- reset episode with a fresh, distinct pipe layout -----
            ep_seed = base_seed + episode
            random.seed(ep_seed)

            bird_y = SCREEN_HEIGHT / 2.0
            bird_vy = 0.0
            score = 0
            frames = 0
            pipes: list[PipePair] = [
                spawn_pipe_pair(PIPE_X_START),
                spawn_pipe_pair(PIPE_X_START + PIPE_SPAWN_DISTANCE),
            ]

            running = True
            last_action = 0
            last_q = (0.0, 0.0)
            end_reason = "crash"

            while running:
                # ----- input handling -----
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        running = False
                        end_reason = "skipped"

                # ----- 1. extract state from the current Pygame world -----
                state = extract_state(bird_y, pipes)

                # ----- 2. inference: epsilon = 0, pure exploitation via argmax -----
                with torch.no_grad():
                    s = torch.from_numpy(state).unsqueeze(0).to(device)
                    q = model(s)
                    action = int(q.argmax(dim=1).item())
                    last_q = (float(q[0, 0].item()), float(q[0, 1].item()))
                last_action = action

                # ----- 3. apply action -----
                if action == 1:           # 1 = FLAP
                    bird_vy = JUMP_SPEED

                # ----- 4. physics -----
                bird_vy += GRAVITY
                bird_y += bird_vy
                frames += 1

                # ----- 5. scroll pipes & recycle off-screen ones -----
                for p in pipes:
                    p.x -= GAME_SPEED
                if pipes[0].x + PIPE_WIDTH < 0:
                    pipes.pop(0)
                    pipes.append(spawn_pipe_pair(pipes[-1].x + PIPE_SPAWN_DISTANCE))

                # ----- 6. score: count once when bird fully clears a pipe -----
                for p in pipes:
                    if (not p.passed) and (p.x + PIPE_WIDTH < BIRD_X):
                        p.passed = True
                        score += 1

                # ----- 7. crash check (or pipe cap reached) -----
                if check_collision(bird_y, pipes):
                    running = False
                elif max_pipes > 0 and score >= max_pipes:
                    running = False
                    end_reason = "cap"

                # ----- 8. render -----
                screen.blit(sprites["bg"], (0, 0))
                for p in pipes:
                    top_y_screen = p.top_pipe_bottom_y - PIPE_HEIGHT
                    screen.blit(sprites["pipe_top"], (p.x, top_y_screen))
                    screen.blit(sprites["pipe"], (p.x, p.bottom_pipe_top_y))
                screen.blit(sprites["ground"], (0, SCREEN_HEIGHT - GROUND_HEIGHT))
                screen.blit(sprites["bird"], (BIRD_X, bird_y))

                # HUD: makes the input/output of the network visible on the recording.
                np_pipe = next_pipe_ahead(pipes)
                act_str = "FLAP" if last_action == 1 else "NOOP"
                hud_lines = [
                    f"Map #{episode}  seed={ep_seed}  Pipes {score}  Best {best_score}",
                    f"bird_y     = {bird_y:6.1f}",
                    f"top_pipe_y = {np_pipe.top_pipe_bottom_y:6.1f}",
                    f"bot_pipe_y = {np_pipe.bottom_pipe_top_y:6.1f}",
                    f"Q[NOOP]={last_q[0]:+.2f}  Q[FLAP]={last_q[1]:+.2f}",
                    f"Action    -> {act_str}",
                ]
                for i, line in enumerate(hud_lines):
                    surf = font.render(line, True, (255, 255, 255))
                    # Drop shadow for readability against the sky background.
                    shadow = font.render(line, True, (0, 0, 0))
                    screen.blit(shadow, (9, 9 + i * 18))
                    screen.blit(surf, (8, 8 + i * 18))

                pygame.display.flip()
                clock.tick(FPS)

            # ----- end of episode -----
            if score > best_score:
                best_score = score
            tag = {"crash": "CRASH", "cap": "PASSED", "skipped": "SKIPPED"}[end_reason]
            print(
                f"[map {episode:4d}] seed={ep_seed}  pipes={score:4d}  frames={frames:5d}  "
                f"{tag}  best={best_score}",
                flush=True,
            )

            # Brief overlay so the result is readable in the recording.
            banner_text = (
                f"PASSED {score} pipes" if end_reason == "cap"
                else (f"SKIPPED ({score} pipes)" if end_reason == "skipped"
                      else f"GAME OVER  -  {score} pipes")
            )
            banner_color = (80, 220, 80) if end_reason == "cap" else (255, 80, 80)
            overlay = big_font.render(banner_text, True, banner_color)
            shadow = big_font.render(banner_text, True, (0, 0, 0))
            ox = SCREEN_WIDTH // 2 - overlay.get_width() // 2
            oy = SCREEN_HEIGHT // 2 - overlay.get_height() // 2
            screen.blit(shadow, (ox + 2, oy + 2))
            screen.blit(overlay, (ox, oy))
            pygame.display.flip()
            pygame.time.delay(800)

    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
