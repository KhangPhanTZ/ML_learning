"""
Flappy Bird RL environment.

Refactored from the LeonMarqs/Flappy-bird-python reference into a clean
step()/reset() interface for DQN training.

- render_mode=False  -> pure-physics, no pygame import, no display required.
                       Used during training for maximum speed.
- render_mode=True   -> pygame is initialised and the original sprites
                       (../Flappy-bird-python/assets/sprites/*) are loaded
                       and drawn. Used during evaluation.

Constants are kept identical to the reference game so the agent's learned
policy transfers 1:1 between training and rendered evaluation.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GROUND_HEIGHT = 100
PLAYABLE_HEIGHT = SCREEN_HEIGHT - GROUND_HEIGHT  # 500 px the bird can fly in.

GAME_SPEED = 15        # Horizontal scroll speed of pipes/ground (px / frame).
GRAVITY = 2.5          # Velocity added to bird each frame.
JUMP_SPEED = -20.0     # Bird vertical velocity right after a flap (negative = up).

PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150
PIPE_X_START = 800                       # First pipe pair x at game start.
PIPE_SPAWN_DISTANCE = SCREEN_WIDTH       # Horizontal spacing between pipe pairs.
PIPE_SIZE_MIN = 100                      # Bottom-pipe height range (matches reference).
PIPE_SIZE_MAX = 300

BIRD_X = SCREEN_WIDTH // 6               # Bird's fixed x position (≈ 66).
BIRD_WIDTH = 34                          # Approx. bluebird sprite width.
BIRD_HEIGHT = 24                         # Approx. bluebird sprite height.

ACTION_NOOP = 0
ACTION_FLAP = 1
NUM_ACTIONS = 2

ASSETS_DIR = Path(__file__).resolve().parent.parent / "Flappy-bird-python" / "assets"


@dataclass
class PipePair:
    """A vertically-aligned pair of pipes scrolling left across the screen.

    Geometry (pixels, y grows downward):
        top_pipe_bottom_y  = bottom edge of upper pipe (= top of the gap)
        bottom_pipe_top_y  = top edge of lower pipe    (= bottom of the gap)
    The bird must keep its y between these two values to survive.
    """

    x: float
    top_pipe_bottom_y: float
    bottom_pipe_top_y: float
    passed: bool = False


def _spawn_pipe_pair(x: float) -> PipePair:
    # Mirrors `get_random_pipes` in the reference flappy.py.
    size = random.randint(PIPE_SIZE_MIN, PIPE_SIZE_MAX)
    bottom_pipe_top_y = SCREEN_HEIGHT - size               # 600 - size
    top_pipe_bottom_y = SCREEN_HEIGHT - size - PIPE_GAP    # 450 - size
    return PipePair(x=x, top_pipe_bottom_y=top_pipe_bottom_y, bottom_pipe_top_y=bottom_pipe_top_y)


class FlappyBirdEnv:
    """Single-agent Flappy Bird environment with a tiny gym-like API."""

    def __init__(
        self,
        render_mode: bool = False,
        fps: int = 30,
        crash_penalty: float = -1.0,
        survive_reward: float = 0.1,
        pass_reward: float = 1.0,
    ):
        self.render_mode = render_mode
        self.fps = fps
        self.crash_penalty = crash_penalty
        self.survive_reward = survive_reward
        self.pass_reward = pass_reward

        self._pygame = None
        self._screen = None
        self._clock = None
        self._sprites = None
        self._font = None

        if self.render_mode:
            self._init_pygame()
        self.reset()

    # -------------------------------------------------------------- pygame
    def _init_pygame(self):
        import pygame  # Imported lazily so headless training has zero pygame cost.

        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        pygame.font.init()
        self._pygame = pygame
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird DQN - Evaluation")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, 28)
        self._sprites = self._load_sprites()

    def _load_sprites(self):
        pygame = self._pygame
        sprites_dir = ASSETS_DIR / "sprites"
        if not sprites_dir.exists():
            raise FileNotFoundError(
                f"Sprite directory not found: {sprites_dir}. "
                "Ensure the Flappy-bird-python reference repo is at "
                "BT3/FlappyBird/Flappy-bird-python/."
            )

        bg = pygame.image.load(str(sprites_dir / "background-day.png")).convert()
        bg = pygame.transform.scale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        bird = pygame.image.load(str(sprites_dir / "bluebird-midflap.png")).convert_alpha()

        pipe = pygame.image.load(str(sprites_dir / "pipe-green.png")).convert_alpha()
        pipe = pygame.transform.scale(pipe, (PIPE_WIDTH, PIPE_HEIGHT))
        pipe_top = pygame.transform.flip(pipe, False, True)

        ground = pygame.image.load(str(sprites_dir / "base.png")).convert_alpha()
        ground = pygame.transform.scale(ground, (SCREEN_WIDTH, GROUND_HEIGHT))

        return {"bg": bg, "bird": bird, "pipe": pipe, "pipe_top": pipe_top, "ground": ground}

    # ---------------------------------------------------------------- API
    def reset(self):
        self.bird_y = SCREEN_HEIGHT / 2.0
        self.bird_vy = 0.0
        self.score = 0      # Number of pipe pairs cleared.
        self.frames = 0
        self.done = False

        # Always keep two pipe pairs alive on screen, just like the reference.
        self.pipes: list[PipePair] = [
            _spawn_pipe_pair(PIPE_X_START),
            _spawn_pipe_pair(PIPE_X_START + PIPE_SPAWN_DISTANCE),
        ]
        return self._get_state()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called after termination; call reset() first.")

        if action == ACTION_FLAP:
            self.bird_vy = JUMP_SPEED

        # Physics (per-frame, matches reference timing at clock.tick(15)).
        self.bird_vy += GRAVITY
        self.bird_y += self.bird_vy
        self.frames += 1

        # Scroll pipes left.
        for p in self.pipes:
            p.x -= GAME_SPEED

        # Recycle the leftmost pipe pair once it's fully off-screen.
        if self.pipes[0].x + PIPE_WIDTH < 0:
            self.pipes.pop(0)
            new_x = self.pipes[-1].x + PIPE_SPAWN_DISTANCE
            self.pipes.append(_spawn_pipe_pair(new_x))

        # Reward shaping.
        reward = self.survive_reward
        for p in self.pipes:
            # Award once when the bird's left edge fully clears the pipe's right edge.
            if (not p.passed) and (p.x + PIPE_WIDTH < BIRD_X):
                p.passed = True
                self.score += 1
                reward += self.pass_reward

        # Crash check (overrides positive reward — the spec asks for a clean crash signal).
        if self._check_collision():
            reward = self.crash_penalty
            self.done = True

        return self._get_state(), reward, self.done, {"score": self.score, "frames": self.frames}

    def render(self):
        if not self.render_mode:
            return
        pg = self._pygame

        # Drain events so the OS doesn't think the window is hung.
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                raise SystemExit("Window closed by user.")

        self._screen.blit(self._sprites["bg"], (0, 0))

        for p in self.pipes:
            # Top pipe sprite is flipped; its top-left y = bottom_edge - PIPE_HEIGHT.
            top_pipe_y = p.top_pipe_bottom_y - PIPE_HEIGHT
            self._screen.blit(self._sprites["pipe_top"], (p.x, top_pipe_y))
            self._screen.blit(self._sprites["pipe"], (p.x, p.bottom_pipe_top_y))

        self._screen.blit(self._sprites["ground"], (0, SCREEN_HEIGHT - GROUND_HEIGHT))
        self._screen.blit(self._sprites["bird"], (BIRD_X, self.bird_y))

        # HUD: pipes passed + the 3 raw features the network is reading.
        pipe = self._next_pipe()
        hud_lines = [
            f"Pipes: {self.score}",
            f"bird_y     = {self.bird_y:6.1f}",
            f"top_pipe_y = {pipe.top_pipe_bottom_y:6.1f}",
            f"bot_pipe_y = {pipe.bottom_pipe_top_y:6.1f}",
        ]
        for i, line in enumerate(hud_lines):
            surf = self._font.render(line, True, (255, 255, 255))
            self._screen.blit(surf, (8, 8 + i * 22))

        pg.display.flip()
        self._clock.tick(self.fps)

    def close(self):
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None

    # ------------------------------------------------------------ internals
    def _next_pipe(self) -> PipePair:
        """The closest pipe pair the bird has not yet fully cleared.

        The bird is at fixed BIRD_X. A pipe pair is "ahead" while its right
        edge has not crossed the bird's left edge.
        """
        for p in self.pipes:
            if p.x + PIPE_WIDTH >= BIRD_X:
                return p
        # Fallback safeguard: should never trigger because we always keep
        # two pipe pairs alive and recycle them before they're fully passed.
        return self.pipes[-1]

    def _get_state(self) -> np.ndarray:
        """Return [bird_y, top_pipe_y, bottom_pipe_y] normalised to [0, 1].

        Normalisation by the playable height keeps inputs in a stable range
        so the small 24-neuron MLP doesn't have to learn a wide scale.
        """
        p = self._next_pipe()
        return np.array(
            [
                self.bird_y / PLAYABLE_HEIGHT,
                p.top_pipe_bottom_y / PLAYABLE_HEIGHT,
                p.bottom_pipe_top_y / PLAYABLE_HEIGHT,
            ],
            dtype=np.float32,
        )

    def _check_collision(self) -> bool:
        # Ceiling.
        if self.bird_y < 0:
            return True
        # Ground.
        if self.bird_y + BIRD_HEIGHT >= PLAYABLE_HEIGHT:
            return True

        bird_left = BIRD_X
        bird_right = BIRD_X + BIRD_WIDTH
        bird_top = self.bird_y
        bird_bot = self.bird_y + BIRD_HEIGHT

        for p in self.pipes:
            pipe_left = p.x
            pipe_right = p.x + PIPE_WIDTH
            # No horizontal overlap -> no collision possible with this pair.
            if bird_right < pipe_left or bird_left > pipe_right:
                continue
            # Horizontal overlap: bird must fit strictly inside the gap.
            if bird_top < p.top_pipe_bottom_y or bird_bot > p.bottom_pipe_top_y:
                return True
        return False
