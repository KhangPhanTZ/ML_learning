# ML_learning

## Requirements

- Python 3.8+ (use a virtual environment). To install Python deps for the FlappyBird DQN project run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r BT3/FlappyBird/dqn/requirements.txt
```

If you prefer, install specific packages manually (examples):

```bash
pip install torch numpy gym pygame matplotlib
```

## Run the FlappyBird DQN (BT3)

Change into the DQN folder and run training. The flags below were
empirically tuned on a GTX 1650: in a 90-second smoke test they reached
8 pipes, while the previous (slower) config reached 0. Total budget
below is 2 hours; the run stops cleanly between episodes when the
wall-clock cap is hit.

```bash
cd BT3/FlappyBird/dqn
# optional: remove existing checkpoint (skip if you want to keep backup)
rm -f best_dqn_model.pth

python train.py \
	--max-time-sec 7200 \
	--episodes 1000000 \
	--max-steps 10000 \
	--lr 1e-3 \
	--gamma 0.95 \
	--batch-size 64 \
	--memory-size 10000 \
	--eps-decay-steps 5000 \
	--target-update-steps 500 \
	--crash-penalty -1.0 \
	--target-score 999999 \
	--log-every 100 \
	--seed 7 \
	--model-path best_dqn_model.pth
```

Why these values matter (in case you tune further):

- `--eps-decay-steps 5000` — the bird crashes within ~25 frames during
  random play, but the first pipe arrives ~49 frames in. If exploration
  decays too slowly, the replay buffer fills with crash-only transitions
  before the agent ever sees a `+1.0` pass-pipe reward. Decaying over
  ~5k gradient steps lets the agent exit pure random play in under a
  minute and start learning from real pipe-passing transitions.
- `--memory-size 10000` — keeps the replay buffer fresh. A larger
  buffer (e.g. 200k) holds onto stale early-random transitions long
  after they're useful, which biases learning toward "everything ends
  in a crash".
- `--crash-penalty -1.0` — matches the spec scale (survive +0.1, pass
  +1.0, crash -1.0). A heavier penalty like -10 makes Q-values collapse
  deep into the negative range and the policy degenerates to a single
  fixed action.
- `--gamma 0.95` — the relevant decisions (when to flap) pay off within
  ~30 frames; a longer horizon (0.99) just adds variance.
- `--target-update-steps 500` and `--lr 1e-3` — keeps the target
  network and policy network close enough that learning converges fast
  on this tiny 24-24 MLP.

To visualize a trained agent (uses `./best_dqn_model.pth` by default):

```bash
cd BT3/FlappyBird/dqn
python visualize_agent.py                # uses ./best_dqn_model.pth
python visualize_agent.py /path/to/x.pth # custom checkpoint path
```

`visualize_agent.py` is pure inference — `model.eval()` + `torch.no_grad()`,
no optimizer, no `torch.save`. It will not modify your `.pth` file no
matter how long you let it run.

## Notes

- The `requirements.txt` for the DQN is at `BT3/FlappyBird/dqn/requirements.txt`.
- For shorter experiments, lower `--max-time-sec` (e.g. 600 for 10 min).
- To resume from an existing checkpoint instead of starting fresh, add
  `--resume`. Skip the `rm -f best_dqn_model.pth` line in that case.
