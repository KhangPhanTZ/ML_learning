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

Change into the DQN folder and run training with recommended flags:

```bash
cd BT3/FlappyBird/dqn
# optional: remove existing checkpoint (skip if you want to keep backup)
rm -f best_dqn_model.pth

python train.py \
	--max-time-sec 7200 \
	--episodes 1000000 \
	--max-steps 10000 \
	--lr 5e-4 \
	--gamma 0.99 \
	--batch-size 128 \
	--memory-size 200000 \
	--eps-decay-steps 100000 \
	--target-update-steps 1000 \
	--crash-penalty -10 \
	--target-score 999999 \
	--log-every 50 \
	--seed 7 \
	--model-path best_dqn_model.pth
```

To visualize a trained agent (uses `./best_dqn_model.pth` by default):

```bash
cd BT3/FlappyBird/dqn
python visualize_agent.py                # uses ./best_dqn_model.pth
python visualize_agent.py /path/to/x.pth # custom checkpoint path
```

## Notes

- The `requirements.txt` for the DQN is at `BT3/FlappyBird/dqn/requirements.txt`.
- Adjust flags (learning rate, batch size, memory size, etc.) as desired.
- Training can be long — consider using smaller `--episodes` or `--max-time-sec` for experiments.
