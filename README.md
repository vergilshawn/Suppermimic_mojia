# Robot Motion Control — G1

Model-based motion tracking with PyTorch and MuJoCo Warp. This branch targets **Unitree G1, 29 DOF**. Use `git switch g1` or `git switch mojia` to change robot implementations. Checkpoints are not interchangeable.

## Setup

```bash
conda activate mojia_rl
pip install -e .
```

Python 3.11 and an existing CUDA-compatible PyTorch installation are required. For RTX 5090, install the CUDA 12.8 wheel matching `requirements.txt`. No installation is needed in the already verified local environment.

## Train

Prepared references belong in `data/g1/`; see [data preparation](docs/DATA.md). Check the recipe before launch:

```bash
python train.py --config configs/train.json --preflight
python train.py --config configs/train.json --out runs/g1_new --device cuda:0 --seed 42
```

Training is currently stopped. The retained G1 model is the best policy-only snapshot at iteration 30000.

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/g1/best.pt --device cuda:0 --out runs/g1_evaluation.json
```

`--device cpu --max-steps 100` provides a bounded functionality check; it is not a full-clip success test.

## Layout

- `robot_control/`: training, physics, policy, world model and evaluation.
- `configs/`: training and robot settings.
- `robots/`: required robot assets where applicable.
- `tools/`: data and checkpoint utilities.
- `tests/`: focused checks.
- `web/`: final visual comparison, shared between branches.
- `data/`, `checkpoints/`, `runs/`: local files excluded from Git.

See [model status](docs/MODELS.md) and [attribution](THIRD_PARTY_NOTICES.md). Organizing the project does not establish hardware readiness or paper-level performance.
