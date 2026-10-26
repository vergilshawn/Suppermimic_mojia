# Robot Motion Control — MOJIA

Model-based motion tracking with PyTorch and MuJoCo Warp. This branch targets **Mojia v3.8, 23 DOF**. Use `git switch g1` or `git switch mojia` to change robot implementations. Checkpoints are not interchangeable.

## Setup

```bash
conda activate mojia_rl
pip install -e .
```

Python 3.11 and an existing CUDA-compatible PyTorch installation are required. For RTX 5090, install the CUDA 12.8 wheel matching `requirements.txt`. No installation is needed in the already verified local environment.

## Train

Prepared references belong in `data/mojia/`; see [data preparation](docs/DATA.md). Check the recipe before launch:

```bash
python scripts/train.py --config configs/train.json --preflight
python scripts/train.py --config configs/train.json --out runs/mojia_new --device cuda:0 --seed 42
```

Training is currently stopped. The stopped Mojia run reached 812 iterations of its 50000-iteration limit. Resume only from its full checkpoint, not the learner weights.

To resume the retained local run (not started automatically):

```bash
python scripts/train.py --config configs/train.json --out runs/mojia \
  --resume runs/mojia/latest.pt --device cuda:0 --seed 42
```

## Evaluate

```bash
python scripts/evaluate.py single --checkpoint checkpoints/mojia/learner.pt --device cuda:0 --out runs/mojia_evaluation.json
```

`--device cpu --max-steps 100` provides a bounded functionality check; it is not a full-clip success test.

## Layout

- `robot_control/`: training, physics, policy, world model and evaluation.
- `configs/`: training and robot settings.
- `assets/robots/`: required robot assets where applicable.
- `scripts/`: train, evaluate, visualize, data and checkpoint entry points.
- `tests/`: focused checks.
- `web/`: final visual comparison, shared between branches.
- `data/`, `checkpoints/`, `runs/`: local files excluded from Git.

See [model status](docs/MODELS.md) and [attribution](THIRD_PARTY_NOTICES.md). Organizing the project does not establish hardware readiness or paper-level performance.

## Commands

All executable entry points live in `scripts/`. These scripts contain argument parsing and command logic directly; reusable training and physics code lives in `robot_control/runtime/`.
Run commands from the repository root. Each command supports `--help`.

| Entry | Operations |
| --- | --- |
| `scripts/train.py` | Train, `--resume`, `--preflight`, bounded verification |
| `scripts/evaluate.py` | `single`, `batch`, `worker` (existing asynchronous queue) |
| `scripts/visualize.py` | `render` a policy/reference video, `serve` the final website |
| `scripts/data.py` | `prepare` references, `check` dataset/robot contract, `retarget` one BVH, `pipeline` a BVH library |
| `scripts/checkpoint.py` | `inspect`, explicit trusted-checkpoint `import` after relocation |

```bash
python scripts/data.py check --config configs/train.json
python scripts/checkpoint.py inspect --path CHECKPOINT
python scripts/visualize.py render --checkpoint CHECKPOINT \
  --motion dance1_subject2 fight1_subject2 --seconds 12 \
  --width 1920 --height 1080 --out runs/preview
python scripts/visualize.py serve --directory web --port 8000
```

Replace `CHECKPOINT` with `checkpoints/mojia/learner.pt`.
Video rendering uses deterministic CPU physics and EGL rendering; it does not
start training. The translucent robot preserves world-space tracking errors.
`render_report.json` distinguishes falls, full completion and a limited preview.
Existing video files are protected against overwrite. Optional video dependencies:
`pip install -e '.[visualization]'` (FFmpeg is supplied by imageio-ffmpeg).

Robot source models and meshes are under `assets/robots/`. Prepared motion caches
remain under `data/`, outside Git. Checkpoint import changes path/contract metadata;
it is only for trusted, equivalent assets and does not convert between robots or
repair an incompatible model. Hardware-ready ONNX export is not provided by this
command.
