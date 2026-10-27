# Robot Motion Control — Mojia

PyTorch model-based motion tracking with MuJoCo Warp for **Mojia v3.8, 23 DOF**.
The default model has no forearm support rods. Elbow coordinate zero represents
an approximately 90-degree physical bend; both elbow nominal values are zero.
The `g1` branch contains the G1 implementation. Robot weights are incompatible.

## Setup

```bash
conda activate mojia_rl
pip install -e .
```

The existing environment is verified with Python 3.11, PyTorch 2.7/CUDA 12.8,
MuJoCo 3.8.1 and an RTX 5090. See `requirements.txt` for dependencies.

## Training

```bash
python scripts/train.py --config configs/train.json --preflight
python scripts/train.py --config configs/train.json \
  --out runs/mojia --device cuda:0 --seed 42
```

This starts a fresh run, with a maximum of **50000 iterations**. Independent GPU
evaluation runs every 1000 iterations; best checkpoints and sustained-regression
stopping are enabled. To stop, press Ctrl+C once and wait for checkpoint saving.
To resume this same run later:

```bash
python scripts/train.py --config configs/train.json \
  --out runs/mojia --resume runs/mojia/latest.pt --device cuda:0 --seed 42
```

The 40 reference clips pass joint-limit, velocity and data-contract checks.
Residual leg collisions affect approximately 1.45% of frames at a 2cm review
threshold, and floor-height corrections reach 39.7cm. These are known reference
limitations; successful control or hardware deployment has not been established.
No reference clips are automatically filtered.

## Commands

All executable commands are in `scripts/`; reusable implementation is in
`robot_control/`. Run from the repository root. Use `--help` for parameters.

| Script | Operations |
| --- | --- |
| `train.py` | Train, resume, preflight, bounded verification |
| `evaluate.py` | `single`, `batch`, asynchronous `worker` |
| `visualize.py` | `reference`, policy `render`, website `serve` |
| `data.py` | `retarget`, `prepare`, `pipeline`, `check`, full-frame `audit` |
| `checkpoint.py` | `inspect`, explicit trusted-checkpoint `import` |

```bash
python scripts/evaluate.py single --checkpoint runs/mojia/best_learner.pt \
  --device cuda:0 --out runs/evaluation.json
python scripts/visualize.py reference --config configs/train.json \
  --motion dance1_subject2 fight1_subject2 --seconds 20 --out runs/reference_new
python scripts/visualize.py serve --directory runs/mojia_reference --port 8000
python scripts/data.py audit --config configs/train.json --out runs/audit_new
```

Reference playback sets joint poses directly; it does not test a learned policy.
Use `--seconds 0` for a full reference clip and `--start` for a later segment.
Video dependencies: `pip install -e '.[visualization]'`.

## Layout

- `configs/`: one robot contract, one training recipe, selected motion names.
- `assets/robots/mojia/`: the active URDF and required meshes.
- `robot_control/`: learning, physics, retargeting and data checks.
- `scripts/`: functional command entry points.
- `data/mojia_raw/`, `data/mojia/`: source CSVs and prepared references.
- `runs/mojia_reference/`, `runs/mojia_audit/`: current previews and full audit.
- `web/`: retained final G1 comparison showcase.
- `checkpoints/g1/best.pt`: retained successful G1 policy, used on the G1 branch.

Datasets, weights and run outputs remain local and are excluded from Git.
See [data preparation](docs/DATA.md), [model status](docs/MODELS.md),
[licenses and attribution](THIRD_PARTY_NOTICES.md).

## Robot contract checks

See [G1/Mojia coordinate, keypoint and training audit](docs/ROBOT_CONTRACT.md).
The active contract uses explicit root/torso names, distal hand points, and
future references t+1..t+10. GPU policy rendering supports `--device cuda:0`;
GPU evaluation supports `--detailed` keypoint and torso metrics. CPU contact
trajectories can differ materially from GPU trajectories, especially at reference
self-penetration. These remaining limitations are documented in the audit.

### 本地模型工作台

```bash
conda activate mojia_rl
python scripts/visualize.py studio --port 8765
```

打开 http://127.0.0.1:8765 。选择机器人、冻结的模型版本和动作，生成实际策略与青色参考叠加的视频。支持 CPU 对照、GPU 训练后端、1080p、限时预览与完整动作。限时没有跌倒不等于整段完成。

模型库在 `checkpoints/library/`，全动作评估优先按完成率、其次按回报推荐；不同机器人不直接排名。只有你在网页点击“确认保留”后，模型才归入 `checkpoints/library/approved/<robot>/<model>/`，附带指纹与评估信息。新保存模型可点击“扫描”。候选模型不自动删除。

接口并发受理：`GET /api/catalog`、`GET /api/jobs`；`POST /api/jobs` 接受 `model_id`、`kind`（`render` / `evaluate`）、`device`（`cpu` / `cuda:0`），播放另含 `motion`、`seconds`（0 为整段）、`height`（720 / 1080）。GPU 工作和渲染串行，训练正在运行时等待；CPU 评估可并行。视频和日志保存在 `runs/studio/jobs/`，服务重启后中断任务可重新提交。
