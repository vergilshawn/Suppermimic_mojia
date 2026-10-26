# Models and data

Model weights and datasets are kept outside Git (`checkpoints/`, `data/`, `runs/`). Publish selected inference weights as GitHub Release attachments, with SHA256 checksums and model cards. Full optimizer/replay checkpoints are local training recovery artifacts, not inference downloads.

G1: best policy at iteration 30000, 39/40 complete LAFAN clips, mean normalized return 0.910466. It is a policy-only snapshot, not a complete resumable training state.

Mojia: training stopped normally at iteration 812; learner and full recovery state retained. This is an early training checkpoint, not a validated control policy or hardware-ready model.

Checkpoint import after this directory refactor records original metadata in `packaging_origin` and updates only configuration paths, source fingerprints and equivalent simulator contracts. Tensor contents are unchanged. This tool is for trusted local checkpoints; it is not a compatibility converter between robot models or algorithms.
