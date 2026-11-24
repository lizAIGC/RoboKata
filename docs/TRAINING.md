# Data preparation and training

Run commands from the release root in the selected training environment.
Set these variables to your own locations; paths below are placeholders.
`MODEL` must point to the compatible pretrained backbone described in
[third-party dependencies](THIRD_PARTY.md#pretrained-backbone).

```bash
DATA=/absolute/path/to/recordings
MIXED=/absolute/path/to/mixed25000-manifest
WORK=/absolute/path/to/new/cache
MODEL=/absolute/path/to/pretrained-backbone
DINO=/absolute/path/to/dinov2-base
EMPTY_TEXT=/absolute/path/to/empty_text_emb.pt
RUN=/absolute/path/to/new/run
```

The reference recipe uses 12,000 official v2.1 pairs and 13,000 selected old pairs,
with a separate 1,200-pair evaluation split. Supply the matching recording
release and mixed manifest. Data assets are supplied separately; do not
substitute the old validation split, which overlaps this training set.
Demo inputs use the external/head camera; query inputs use external + left wrist.

```bash
python scripts/robokata.py prepare plan --root "$DATA" --mixed-manifest "$MIXED" --work "$WORK" --dual-view --demo-top-view
python scripts/robokata.py check-data --work "$WORK" --workers 8 --output "$WORK/data-check.json"
python scripts/robokata.py prepare stats --work "$WORK"
python scripts/robokata.py prepare cache --work "$WORK" --model "$MODEL"
python scripts/robokata.py prepare verify --work "$WORK" --model "$MODEL"
python scripts/robokata.py demo-features --work "$WORK" --encoder "$DINO" --output "$WORK/demo-dino.pt" --device cuda:0 --bf16
python scripts/robokata.py semantic-features --work "$WORK" --encoder "$DINO" --output "$WORK/query-dino.pt" --device cuda:0 --bf16
python scripts/robokata.py semantic-features --work "$WORK" --encoder "$DINO" --output "$WORK/query-dino.pt" --verify
```

Select the four available devices for your run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/robokata.py --gpus 4 train \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY_TEXT" \
  --run-dir "$RUN" --demo-features "$WORK/demo-dino.pt" \
  --semantic-features "$WORK/query-dino.pt"
```

The launcher applies `configs/train.json` and enables foresight indexing.
Explicit CLI arguments override recipe values. Defaults: 20 epochs, batch 16
per GPU, LoRA rank 106, learning rate 5e-5, warmup 0, seed 1, demo dropout 0.3,
semantic and indexing loss weights 0.5 each. GPU selection is never automatic.

To resume, repeat the same command with `--resume-from "$RUN/checkpoints/latest.pt"`.
Keep the same caches, GPU count, batch and schedule. Training checkpoints include
restart state; use trusted checkpoints only. Weights are stored locally. First validate a short run using `--max-steps 1` in a separate run directory;
that checks execution, not convergence or paper-level performance.
