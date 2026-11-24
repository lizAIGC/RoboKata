# Offline recorded-pair inference

Use the same verified work directory and base model as training. Set `WORK`,
`MODEL` and `EMPTY_TEXT` as in TRAINING.md. `OUTPUT` must be a new JSON filename
whose parent directory already exists.

## A checkpoint from your own training run

```bash
CHECKPOINT=/absolute/path/to/run/checkpoints/epoch_001.pt
OUTPUT=/absolute/path/to/existing/output/prediction.json
CUDA_VISIBLE_DEVICES=0 python scripts/robokata.py infer \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY_TEXT" \
  --demo-features "$WORK/demo-dino.pt" --checkpoint "$CHECKPOINT" \
  --pair-index 0 --seed 1 --output "$OUTPUT"
```

## Published adapter

Use a separately supplied compatible adapter.
Keep `adapter.json` beside `adapter.safetensors` and supply the repository-level
`adapter_config.json`:

```bash
ADAPTER=/absolute/path/to/selected-epoch/adapter.safetensors
ADAPTER_CONFIG=/absolute/path/to/adapter_config.json
CUDA_VISIBLE_DEVICES=0 python scripts/robokata.py infer \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY_TEXT" \
  --demo-features "$WORK/demo-dino.pt" --checkpoint "$ADAPTER" \
  --adapter-config "$ADAPTER_CONFIG" --pair-index 0 --seed 1 --output "$OUTPUT"
```

**Portability limitation:** this command requires the exact compatible verified
cache identities. The existing loader checks the training cache receipt, plan,
normalization, demo-feature hashes, source recording hashes and latent file
size/timestamps. Local `.pt` checkpoints also check the absolute base-model path.
A copied or regenerated cache can fail these checks. Use matching data and receipts; do not disable checks to force mismatched assets to load.

Output: 16 × 10 finite absolute world-hand action targets, 50 Hz control timing,
checkpoint/pair metadata and seed. Query semantic features supervise training
only; inference uses the demonstration feature cache. `--disable-demo-attention`
is an A-off ablation that retains VAE demo context, not a no-demo baseline.

Only load trusted `.pt` files: the upstream checkpoint format loads pickled
training metadata. Public inference should prefer the selected Safetensors adapter.
