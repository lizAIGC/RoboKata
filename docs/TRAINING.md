# Training the main RoboKata model

## Data and split

Use the original verified Franka recording release with this layout:

```text
DATA/
  release_status.json
  verified-train3000.json
  train3000/
    bundle.json
    manifest.json
    pairing.json
    records/<recording-id>/policy_recording.npz
```

The preparation step verifies the release indices and uses split seed `20261105`. It holds out 50 of the 500 physical initial-state families before forming the within-split pair plan. Each family contains six task recordings. Training therefore has 2,700 recordings from 450 families; validation has 300 recordings from 50 families.

Release validation uses the recording and pair counts, bundle digest, verified-release receipt and index hashes. It does not require a particular hosting account or repository name in the release metadata.

The release's task-matched directed pairs are restricted to each split. Demo and Query must be different recordings from different families with the same complete task identifier. This excludes 3,000 self-pairs and 4,880 cross-split pairs, leaving 21,860 training pairs and 260 validation pairs. All 2,700 training recordings appear in both training roles. Validation retains all 300 recordings in the split; only recordings with a non-self partner contribute a validation pair.

Both Demo and Query are Franka executions. Only camera 0, the external view, enters the model. Raw recordings may also contain a wrist image; it is not selected by this recipe. Demo VAE inputs are 224×224 and Query VAE inputs are 224×288. DINOv2 inputs are separately resized to 168×168.

## Prepare and cache

Run from the repository root. Set paths to the supplied assets:

```bash
DATA=/absolute/path/to/verified-recording-release
WORK=/absolute/path/to/new/main-cache
MODEL=/absolute/path/to/zero-wam-pretrain
DINO=/absolute/path/to/dinov2-base
EMPTY_TEXT=/absolute/path/to/empty_text_emb.pt
RUN=/absolute/path/to/new/main-run

python scripts/robokata.py prepare plan --root "$DATA" --work "$WORK"
python scripts/robokata.py check-data --work "$WORK" --workers 8 --output "$WORK/data-check.json"
python scripts/robokata.py prepare stats --work "$WORK"
python scripts/robokata.py prepare cache --work "$WORK" --model "$MODEL"
python scripts/robokata.py prepare verify --work "$WORK" --model "$MODEL"
python scripts/robokata.py demo-features --work "$WORK" --encoder "$DINO" --output "$WORK/demo-dino.pt" --device cuda:0 --bf16
python scripts/robokata.py semantic-features --work "$WORK" --encoder "$DINO" --output "$WORK/query-dino.pt" --device cuda:0 --bf16
python scripts/robokata.py semantic-features --work "$WORK" --encoder "$DINO" --output "$WORK/query-dino.pt" --verify
```

Set WORK to a training cache directory. The preparation plan records the split, single-view input sizes and encoder identity. Normalization is fitted on the 2,700 unique training Query recordings only, not on repeated pairs or validation recordings. The Demo and Query semantic caches must use the same DINOv2 weights.

VAE caching accepts `--shards N --shard I`; DINO feature caching also supports sharded extraction followed by merging. Use the original entry-point help for these optional cache settings.

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/robokata.py --gpus 4 train \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY_TEXT" \
  --run-dir "$RUN" --demo-features "$WORK/demo-dino.pt" \
  --semantic-features "$WORK/query-dino.pt"
```

The launcher reads `configs/train.json` and enables FI. Training starts from the Zero-WAM pretrained foundation. Demo attention gates and indexing strengths start at zero. The shared temperature starts at one. SF and key-loss weights are both 0.5; Demo-token dropout is 0.3. FI training uses clean video conditioning.

| Parameter | Main recipe |
|---|---|
| Epochs | 20 |
| GPUs × per-GPU batch | 4 × 16 |
| Global batch / gradient accumulation | 64 / 1 |
| AdamW moments / epsilon | (0.9, 0.95) / 1e-8 |
| Weight decay / gradient clipping | 0.01 / global norm 1.0 |
| Learning rate | 1e-5 to 1e-6 |
| Warmup / cosine decay | 100 / 6,500 optimizer updates |
| LoRA rank / alpha | 106 / 212 |
| Precision / sharding | BF16 / FSDP2 |
| Activation recomputation | Enabled on backbone blocks |
| Checkpoint selection | Final epoch |

Exact-length batching uses 115 Query/Demo length buckets. Each epoch forms 1,323 full local batches of 16, then keeps 1,320 batches for four-rank synchronization. This gives 330 optimizer updates and 21,120 pair presentations per epoch; 20 epochs give 6,600 updates and 422,400 presentations. A pair is used at most once per epoch. Of the 740 omitted pairs each epoch, 180 belong to buckets smaller than 16; the other omissions depend on the epoch shuffle. Changing the GPU count or local batch can change this schedule.

Every completed epoch has a checkpoint; `latest.pt` supports recovery. To resume, repeat the same command with `--resume-from "$RUN/checkpoints/latest.pt"`. Keep the same data, caches, world size, batch and optimizer schedule. CLI overrides are available for explicit experiments; the table above is the main-model configuration.

To inspect the resolved training command without launching it:

```bash
python scripts/robokata.py --print-command train --work "$WORK" --model "$MODEL" --empty-text "$EMPTY_TEXT" --run-dir "$RUN" --demo-features "$WORK/demo-dino.pt" --semantic-features "$WORK/query-dino.pt"
```
