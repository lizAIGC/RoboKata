# Inference

The inference command predicts 16 absolute action targets from a demonstration video, a current external-camera image and the current robot state. It uses the same DINOv2 features, causal VAE sampling, normalization and FI attention as the model's streaming policy.

## Assets

- The matching Zero-WAM pretraining directory containing `transformer/` and `vae/`.
- DINOv2 ViT-B/14 weights without register tokens.
- The empty-text embedding, shape 512×4096.
- A RoboKata adapter: `adapter.safetensors` with `adapter_config.json`, or a trusted training `.pt` checkpoint.
- State/action normalization in the adapter configuration or checkpoint. If it is stored separately, supply `--stats stats.json`.

The adapter configuration specifies `action_channels`, `demo_attention`, `foresight_indexing`, `lora_rank`, normalization and inference shifts. Parameter names and shapes are checked when loading. Asset directories can be placed anywhere; no training plan, upload receipt or feature-cache hash is required.

## Command

```bash
python scripts/robokata.py infer \
  --model /path/to/zero-wam-pretrain \
  --empty-text /path/to/empty_text_emb.pt \
  --checkpoint /path/to/adapter.safetensors \
  --dino /path/to/dinov2-base \
  --demo /path/to/demo.mp4 \
  --query-image /path/to/query.png \
  --state /path/to/state.json \
  --output outputs/prediction.json
```

The default adapter configuration is `adapter_config.json` beside the Safetensors file. Use `--adapter-config` to specify another location. A training checkpoint reads its configuration and normalization directly from the saved checkpoint.

## Inputs

**Demo.** Supply a constant-frame-rate video or an NPZ with `rgb` as uint8 `[T,H,W,3]`. For `[T,V,H,W,3]` recordings, camera 0 is selected. Only RGB enters the model. Video frame rate comes from its metadata; NPZ uses `timestamp_s` when available, otherwise 50 Hz. `--demo-fps` overrides it. Use the original recording speed. The VAE samples at 12 Hz and resizes to 224×224; DINOv2 uses eight segment-midpoint frames at 168×168. Features are computed for the supplied demonstration.

**Query.** A PNG/JPEG external-camera observation, resized to 224×288 by the policy. It should match the time of the supplied robot state.

**State.** A JSON array of ten values, or an object with a `state` array. Values use the trained state representation: world-frame tool XYZ in metres, six rotation-representation components, and gripper position normalized to [-1,1]. Use the same tool frame and 6D rotation ordering as the training recordings.

```json
{"state": [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]}
```

This is a format example; supply the robot's measured state. Normalization uses the checkpoint's state quantiles before inference and its action quantiles after prediction.

## Output and execution

The JSON output contains `actions` with shape 16×10 and `control_hz: 50`. Actions are absolute world-frame tool XYZ, six rotation components and the gripper command. The command predicts one initial chunk; simulator execution and benchmark scoring are separate integrations. The underlying streaming policy supports subsequent 32-action chunks when supplied with executed-action and observation history.

For a streaming integration, call `actor.accept_executed(commands, frames)` after executing each chunk. Supply the raw absolute commands with shape 16×10 for the first chunk and 32×10 thereafter, plus external RGB observations every four executed actions as uint8 images of size 224×224. Then call `actor.get_action(obs)` with the current image and measured state. The adapter applies the trained action normalization to the executed history and the physical gripper bounds before committing it.

Video and action denoising default to 50 steps each. `--seed` selects denoising noise, and `--cache-dir` selects a local runtime cache directory. Training data and existing training caches are not needed for inference.
