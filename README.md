# RoboKata

**Watching the Demo, or Ignoring It? Demonstration-Faithful In-Context Imitation for World Action Models**

RoboKata predicts a sequence of Franka action targets from a demonstration video, a current camera image, and the robot state.

## 🛠️ Environment

Run the commands below from the repository root on Linux with Python 3.12 and an NVIDIA GPU. The dependency lock uses PyTorch 2.9.0 with CUDA 12.6.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements/environment.lock.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The CUDA check should print `True`. Supply the following assets separately and set their paths for both training and inference.

```bash
MODEL=/path/to/zero-wam-pretrain       # contains transformer/ and vae/
DINO=/path/to/dinov2-base              # ViT-B/14 without register tokens
EMPTY_TEXT=/path/to/empty_text_emb.pt
```

## 🚀 Training

Set `DATA` to the verified Franka recording release. Use new directories for the feature cache and training run. Run the preparation commands once before launching training.

```bash
DATA=/path/to/verified-recording-release
WORK=/path/to/main-cache
RUN=/path/to/main-run

python scripts/robokata.py prepare plan --root "$DATA" --work "$WORK"
python scripts/robokata.py check-data --work "$WORK" --workers 8 --output "$WORK/data-check.json"
python scripts/robokata.py prepare stats --work "$WORK"
python scripts/robokata.py prepare cache --work "$WORK" --model "$MODEL"
python scripts/robokata.py prepare verify --work "$WORK" --model "$MODEL"
python scripts/robokata.py demo-features --work "$WORK" --encoder "$DINO" --output "$WORK/demo-dino.pt" --device cuda:0 --bf16
python scripts/robokata.py semantic-features --work "$WORK" --encoder "$DINO" --output "$WORK/query-dino.pt" --device cuda:0 --bf16
```

The default recipe trains for 20 epochs on four GPUs and writes checkpoints to `$RUN/checkpoints/`. See the [training guide](docs/TRAINING.md) for the dataset layout and resume command.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/robokata.py --gpus 4 train \
  --work "$WORK" --model "$MODEL" --empty-text "$EMPTY_TEXT" \
  --demo-features "$WORK/demo-dino.pt" --semantic-features "$WORK/query-dino.pt" \
  --run-dir "$RUN"
```

## 🎯 Inference

Supply a demonstration video, a current external-camera image, and the robot state measured at the time of that image. The state JSON holds ten values in training order (tool XYZ in metres, 6D rotation, and normalized gripper position). The example uses the latest training checkpoint. For a released adapter, pass `adapter.safetensors` with `adapter_config.json` alongside it.

```bash
python scripts/robokata.py infer \
  --model "$MODEL" --dino "$DINO" --empty-text "$EMPTY_TEXT" \
  --checkpoint "$RUN/checkpoints/latest.pt" \
  --demo /path/to/demo.mp4 --query-image /path/to/query.png \
  --state /path/to/state.json --output outputs/prediction.json
```

The output JSON contains 16 absolute action targets for 50 Hz control. See the [inference guide](docs/INFERENCE.md) for a state JSON example and the output format.
