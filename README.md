<div align="center">

# 🤖 RoboKata

### Watching the Demo, or Ignoring It?
### Demonstration-Faithful In-Context Imitation for World Action Models

**Anonymous Code**

</div>

---

## ✨ Overview

RoboKata learns to follow a demonstration through three complementary components:

| Component | Role |
| :--- | :--- |
| **Demo Cross-Attention** | Read visual features from the demonstration. |
| **Semantic Foresight** | Predict future semantic features. |
| **Foresight Indexing** | Use predicted features to guide attention to the demonstration. |

This supplement includes the **model architecture, training pipeline and offline recorded-pair inference** for the main A+B1+FI model. See the [model guide](docs/MODEL.md) for implementation entry points.

## 📁 Repository Structure

```text
.
├── backend/                  # Transformer backbone and action prediction
├── roboicl/                  # Data preparation, training and inference
├── configs/
│   └── train.json            # Reference training hyperparameters
├── scripts/
│   ├── robokata.py           # Unified command launcher
│   └── verify_release.py     # Dependency-free packaging checks
├── requirements/
│   └── environment.lock.txt  # Dependency snapshot
└── docs/                     # Model, training and inference guides
```

## 🛠️ Environment

| Requirement | Reference setup |
| :--- | :--- |
| Operating system | Linux |
| Python | 3.10 |
| PyTorch | 2.9.0 |
| CUDA | 12.6 |
| Training | 4 GPUs · batch 16 per GPU · BF16 |

The [dependency snapshot](requirements/environment.lock.txt) records the original environment; a clean installation has not been validated for this supplement. Minimum GPU memory has not been established.

Model weights, recordings, DINOv2 and the empty-text embedding must be supplied separately. No weights or sample recordings are bundled.

## 🚀 Training

The [training guide](docs/TRAINING.md) covers data preparation, feature caching, main-model training and checkpoint resume.

1. Prepare the recording manifest and normalization statistics.
2. Extract and verify VAE, demonstration and query feature caches.
3. Launch A+B1+FI training using the [reference recipe](configs/train.json).

The unified launcher sets the backend import path automatically. Run commands from the repository root. To inspect the launcher or preview the training command without loading GPU dependencies:

```bash
python scripts/robokata.py --help
python scripts/robokata.py --print-command train --help
```

Complete training commands and required paths are provided in the guide.

## 🎯 Offline Inference

The [inference guide](docs/INFERENCE.md) supports a local training checkpoint or a compatible Safetensors adapter.

**Input:** a held-out recorded pair, compatible model weights, normalization and verified caches.  
**Output:** the first **16 × 10 action targets**, with pair and checkpoint metadata, saved as JSON. The recorded control rate is **50 Hz**.

This entry point performs offline action prediction. Closed-loop evaluation and robot deployment are outside the scope of this supplement. Cache and checkpoint compatibility checks remain enabled.

## ✅ Code Checks

Run the packaging checks without installing model dependencies:

```bash
python scripts/verify_release.py
```

These checks validate the core-source integrity inventory, Python syntax and launcher argument handling. They do **not** execute the model. GPU training and inference have not been tested in this local packaging environment.

## 📄 Third-Party Notices

Third-party copyright and license notices are retained in their original files. See [third-party notices](docs/THIRD_PARTY.md) for dependency attribution.
