# 🧠 RoboKata Model Structure

RoboKata combines demonstration reading, future semantic prediction and
foresight-guided indexing within a video-action model.

## 1. Read the demonstration

[Demo attention](../roboicl/demo_attention.py) attaches demonstration attention
to the transformer. [Demo features](../roboicl/demo_features.py) extracts frozen
DINOv2 features. The reference recipe uses external/head-camera demo input.

## 2. Predict future semantics

[Semantic target extraction](../roboicl/semantic_features.py) prepares query
feature targets. [Semantic prediction heads](../backend/examples/baselines/zero_wam/zero_wam/wan_va/semantic_ifp.py)
provide future-feature supervision. Query input uses external and left-wrist views.

## 3. Index the demonstration

[Foresight Indexing](../backend/examples/baselines/zero_wam/zero_wam/wan_va/foresight_indexing.py)
guides demonstration attention using predicted features. The
[transformer integration](../backend/examples/baselines/zero_wam/zero_wam/wan_va/modules/icl_model.py)
connects these components to the backbone.

## 🚀 Training and inference

| Entry point | Purpose |
| :--- | :--- |
| [Unified launcher](../scripts/robokata.py) | Preparation, training and inference commands |
| [Training pipeline](../roboicl/train.py) | Model adaptation and training configuration |
| [Distributed entry point](../roboicl/train_entry.py) | Multi-GPU training |
| [Offline inference](../roboicl/infer_checkpoint.py) | Checkpoint reconstruction and action prediction |
| [Action policy](../backend/examples/baselines/zero_wam/zero_wam/policy.py) | Underlying inference and cache handling |

The supplied launcher enables A+B1+FI. Shared core support for other variants
is retained but is not the focus of this supplement. Backbone provenance and
third-party notices are documented in [THIRD_PARTY.md](THIRD_PARTY.md).
