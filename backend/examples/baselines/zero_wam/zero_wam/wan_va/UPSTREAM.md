# Upstream provenance

Source: https://github.com/robbyant-research/Zero-WAM

Reference commit: `08e2c4ae41e2b63573a299825cebe6753481407c`.

This is a required-source subset, not a second complete upstream checkout. It contains no model assets, latent caches or dataset files. The Apache-2.0 license and existing source notices are retained.

Local changes relative to the released training path:

- `train.py`: retain video/action noise and flow-matching objectives, add semantic-foresight and key losses, and delegate data, adapters, logging and full-state checkpoints to the sibling `training` module. Remove unused dataset launchers and WandB startup. Activation checkpointing follows the training configuration.
- `distributed/fsdp.py`: support an optional host-local sharding mesh, replicated between hosts. Base compute uses BF16; gradient reductions use FP32.
- `modules/icl_model.py`: support padded per-example masks, action-to-Demo reads, semantic targets and cached frame indices. Training repeats the frame-index sequence for noisy and clean action streams. The native chunk-level temporal predicates are retained.
- `__init__.py`: avoid importing unused upstream launch configurations.
- `modules/model.py`: make FlashAttention an optional import; require it only when the `flashattn` backend is selected. The flex-attention path is unchanged.
- `utils/__init__.py`: expose the utilities used by this baseline without importing the unused websocket inference server.

The video/action backbone is reused from Zero-WAM. Action-channel packing and observed-state conditioning live outside the vendored tree in `model.py`; LoRA scope/rank selection lives in `lora.py`. RoboKata adapters and data preparation are in `roboicl/`.

Training samples chunks of one to four latent frames; inference uses chunks of two after the initial frame. The main RoboKata recipe replaces latent MCP supervision with semantic-foresight regression. See `docs/MODEL.md` and `docs/TRAINING.md` at the repository root for the model and training settings.
