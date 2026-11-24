# Model implementation

RoboKata uses the 30-layer Wan-2.2-TI2V-5B video-action backbone from Zero-WAM. Layers are indexed from zero. The main model combines DCA, SF and FI.

## Inputs

- Demo: external-camera RGB, 224×224 for the video autoencoder.
- Query: external-camera RGB, 224×288 for the video autoencoder.
- Semantic encoder: frozen DINOv2 ViT-B/14 without register tokens, 168×168.
- Demo tokens: midpoint of each of eight temporal segments, 2×2 pooled spatial cells per frame. The 768 visual dimensions plus eight frame-code and four cell-code dimensions give 32 tokens of width 780.

Demo actions, states, task identifiers and waypoint annotations are not policy inputs. Task identifiers are used only to construct the supervised pair plan.

## Demo Cross-Attention

`roboicl/demo_features.py` builds the cached Demo tokens. `roboicl/demo_attention.py` attaches a gated attention block to action layers 7, 15 and 23. Each block uses four heads, projection width 256, bias-free projections, query layer normalization and a tanh residual gate initialized at zero. Training drops all Demo tokens of a sample together with probability 0.3, including their position codes.

## Semantic Foresight

`roboicl/semantic_features.py` prepares frozen Query semantic targets. `backend/examples/baselines/zero_wam/zero_wam/wan_va/semantic_ifp.py` implements the prediction heads. States from layers 3, 11, 19 and 29 are fused, followed by a shared 256-dimensional bottleneck with GELU and four 768-dimensional output heads. Future offsets are 1, 3, 5 and 7 chunks; cosine losses use weights 0.50, 0.25, 0.15 and 0.10, with overall weight 0.5. Invalid temporal targets are masked. SF replaces the original latent-MCP modules and is used for training supervision.

## Foresight Indexing

`backend/examples/baselines/zero_wam/zero_wam/wan_va/foresight_indexing.py` supplies keys at layers 3, 11 and 19 for reads at 7, 15 and 23. Each key head has a 256-dimensional hidden layer, GELU and a 768-dimensional output. Matching uses the visual features, a shared bias-free 256-dimensional projection and cell-wise cosine similarity.

The key is detached before matching. One learned temperature is shared across read layers, initialized at one and implemented as `exp(clamp(log_temperature, -5, 5))`. Every read layer has an indexing strength initialized at zero. The frame-wise log-probability bias is added to action-to-Demo attention logits. During training, the frame-index sequence is repeated for the concatenated noisy and clean action streams. The key-prediction loss is weighted by 0.5; no frame-correspondence labels are supplied. Parameters are optimized with the joint video, action, semantic-foresight and key-prediction objective.

The final video cache-writing pass stores the frame indices; subsequent action denoising passes reuse them. FI adds no extra backbone pass. The learned indexing strengths are restored from checkpoints during inference.

## Training and execution

`roboicl/train.py` adapts the backbone, constructs the full-episode dataset and applies exact-length batching. The main configuration uses LoRA rank106/alpha212, state/action adaptation and DCA/SF/FI modules, with a frozen DINOv2 encoder. Backbone activations are recomputed and parameters are sharded with FSDP2.

The action representation is absolute world-frame tool XYZ, a 6D rotation representation and a normalized gripper command, for ten dimensions. One latent frame corresponds to 16 actions at 50 Hz. The offline inference entry point produces the first 16 actions; the policy supports later chunks of 32 actions. Video and action denoising use 50 steps, guidance 1 and timestep shifts 5 and 1.

Temporal forcing uses chunk-level causal masks. Training samples a chunk length of one to four latent frames; inference uses two after the initial frame. Actions can read the clean video of the current chunk and the permitted history. SF offsets are measured in units of the sampled chunk length.
