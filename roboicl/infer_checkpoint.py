"""Predict an action chunk from RGB demonstration and current observation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from examples.baselines.zero_wam.zero_wam import policy
from examples.baselines.zero_wam.zero_wam.action_space import EpisodeActionSpace
from examples.baselines.zero_wam.zero_wam.config import ZeroWAMConfig

from prepare import ACTION_CHANNELS, ACTION_REPRESENTATION, ENCODING, read_json


class RoboICLPolicy(policy.ZeroWAMAgent):
    def clear_cache(self):
        super().clear_cache()
        self.demo_features = None

    def _state(self, obs):
        return F.pad(self.roboicl_space.normalize(obs["state"].to(self.device), "state"), (0, 8))

    @torch.no_grad()
    def accept_executed(self, commands, frames):
        """Commit raw executed world-hand10 targets and every fourth RGB frame."""
        count = 16 if self.chunk_idx == 0 else 32
        commands = torch.as_tensor(commands, device=self.device, dtype=torch.float32).clone()
        frames = torch.as_tensor(frames, device=self.device)
        if commands.shape != (count, 10) or frames.shape != (count // 4, 224, 224, 3):
            raise ValueError("Expected one executed chunk and its 50/4 Hz external RGB feedback")
        if frames.dtype != torch.uint8 or not torch.isfinite(commands).all():
            raise ValueError("Executed commands must be finite and RGB feedback must be uint8")
        if self.executed_actions or self.observed_frames:
            raise RuntimeError("Previous executed chunk has not been consumed")
        commands[:, 9].clamp_(-1, 1)
        low, high = self.roboicl_space.bounds(commands, "action")
        normalized = 2 * (commands - low) / (high - low + 1e-6) - 1
        self.executed_actions.extend(value[None].clone() for value in normalized)
        rgb = frames.float().div_(255)
        self.observed_frames.extend(value[None, None] for value in rgb)
        self.total_executed_steps += count

    def _stream(self, data, timestep, action, cache_type, frame_id, icl=False, frame_shift=None):
        inputs = super()._stream(data, timestep, action, cache_type, frame_id, icl, frame_shift)
        indexed_video = not action and not icl and hasattr(self.transformer, "foresight_indexing")
        if self.demo_features is not None and (action or indexed_video):
            inputs["demo_features"] = self.demo_features
            if indexed_video:
                inputs["demo_index_frames"] = data.shape[2]
        return inputs


def normalization(statistics):
    if not isinstance(statistics, dict) or statistics.get("representation") != ACTION_REPRESENTATION:
        raise ValueError("Supply the checkpoint's absolute world-frame normalization statistics")
    result = {"representation": ACTION_REPRESENTATION, "clip": float(statistics["clip"])}
    if not np.isfinite(result["clip"]) or result["clip"] <= 0:
        raise ValueError("Normalization clip must be positive and finite")
    for name in ("state", "action"):
        lo = np.asarray(statistics[name]["q01"], dtype=np.float64)
        hi = np.asarray(statistics[name]["q99"], dtype=np.float64)
        if lo.shape != (10,) or hi.shape != (10,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError(f"Expected ten finite ordered quantile bounds for {name}")
        result[name] = {"q01": lo.tolist(), "q99": hi.tolist()}
    return result


def load_checkpoint(path, adapter_config=None, stats_path=None):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        config = read_json(adapter_config or path.with_name("adapter_config.json"))
        if config.get("action_representation", ACTION_REPRESENTATION) != ACTION_REPRESENTATION or config.get("actions_per_latent", 16) != 16:
            raise ValueError("Expected world-frame actions with sixteen actions per latent")
        settings = {
            "ig_roboicl_channels": config["action_channels"],
            "ig_demo_attention": config["demo_attention"],
            "ig_foresight_indexing": config.get("foresight_indexing"),
            "ig_lora_rank": config["lora_rank"],
            "snr_shift": config.get("video_snr_shift", 5.0),
            "action_snr_shift": config.get("action_snr_shift", 1.0),
            "icl_rope_h": config.get("icl_rope_h", 24),
            "ig_roboicl_encoding": config.get("video_encoding", ENCODING),
        }
        embedded_stats = config.get("normalization")
        adapter = load_file(path, device="cpu")
    else:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        settings = checkpoint["native_config"]
        embedded_stats = settings.get("ig_normalization")
        adapter = checkpoint["adapter_state_dict"]
    from demo_attention import SPEC
    from examples.baselines.zero_wam.zero_wam.wan_va.foresight_indexing import SCHEMA

    if settings.get("ig_roboicl_channels") != list(ACTION_CHANNELS):
        raise ValueError("Expected the ten RoboKata action channels")
    if settings.get("ig_roboicl_encoding", ENCODING) != ENCODING:
        raise ValueError("Expected a single external-camera video input")
    if settings.get("ig_demo_attention") != SPEC:
        raise ValueError("Expected the single-view DINO demonstration attention configuration")
    if (settings.get("ig_foresight_indexing") or {}).get("schema") != SCHEMA:
        raise ValueError("Expected a RoboKata checkpoint with foresight indexing")
    if not isinstance(settings.get("ig_lora_rank"), int) or settings["ig_lora_rank"] < 1:
        raise ValueError("Checkpoint must specify a positive LoRA rank")
    statistics = normalization(read_json(stats_path) if stats_path else embedded_stats)
    if embedded_stats is not None and statistics != normalization(embedded_stats):
        raise ValueError("Supplied normalization differs from the checkpoint configuration")
    return settings, adapter, statistics


def read_demo(path, fps=None):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            frames = data["rgb"].copy()
            if frames.ndim == 5:
                frames = frames[:, 0]
            if fps is None and "timestamp_s" in data and len(data["timestamp_s"]) > 1:
                dt = np.diff(data["timestamp_s"])
                if not np.isfinite(dt).all() or (dt <= 0).any() or not np.allclose(dt, np.median(dt), atol=1e-5):
                    raise ValueError("Demo timestamps must use a constant positive frame interval")
                fps = 1.0 / float(np.median(dt))
                if np.isclose(fps, round(fps), rtol=0, atol=1e-6):
                    fps = float(round(fps))
        fps = 50.0 if fps is None else fps
    else:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = fps if fps is not None else (float(stream.average_rate) if stream.average_rate else None)
            decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        if not decoded:
            raise ValueError("The demonstration video contains no frames")
        frames = np.stack(decoded)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or len(frames) == 0:
        raise ValueError("Demo RGB must be uint8 [frames, height, width, 3]")
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Supply --demo-fps when the frame rate is not recorded")
    return frames, float(fps)


def read_observation(image_path, state_path):
    from PIL import Image

    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    state = read_json(state_path)
    if isinstance(state, dict):
        state = state["state"]
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError("State JSON must contain ten finite world-frame values")
    return {"rgb": torch.from_numpy(pixels)[None, None].float().div_(255),
            "state": torch.from_numpy(state)[None]}


def build_actor(settings, adapter, statistics, args):
    def pack(actions, _offset):
        if actions.shape[-1] != 10:
            raise ValueError("Expected ten action channels")
        packed = actions.new_zeros(*actions.shape[:-1], 30)
        packed[..., list(ACTION_CHANNELS)] = actions
        return packed

    policy.pack_actions = pack
    policy.unpack_actions = lambda value, _offset: value[..., list(ACTION_CHANNELS)]
    actor = RoboICLPolicy(ZeroWAMConfig(
        model_path=str(args.model), empty_text_embedding=str(args.empty_text),
        cache_dir=str(args.cache_dir), action_dim=10, adapter_rank=settings["ig_lora_rank"],
        image_size=224, robot_width=288, human_height=224, human_width=224,
        video_snr_shift=settings.get("snr_shift", 5.0),
        action_snr_shift=settings.get("action_snr_shift", 1.0),
        icl_rope_h=settings.get("icl_rope_h", 24),
        video_inference_steps=args.denoise_steps, action_inference_steps=args.denoise_steps,
        device=args.device,
    ))
    actor.action_channel_ids = ACTION_CHANNELS
    actor.roboicl_space = EpisodeActionSpace(statistics)
    from demo_attention import attach_demo_attention, attach_foresight_indexing

    spec = settings["ig_demo_attention"]
    attach_demo_attention(actor.transformer, spec)
    attach_foresight_indexing(actor.transformer, settings["ig_foresight_indexing"]["schema"], query_views=1)
    actor.transformer.foresight_indexing.to(device=actor.device, dtype=actor.dtype)
    for index in spec["layers"]:
        actor.transformer.blocks[index].demo_attention.to(device=actor.device, dtype=actor.dtype)
    expected = {name: param for name, param in actor.transformer.named_parameters() if param.requires_grad}
    missing = set(expected) - set(adapter)
    extras = set(adapter) - set(expected)
    if missing or any(not name.startswith(("semantic_ifp.", "mcp_blocks.", "mcp_mlp_hidden.", "mcp_projections.")) for name in extras):
        raise ValueError(f"Incompatible adapter parameters; missing={sorted(missing)}, extra={sorted(extras)}")
    for name, param in expected.items():
        if adapter[name].shape != param.shape:
            raise ValueError(f"Adapter shape differs for {name}: {adapter[name].shape} vs {param.shape}")
    actor.load_adapter({name: adapter[name] for name in expected})
    return actor.eval()


@torch.inference_mode()
def prepare_demo(actor, frames, fps, dino_path):
    from demo_features import FRAMES, encode as encode_dino, load_encoder
    from data_adapter import encode as encode_vae
    from examples.baselines.zero_wam.zero_wam.cache import sample_plan

    ids = ((np.arange(FRAMES) + 0.5) * len(frames) / FRAMES).astype(int)
    encoder = load_encoder(dino_path, actor.device)
    with torch.autocast(actor.device.type, dtype=torch.bfloat16, enabled=actor.device.type == "cuda"):
        features = encode_dino(encoder, torch.from_numpy(frames[ids]))
    del encoder
    plan = sample_plan(len(frames), fps, "human", {"video": {"human": {"target_fps": 12.0}}})
    latent = encode_vae(actor.streaming_vae, frames, plan["sampled_frame_indices"], (224, 224))
    if features.shape != (1, 32, 780) or not torch.isfinite(features).all():
        raise ValueError("Unexpected DINO demonstration features")
    if latent.shape != (48, len(plan["latent_anchor_indices"]), 14, 14) or not torch.isfinite(latent).all():
        raise ValueError("Unexpected demonstration video latents")
    actor.preview_human_latent = latent[None]
    return features.to(device=actor.device, dtype=actor.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "empty-text", "checkpoint", "dino", "demo", "query-image", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, help="Defaults to adapter_config.json beside Safetensors")
    parser.add_argument("--stats", type=Path, help="State/action statistics if not embedded in the checkpoint configuration")
    parser.add_argument("--demo-fps", type=float, help="Override the demonstration frame rate")
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/inference-cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--denoise-steps", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists")
    if args.denoise_steps < 1:
        parser.error("--denoise-steps must be positive")
    settings, adapter, stats = load_checkpoint(args.checkpoint, args.adapter_config, args.stats)
    frames, fps = read_demo(args.demo, args.demo_fps)
    obs = read_observation(args.query_image, args.state)
    actor = build_actor(settings, adapter, stats, args)
    obs = {name: value.to(actor.device) for name, value in obs.items()}
    with torch.inference_mode():
        demo = prepare_demo(actor, frames, fps, args.dino)
        actor.prepare_for_eval(torch.empty(1), obs)
        actor.demo_features = demo
        torch.manual_seed(args.seed)
        action = actor.roboicl_space.denormalize(actor.get_action(obs), "action")
    if action.shape != (1, 16, 10) or not torch.isfinite(action).all():
        raise RuntimeError(f"Invalid predicted action chunk: {action.shape}")
    result = {"seed": args.seed, "control_hz": 50, "denoise_steps": args.denoise_steps,
              "action_representation": ACTION_REPRESENTATION, "actions": action[0].float().cpu().tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {action.shape[1]} action targets to {args.output}")


if __name__ == "__main__":
    main()
