"""Recording, streaming VAE and dataset adapters for RoboKata."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from examples.baselines.zero_wam.zero_wam.dataset import IGDataset

ACTION_CHANNELS = (*range(9), 28)
ACTION_INTERVAL = 16


def recording(root, episode, *, views=1, record=None):
    from prepare import recording_path

    path = recording_path(root, episode, record)
    if path.suffix != ".npz":
        raise ValueError("Supervised query actions require a Franka NPZ recording")
    with np.load(path, allow_pickle=False) as data:
        rgb = data["rgb"]
        actions = data["action"]
        states = data["state"]
        time = data["timestamp_s"]
        step = data["observation_step"]
    if rgb.dtype != np.uint8 or rgb.shape != (len(time), 2, 224, 224, 3):
        raise ValueError(f"Unexpected RGB schema: {path}")
    if actions.shape != (len(time), 10) or states.shape != actions.shape:
        raise ValueError(f"Unexpected action/state schema: {path}")
    if not np.all(np.diff(step) == 5) or not np.allclose(np.diff(time), 0.02, atol=1e-5):
        raise ValueError(f"Invalid control cadence: {path}")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError(f"Non-finite robot data: {path}")
    pixels = rgb[:, 0] if views == 1 else np.concatenate((rgb[:, 0], rgb[:, 1]), axis=2)
    return pixels, torch.from_numpy(actions).float(), torch.from_numpy(states).float()


@torch.inference_mode()
def encode(wrapper, frames, ids, resize, *, chunk_frames=16):
    vae = wrapper.vae
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    wrapper.clear_cache()
    chunks = [ids[:1], *[ids[i:i + chunk_frames] for i in range(1, len(ids), chunk_frames)]]
    latents = []
    for chunk in chunks:
        video = torch.from_numpy(frames[chunk]).permute(0, 3, 1, 2).float()
        video = F.interpolate(video, size=resize, mode="bilinear", align_corners=False)
        video = (video / 127.5 - 1).permute(1, 0, 2, 3).unsqueeze(0)
        result = wrapper.encode_chunk(video.to(device=device, dtype=dtype))
        mean_latent = result.chunk(2, dim=1)[0].float()
        latents.append(((mean_latent - mean) / std)[0].to(dtype).cpu())
    latent = torch.cat(latents, dim=1).contiguous()
    if not torch.isfinite(latent).all():
        raise ValueError("VAE produced a non-finite latent")
    return latent


class RoboICLDataset(IGDataset):
    """Reuse IG causal grouping with explicitly masked RoboICL action channels."""

    def __init__(self, manifest_path, empty_text_path, seed, action_per_frame,
                 normalization_path, sampling_mode, action_history_mode, **_):
        from examples.baselines.zero_wam.zero_wam.action_space import EpisodeActionSpace

        self.manifest_path = Path(manifest_path)
        self.manifest = manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.sampling_mode = sampling_mode
        self.action_history_mode = action_history_mode
        self.records = manifest["sim_episodes"]
        self.humans = manifest["human_episodes"]
        self.repo_indices = {self.records[0]["repo_id"]: 0}
        self.humans_by_repo = {self.humans[0]["repo_id"]: [0]}
        self.action_channels = list(manifest["action_channels"])
        self.action_space = EpisodeActionSpace(json.loads(Path(normalization_path).read_text(encoding="utf-8")))
        self.action_per_frame = action_per_frame
        self.seed = seed
        self.payloads = OrderedDict()
        self.empty_text = torch.load(empty_text_path, map_location="cpu", weights_only=True)

    def episode_item(self, key):
        result = super().episode_item(key)
        # The existing observed-state projector accepts 18 inputs. The final
        # eight inputs are reserved, while the supervised action space stays 10D.
        result["ig_states"] = F.pad(result["ig_states"], (0, 8))
        return result
