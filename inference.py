from __future__ import annotations

import random
import tempfile
import imageio
import numpy as np
import torch
from src.utils.model import pipe
import yaml
config_path = 'Your Path'
with open(config_path, "r") as stream:
    params = yaml.safe_load(stream)

MAX_NUM_FRAMES = 48
DEFAULT_NUM_FRAMES = min(MAX_NUM_FRAMES, 24)
MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def to_video(frames: np.ndarray, fps: int) -> str:
    frames = np.clip((frames * 255), 0, 255).astype(np.uint8)
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = imageio.get_writer(out_file.name, format="FFMPEG", fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return out_file.name

def generate(
    prompt: str,
    seed: int,
    num_frames: int,
    num_inference_steps: int,
) -> str:
    generator = torch.Generator().manual_seed(seed)
    frames = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        width=576,
        height=320,
        generator=generator,
    ).frames[0]
    return to_video(frames, 8)

prompt = params['prompt']
seed = params['seed']
randomize_seed = params['randomize_seed']
num_frames = params['num_frames']
num_inference_steps = params['num_inference_steps']

seed = randomize_seed_fn(seed, randomize_seed)
video_file = generate(prompt, seed, num_frames, num_inference_steps)

print(video_file)