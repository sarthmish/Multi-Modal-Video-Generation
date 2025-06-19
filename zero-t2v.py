# %%
from __future__ import annotations

import os
import random
import tempfile

# %%
!pip install spaces 
!pip install diffusers transformers accelerate torch 

# %%
pip install imageio[ffmpeg]

# %%
import imageio
import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# %%
MAX_NUM_FRAMES = 48
DEFAULT_NUM_FRAMES = min(MAX_NUM_FRAMES, 24)
MAX_SEED = np.iinfo(np.int32).max

# %%
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# %%
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

# %%
def to_video(frames: np.ndarray, fps: int) -> str:
    frames = np.clip((frames * 255), 0, 255).astype(np.uint8)
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    writer = imageio.get_writer(out_file.name, format="FFMPEG", fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return out_file.name

# %%
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

# %%
prompt = "A lion walking in the jungle"
seed = 0
randomize_seed = True
num_frames = 24
num_inference_steps = 25

seed = randomize_seed_fn(seed, randomize_seed)
video_file = generate(prompt, seed, num_frames, num_inference_steps)

print(video_file)

# %%
import shutil
shutil.copy(video_file, '/kaggle/working/')


