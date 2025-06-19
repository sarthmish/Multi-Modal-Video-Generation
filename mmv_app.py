from __future__ import annotations

import os
import random
import tempfile

import streamlit as st
import os
import random
import tempfile
import imageio
import numpy as np
import spaces
import torch
from src.utils.model import pipe

DESCRIPTION = "# Text-to-video generation"

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU , This demo does not work on CPU.</p>"

MAX_NUM_FRAMES = int(os.getenv("MAX_NUM_FRAMES", "48"))
DEFAULT_NUM_FRAMES = min(MAX_NUM_FRAMES, int(os.getenv("DEFAULT_NUM_FRAMES", "24")))
MAX_SEED = np.iinfo(np.int32).max

if torch.cuda.is_available():
    pipe.enable_vae_slicing()

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

@spaces.GPU
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


st.markdown(DESCRIPTION)

with st.expander("Advanced options"):
    seed = st.slider(
        label="Seed",
        min_value=0,
        max_value=MAX_SEED,
        step=1,
        value=0,
    )
    randomize_seed = st.checkbox("Randomize seed", value=True)
    num_frames = st.slider(
        label="Number of frames",
        min_value=24,
        max_value=MAX_NUM_FRAMES,
        step=1,
        value=24,
        help="Note that the content of the video also changes when you change the number of frames.",
    )
    num_inference_steps = st.slider(
        label="Number of inference steps",
        min_value=10,
        max_value=50,
        step=1,
        value=25,
    )

prompt = st.text_input(
    label="Prompt",
    max_chars=100,
    help="Enter your prompt",
)

run_button = st.button("Generate video")

if prompt and run_button:
    seed = randomize_seed_fn(seed, randomize_seed)
    video_file = generate(prompt, seed, num_frames, num_inference_steps)
    st.video(video_file)