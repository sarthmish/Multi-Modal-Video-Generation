# MultiModal text-to-video generation

## Table of Contents

1. [Overview](#overview)
2. [Model Approach](#model-approach)
3. [Folder Structure](#folder-structure)
4. [Installation](#installation)
5. [References](#references)

## Overview

### Problem Understanding :
This problem involves the inferencing of multi-modal models for text-to-video generation. The goal of this project is to compare different MLLMs based on video generation from textual prompt and evaluate their performance based on the video quality generated.

### Project Scope :
•	This project will focus on comparing models based on video quality generated.

•	This project will explore different models for text-to-video generation and evaluate their performance respectively.

## Model Approach

Model comparison was done between multi-modal models for text-to-video generation.

Zeroscope is a free and open-source software that uses text-to-video technology to convert written descriptions into high-quality videos. It is an improved version of Modelscope, offering better resolution, no watermarks, and a closer aspect ratio to 16:9.

## Folder Structure

_Directory has been divided in the following structure._

```
Text-to-video generation
│   README.md
|
├───Notebooks
│       damo-t2v.ipynb
│       t2v-zero.ipynb
│
├───src
│   │
│   └───utils
│       │   config.yaml
│       │   inference.py
│       │   model.py
│
└───app.py
```

## Requirements
- Python
- PyTorch
- Diffusers
- huggingface-hub
- Other dependencies (specified in requirements.txt)

## Usage

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  1. Config.yaml file.

Configure the project settings by editing the `config.yaml`.

  ```yaml
  # inference args
  prompt : "A lion walking in the jungle"
  seed : 0
  randomize_seed : True  #If you want to generate different videos everytime for the same prompt, then keep this true, but if exactly same video is required to be generated everytime for the same prompt, then keep this false 
  num_frames : 24
  num_inference_steps : 25
```
_Set path to config.yaml file in inference.py_
```
config_path = 'path/to/config.yaml'
```

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  2. Inference.

_Run command to start inferencing._
```Q
python inference.py
```

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  3. Running the Streamlit App.

_Run command to start the streamlit app on local system._

```
streamlit run app.py
```

## References

- [Zeroscope_v2](https://huggingface.co/cerspense/zeroscope_v2_576w)

- [Damo-vilab-text-to-video-synthesis](https://huggingface.co/ali-vilab/modelscope-damo-text-to-video-synthesis)






