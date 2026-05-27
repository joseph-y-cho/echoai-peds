# EchoAI-Peds

This repository contains the official implementation of **EchoAI-Peds**, a multi-task deep learning model for pediatric echocardiography analysis.

## Overview

**EchoAI-Peds** is a video-based vision transformer trained to simultaneously detect 28 congenital heart defects, structural and functional abnormalities, repairs, and interventions directly from complete pediatric echocardiography studies with multiple videos.

## Preprint

Our work is available as a preprint on medRxiv:

🔗 https://www.medrxiv.org/content/10.1101/2025.10.27.25338912v1

## Model Weights

The pretrained EchoAI-Peds model weights are publicly available on Hugging Face:

🔗 https://huggingface.co/hiesingerlab/echoai-peds

Download `model.pth` from the repository and place it in the project root directory (or update the checkpoint path in the inference script accordingly).

## Inference

`echoai-peds-inference.py` provides a self-contained pipeline for running EchoAI-Peds on new echocardiography studies.

### Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- h5py
- numpy

Install dependencies:

```bash
pip install torch torchvision h5py numpy
```

### Input Format

Studies should be stored as HDF5 files where each dataset represents a single echocardiographic view with shape `(C, T, H, W)` (3 channels, variable frames, 224×224 spatial resolution). Multiple views per study are supported.

### Usage

Update the checkpoint and study paths in the script, then run:

```bash
python echoai-peds-inference.py
```

The model returns a probability vector of length 28, one per condition. The full list of target labels is available in `LABEL_NAMES` within the script.

### Inference Details

- **Clip extraction**: 16 frames sampled with temporal stride 2 (31-frame window), up to 4 clips per view
- **Aggregation**: Clip probabilities are averaged per view, then view probabilities are averaged to produce a study-level prediction

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).

See the LICENSE file for details.
