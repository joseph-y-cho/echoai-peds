import torch
import torch.nn as nn
import torchvision
import numpy as np
import h5py
import random
from torchvision.transforms.v2 import Compose, Normalize, ToDtype


# ───────────────────────── Model ─────────────────────────────

def build_model(num_labels: int):
    model = torchvision.models.video.mvit_v2_s(weights=None)
    model.head[-1] = nn.Linear(model.head[-1].in_features, num_labels)
    return model


def load_echoai_peds_model(checkpoint_path: str, num_labels: int = 28, device: str = "cpu"):
    device = torch.device(device)
    model = build_model(num_labels=num_labels)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"Checkpoint not found at {checkpoint_path}. Returning uninitialized model.")
        model.to(device)
        return model

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ───────────────── Multi-View Multi-Clip Inference ───────────

CLIP_LEN = 16
STRIDE = 2
WINDOW_LEN = (CLIP_LEN - 1) * STRIDE + 1  # 31
MAX_CLIPS_PER_VIEW = 4

MEAN = (0.108, 0.109, 0.122)
STD = (0.175, 0.178, 0.195)
TRANSFORM = Compose([ToDtype(torch.float32, scale=True), Normalize(MEAN, STD)])


def extract_clips_from_h5(h5_path: str):
    """
    Read an HDF5 study file and return clips for every view.

    Each view in the HDF5 file is stored as a dataset with shape
    (C, T, H, W). Clips of 16 frames are extracted with a temporal
    stride of 2 (window length = 31 frames). Up to 4 non-overlapping
    clips are randomly sampled per view.

    Returns
    -------
    all_clips : list[torch.Tensor]
        Each element is (n_clips, C, 16, H, W) for one view.
    """
    all_clips = []
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            video = torch.tensor(f[key][:])  # (C, T_full, H, W)
            T = video.shape[1]
            view_clips = []

            if T < WINDOW_LEN:
                start = random.randint(0, max(0, T - CLIP_LEN))
                clip = video[:, start : start + CLIP_LEN]
                clip = TRANSFORM(clip.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
                view_clips.append(clip)
            else:
                starts = list(range(0, T - WINDOW_LEN + 1, WINDOW_LEN))
                random.shuffle(starts)
                for start in sorted(starts[: min(MAX_CLIPS_PER_VIEW, len(starts))]):
                    idxs = list(range(start, start + WINDOW_LEN, STRIDE))
                    clip = video[:, idxs]
                    clip = TRANSFORM(clip.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
                    view_clips.append(clip)

            all_clips.append(torch.stack(view_clips))
    return all_clips


@torch.no_grad()
def predict_study(model, h5_path: str, device: str = "cpu", chunk: int = 16):
    """
    Run multi-view, multi-clip inference on a single study.

    Parameters
    ----------
    model : nn.Module
        Loaded EchoAI-Peds model (already in eval mode).
    h5_path : str
        Path to an HDF5 file containing echo views.
    device : str
        "cpu" or "cuda".
    chunk : int
        Max clips per forward pass (controls GPU memory).

    Returns
    -------
    probs : np.ndarray, shape (num_labels,)
        Predicted probabilities averaged across all views and clips.
    """
    device = torch.device(device)
    all_view_clips = extract_clips_from_h5(h5_path)

    # Flatten all clips
    flat_clips = []
    num_clips_per_view = []
    for view_tensor in all_view_clips:
        num_clips_per_view.append(view_tensor.size(0))
        flat_clips.append(view_tensor)

    all_clips = torch.cat(flat_clips, dim=0).to(device)

    # Forward in chunks
    all_probs = []
    for i in range(0, all_clips.size(0), chunk):
        logits = model(all_clips[i : i + chunk])
        all_probs.append(torch.sigmoid(logits))
    all_probs = torch.cat(all_probs, dim=0)

    # Average: clips -> views -> study
    view_preds = []
    start = 0
    for n in num_clips_per_view:
        view_preds.append(all_probs[start : start + n].mean(dim=0))
        start += n

    study_probs = torch.stack(view_preds).mean(dim=0).cpu().numpy()
    return study_probs


# ───────────────────────── Validation ────────────────────────

def validate_input(x: torch.Tensor):
    """Ensures input is of shape (B, 3, 16, 224, 224)."""
    if x.ndim != 5:
        raise ValueError(f"Input must be 5D (B, C, T, H, W), got shape {x.shape}")
    B, C, T, H, W = x.shape
    if C != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {C}")
    if T != 16:
        raise ValueError(f"Expected 16 frames, got {T}")
    if H != 224 or W != 224:
        raise ValueError(f"Expected spatial size 224x224, got {H}x{W}")


# ───────────────────────── Labels ────────────────────────────

LABEL_NAMES = ['Prior cardiac surgery (excluding any interventional procedure)', 'PFO (patent foramen ovale)', 'VSD (ventricular septal defect)', 'ASD (atrial septal defect)', 'RV dilation', 'RV hypertrophy', 'LV (left ventricular) systolic dysfunction', 'VSD repair', 'LV dilation', 'Heart transplant', 'Pericardial effusion', 'Tetralogy of Fallot', 'RV (right ventricular) systolic dysfunction', 'PDA (patent ductus arteriosus)', 'LV hypertrophy', 'Pulmonary atresia', 'LV diastolic dysfunction', 'Glenn procedure', 'ASD repair', 'Aortic root dilation', 'Bicuspid aortic valve', 'MAPCAs (major aortopulmonary collateral arteries)', 'Pacemaker', 'PDA ligation', 'Fontan procedure', 'HCM (hypertrophic cardiomyopathy)', 'DORV (double outlet right ventricle)', 'TGA (transposition of the great arteries)']



# ───────────────────────── Main ──────────────────────────────

if __name__ == "__main__":
    CHECKPOINT_PATH = "placeholder.pth"
    H5_PATH = "placeholder_study.h5"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_echoai_peds_model(
        checkpoint_path=CHECKPOINT_PATH,
        num_labels=28,
        device=DEVICE,
    )
    print("EchoAI-Peds model loaded successfully.")

    # ── Multi-view, multi-clip inference on a single study ───
    probs = predict_study(model, H5_PATH, device=DEVICE)

    print("\nPredictions:")
    for name, p in zip(LABEL_NAMES, probs):
        print(f"  {name}: {p:.4f}")