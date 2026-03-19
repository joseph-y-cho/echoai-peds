import torch
import torch.nn as nn
import torchvision


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

    # Support both wrapped and raw state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def validate_input(x: torch.Tensor):
    """
    Ensures input is of shape (B, 3, 16, 224, 224)
    """
    if x.ndim != 5:
        raise ValueError(f"Input must be 5D (B, C, T, H, W), got shape {x.shape}")

    B, C, T, H, W = x.shape

    if C != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {C}")
    if T != 16:
        raise ValueError(f"Expected 16 frames, got {T}")
    if H != 224 or W != 224:
        raise ValueError(f"Expected spatial size 224x224, got {H}x{W}")


if __name__ == "__main__":
    CHECKPOINT_PATH = "placeholder.pth"  
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_echoai_peds_model(
        checkpoint_path=CHECKPOINT_PATH,
        num_labels=28,
        device=DEVICE,
    )

    print("EchoAI-Peds model loaded successfully.")

    # ------------------------------------------------------------------
    # Example forward pass (REQUIRED INPUT FORMAT)
    # ------------------------------------------------------------------
    # Input must be: (B, 3, 16, 224, 224)
    dummy_input = torch.randn(1, 3, 16, 224, 224).to(DEVICE)

    validate_input(dummy_input)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
