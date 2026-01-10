import torch

def get_device():
    """Get the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_device_type(device):
    """Extract device type from device string (e.g., 'cuda:0' -> 'cuda')."""
    if 'cuda' in device:
        return 'cuda'
    if 'mps' in device:
        return 'mps'
    return 'cpu'
