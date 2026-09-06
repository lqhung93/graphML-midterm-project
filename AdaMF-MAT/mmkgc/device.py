import torch


def get_device(use_gpu=True):
    """Resolve the compute device.

    On Linux with CUDA this returns cuda:0, matching the original
    hardcoded .cuda() calls. Falls back to Apple MPS, then CPU, so the same
    code runs locally for smoke tests.
    """
    if not use_gpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
