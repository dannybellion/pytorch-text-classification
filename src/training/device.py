import torch

from src.logging.logging import log_print


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log_print(body="Using Apple Silicon GPU (MPS)", colour="green")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log_print(body="Using NVIDIA GPU (CUDA)", colour="green")
    else:
        device = torch.device("cpu")
        log_print(body="Using CPU", colour="green")
        
    return device