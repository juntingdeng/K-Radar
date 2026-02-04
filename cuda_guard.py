# cuda_guard.py
import os
from torch.utils.data import get_worker_info

def assert_not_in_dataloader_worker(where="Numba CUDA"):
    wi = get_worker_info()
    if wi is not None:
        raise RuntimeError(
            f"{where} called inside a DataLoader worker (id={wi.id}, pid={os.getpid()}). "
            "Set num_workers=0 or move CUDA/Numba calls to the main process."
        )
