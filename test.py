# from numba import cuda
# print("cuda.is_available:", cuda.is_available())
# try:
#     print("gpus:", list(cuda.gpus))
# except Exception as e:
#     print("list(cuda.gpus) failed:", repr(e))

# try:
#     cuda.select_device(0)
# except Exception as e:
#     print(f"select devie failed: {e}")

# try:
#     ctx = cuda.current_context()
#     print("context:", ctx)
#     print("device:", ctx.device)
# except Exception as e:
#     print("current_context failed:", repr(e))


# ctx_test.py
import os
print("PID:", os.getpid())

from numba import cuda
print("is_available:", cuda.is_available())
print("gpus:", list(cuda.gpus))

with cuda.gpus[0]:
    # Inside this block, cuda.current_context() will now work
    ctx = cuda.current_context()
    data = cuda.to_device([1, 2, 3])

# with cuda.gpus[0]:
#     # cuda.select_device(0)
#     cuda.to_device([1, 2, 3]) 
#     ctx = cuda.current_context()
#     print("context ok:", ctx)

# import os
# from numba import cuda
# import torch

# print("PID:", os.getpid())
# print("is_available:", cuda.is_available(), torch.cuda.is_available())

# # Optional: List devices to ensure visibility
# print("gpus:", list(cuda.gpus))

# # FIX: current_context() initializes and returns the context for the current thread
# # You can pass the device index directly to it.
# ctx = cuda.current_context(0) 

print("context ok:", ctx)
# Verify by querying device memory info
free, total = ctx.get_memory_info()
print(f"Free: {free / 1024**2:.0f}MB, Total: {total / 1024**2:.0f}MB")


# import ctypes
# lib = ctypes.CDLL('/lib64/libcuda.so.1')
# print("loaded:", lib._name)