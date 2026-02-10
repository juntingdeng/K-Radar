from numba import cuda

# cuda.current_context(0)
s = cuda.stream()
print("stream ok", s)

# ctx = cuda.current_context(0)  # get device 0 context
# ctx.push()                     # make it CURRENT on this thread
# try:
#     s = cuda.stream()
#     print("stream ok", s)
# finally:
#     ctx.pop()
