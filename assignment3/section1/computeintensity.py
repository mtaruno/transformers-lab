import numpy as np

# Constants
dtype_size = 4  # bytes per element, 4 for float32, 2 for float16
M_values = np.arange(128, 2048 + 1, 128)  # from 128 to 2048 with step 128
print(M_values)
nk_cases = [
    (512, 512),
    (4096, 4096),
    (14336, 4096),
    (4096, 1024),
    (1024, 4096)
]

def compute_oi(M, N, K, dtype_size):
    flops = 2 * M * N * K
    bytes_moved = dtype_size * (M*K + K*N + 2*M*N)
    return flops / bytes_moved

# Create results table
for (N, K) in nk_cases:
    print(f"\n(N, K) = ({N}, {K})")
    print(f"{'M':>5} | {'Operational Intensity':>25}")
    print("-" * 35)
    for M in M_values:
        oi = compute_oi(M, N, K, dtype_size)
        print(f"{M:5d} | {oi:25.4f}")