
def compute_prefill_oi(num_qo_heads, num_kv_heads, p, dtype_size=2):
    return (2 * num_qo_heads * p) / (dtype_size * (num_qo_heads + 2 * num_kv_heads))

# Example usage
oi = compute_prefill_oi(num_qo_heads=64, num_kv_heads=8, p=2048, dtype_size=2)
print(f"Operational Intensity: {oi:.2f} FLOPs/byte")