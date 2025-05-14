import torch
import numpy as np
import matplotlib.pyplot as plt
from flash_attn_interface import flash_attn_func
import flashinfer
import time

def get_model_config(model_name):
    if model_name == "llama2-7B":
        return {
            "hidden_size": 4096,
            "num_heads": 32,
            "head_dim": 128,
            "c_range": range(7, 13),  # 2^7 to 2^12
        }
    elif model_name in ["llama3-8B", "llama3-70B"]:
        return {
            "hidden_size": 4096 if model_name == "llama3-8B" else 8192,
            "num_heads": 32 if model_name == "llama3-8B" else 64,
            "head_dim": 128,
            "c_range": range(7, 16),  # 2^7 to 2^15
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

def calculate_bandwidth(data_bytes, elapsed_time):
    """Calculate memory bandwidth in GB/s from data processed and time elapsed."""
    return data_bytes / (elapsed_time * 1e9)  # Convert to GB/s

def append_to_paged_kv_cache(k_append, v_append, paged_kv_cache, kv_indices, kv_indptr, 
                             kv_last_page_len, page_size, batch_size=1):
    """Append new key-value pairs to a paged KV cache.
    
    Args:
        k_append: New keys to append with shape [nnz, num_heads, head_dim]
        v_append: New values to append with shape [nnz, num_heads, head_dim]
        paged_kv_cache: Existing paged KV cache
        kv_indices: Page indices in the KV cache
        kv_indptr: Indptr for the paged KV cache with shape [batch_size + 1]
        kv_last_page_len: Length of the last page for each sequence in batch
        page_size: Size of each page
        batch_size: Number of sequences in the batch
    """
    # Get the number of new tokens to append
    nnz_kv = k_append.size(0)
    
    # Calculate the current sequence lengths
    seq_lengths = []
    for i in range(batch_size):
        num_pages = kv_indptr[i+1] - kv_indptr[i]
        if num_pages > 0:
            # Full pages + last page
            seq_len = (num_pages - 1) * page_size + kv_last_page_len[i].item()
        else:
            seq_len = 0
        seq_lengths.append(seq_len)
    
    # Create a tensor of sequence lengths
    seq_lens = torch.tensor(seq_lengths, dtype=torch.int32, device="cuda")
    
    # Create batch indices and positions for each token
    if batch_size == 1:
        # Simple case: all tokens belong to the single sequence
        batch_indices = torch.zeros(nnz_kv, dtype=torch.int32, device="cuda")
        positions = torch.arange(seq_lens[0], seq_lens[0] + nnz_kv, dtype=torch.int32, device="cuda")
    else:
        # Create fake indptr for append (assuming all tokens go to the same batch for simplicity)
        # In a real implementation, this would need to handle different distributions of tokens
        fake_append_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        fake_append_indptr[-1] = nnz_kv
        
        # Get batch indices and positions
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            fake_append_indptr, seq_lens, nnz_kv
        )
    
    # Append to paged KV cache
    flashinfer.append_paged_kv_cache(
        k_append,
        v_append,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len
    )

def evaluate_flash_attn_decode_seq(model_config, c, page_size=16):
    """Evaluate FlashAttention decode performance with varying sequence lengths."""
    batch_size = 1
    context_len = 2**c
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    
    # Prepare for decode (single token query and cached KV)
    # Shape: [batch_size, 1, num_heads, head_dim] for q
    # For KV cache: [batch_size, context_len, num_heads, head_dim]
    q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.float16, device="cuda")
    k_cache = torch.randn(batch_size, context_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    v_cache = torch.randn(batch_size, context_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    
    # Current token position in sequence
    cache_seqlens = torch.tensor([context_len], dtype=torch.int32, device="cuda")
    
    # Size of data processed in bytes (q, k_cache, v_cache + output)
    data_size = (
        q.numel() * q.element_size() +  # q tensor
        k_cache.numel() * k_cache.element_size() +  # k_cache tensor
        v_cache.numel() * v_cache.element_size() +  # v_cache tensor
        (batch_size * 1 * num_heads * head_dim * 2)  # output tensor (same shape as q)
    )
    
    # Warmup
    for _ in range(3):
        _ = flash_attn_func(
            q, k_cache, v_cache,
            causal=True,
            softmax_scale=1.0 / (head_dim ** 0.5)
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(20):  # More iterations for reliable timing
        _ = flash_attn_func(
            q, k_cache, v_cache,
            causal=True,
            softmax_scale=1.0 / (head_dim ** 0.5)
        )
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / 20  # Average time per call
    bandwidth = calculate_bandwidth(data_size, elapsed_time)
    
    return bandwidth

def evaluate_flashinfer_decode_seq(model_config, c, page_size=16):
    """Evaluate FlashInfer decode performance with varying sequence lengths."""
    batch_size = 1
    context_len = 2**c
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    
    # Calculate number of pages needed
    num_pages = (context_len + page_size - 1) // page_size
    
    # Calculate last page length
    last_page_len = context_len % page_size
    if last_page_len == 0:
        last_page_len = page_size
    
    # Create query tensor: [batch_size, num_heads, head_dim]
    q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    
    # 1. Create and populate the paged KV cache
    
    # Create the indptr for KV cache
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device="cuda")
    
    # Create the indices for the pages
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    
    # Last page length
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device="cuda")
    
    # Create paged KV cache: [num_pages, 2, page_size, num_heads, head_dim]
    paged_kv = torch.zeros(
        num_pages, 2, page_size, num_heads, head_dim, 
        dtype=torch.float16, device="cuda"
    )
    
    # Generate the initial keys and values
    k_initial = torch.randn(context_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    v_initial = torch.randn(context_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    
    # Append initial keys and values to the paged KV cache
    append_to_paged_kv_cache(
        k_initial, v_initial, 
        paged_kv, kv_indices, kv_indptr, kv_last_page_len,
        page_size, batch_size
    )
    
    # Data size calculation (q + paged_kv + output)
    data_size = (
        q.numel() * q.element_size() +
        paged_kv.numel() * paged_kv.element_size() +
        (batch_size * num_heads * head_dim * 2)  # output size (same shape as q)
    )
    
    # Warmup
    for _ in range(3):
        _ = flashinfer.single_decode_with_kv_cache(
            q=q,
            paged_kv_cache=paged_kv,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            last_page_len=kv_last_page_len,
            page_size=page_size,
            scale=1.0 / (head_dim ** 0.5)
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(20):
        _ = flashinfer.single_decode_with_kv_cache(
            q=q,
            paged_kv_cache=paged_kv,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            last_page_len=kv_last_page_len,
            page_size=page_size,
            scale=1.0 / (head_dim ** 0.5)
        )
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / 20
    bandwidth = calculate_bandwidth(data_size, elapsed_time)
    
    return bandwidth

def evaluate_flash_attn_decode_batch(model_config, batch_size, context_len=1024, page_size=16):
    """Evaluate FlashAttention decode performance with varying batch sizes."""
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    
    # Prepare for decode (single token query and cached KV)
    # Shape: [batch_size, 1, num_heads, head_dim] for q
    # For KV cache: [batch_size, context_len, num_heads, head_dim]
    q = torch.randn(batch_size, 1, num_heads, head_dim, dtype=torch.float16, device="cuda")
    k_cache = torch.randn(batch_size, context_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    v_cache = torch.randn(batch_size, context_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    
    # Current token position in sequence
    cache_seqlens = torch.full((batch_size,), context_len, dtype=torch.int32, device="cuda")
    
    # Size of data processed in bytes (q, k_cache, v_cache + output)
    data_size = (
        q.numel() * q.element_size() +  # q tensor
        k_cache.numel() * k_cache.element_size() +  # k_cache tensor
        v_cache.numel() * v_cache.element_size() +  # v_cache tensor
        (batch_size * 1 * num_heads * head_dim * 2)  # output tensor (same shape as q)
    )
    
    # Warmup
    for _ in range(3):
        _ = flash_attn_func(
            q, k_cache, v_cache,
            causal=True,
            softmax_scale=1.0 / (head_dim ** 0.5)
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(20):
        _ = flash_attn_func(
            q, k_cache, v_cache,
            causal=True,
            softmax_scale=1.0 / (head_dim ** 0.5)
        )
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / 20
    bandwidth = calculate_bandwidth(data_size, elapsed_time)
    
    return bandwidth

def evaluate_flashinfer_decode_batch(model_config, batch_size, context_len=1024, page_size=16):
    """Evaluate FlashInfer decode performance with varying batch sizes."""
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    
    # Calculate number of pages needed per sequence
    num_pages_per_seq = (context_len + page_size - 1) // page_size
    total_pages = batch_size * num_pages_per_seq
    
    # Create query tensor: [batch_size, num_heads, head_dim]
    q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    
    # Calculate last page length
    last_page_len = context_len % page_size
    if last_page_len == 0:
        last_page_len = page_size
    
    # 1. Create and populate the paged KV cache
    
    # Create the indptr for KV cache - pointing to pages for each sequence
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    for i in range(1, batch_size + 1):
        kv_indptr[i] = i * num_pages_per_seq
    
    # Create the indices for the pages
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    
    # Last page length for each sequence
    kv_last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32, device="cuda")
    
    # Create paged KV cache: [total_pages, 2, page_size, num_heads, head_dim]
    paged_kv = torch.zeros(
        total_pages, 2, page_size, num_heads, head_dim, 
        dtype=torch.float16, device="cuda"
    )
    
    # Generate the initial keys and values for each sequence
    for i in range(batch_size):
        # Create sequence-specific keys and values
        k_seq = torch.randn(context_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
        v_seq = torch.randn(context_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
        
        # Get pages for this sequence
        seq_indices = kv_indices[kv_indptr[i]:kv_indptr[i+1]]
        seq_indptr = torch.tensor([0, len(seq_indices)], dtype=torch.int32, device="cuda")
        seq_last_page_len = kv_last_page_len[i:i+1]
        
        # Append to specific slice of the paged KV cache
        batch_indices = torch.zeros(context_len, dtype=torch.int32, device="cuda")
        positions = torch.arange(context_len, dtype=torch.int32, device="cuda")
        
        flashinfer.append_paged_kv_cache(
            k_seq, v_seq,
            batch_indices, positions,
            paged_kv, seq_indices,
            seq_indptr, seq_last_page_len,
            kv_layout="NHD"
        )
    
    # Data size calculation (q + paged_kv + output)
    data_size = (
        q.numel() * q.element_size() +
        paged_kv.numel() * paged_kv.element_size() +
        (batch_size * num_heads * head_dim * 2)  # output size (same shape as q)
    )
    
    # Warmup
    for _ in range(3):
        _ = flashinfer.batch_decode_with_paged_kv_cache(
            q=q,
            paged_kv_cache=paged_kv,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            last_page_len=kv_last_page_len,
            page_size=page_size,
            scale=1.0 / (head_dim ** 0.5)
        )
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(20):
        _ = flashinfer.batch_decode_with_paged_kv_cache(
            q=q,
            paged_kv_cache=paged_kv,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            last_page_len=kv_last_page_len,
            page_size=page_size,
            scale=1.0 / (head_dim ** 0.5)
        )
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) / 20
    bandwidth = calculate_bandwidth(data_size, elapsed_time)
    
    return bandwidth

def evaluate_flash_attn_decode_pagesize(model_config, page_sizes, batch_size=128, context_len=1024):
    """Evaluate FlashAttention decode performance with varying page sizes."""
    # FlashAttention doesn't use paged KV cache, so performance should be consistent
    # We'll evaluate it once and replicate the result
    bandwidth = evaluate_flash_attn_decode_batch(model_config, batch_size, context_len)
    return [bandwidth] * len(page_sizes)

def evaluate_flashinfer_decode_pagesize(model_config, page_sizes, batch_size=128, context_len=1024):
    """Evaluate FlashInfer decode performance with varying page sizes."""
    bandwidths = []
    
    for page_size in page_sizes:
        bandwidth = evaluate_flashinfer_decode_batch(
            model_config, batch_size, context_len, page_size
        )
        bandwidths.append(bandwidth)
    
    return bandwidths

def plot_decode_performance():
    # Model configurations
    models = ["llama2-7B", "llama3-8B", "llama3-70B"]
    model_configs = {name: get_model_config(name) for name in models}
    model_display_names = ["LLaMA2-7B", "LLaMA3-8B", "LLaMA3-70B"]
    
    # 1. Varying sequence length with fixed batch size
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for idx, model_name in enumerate(models):
        config = model_configs[model_name]
        c_values = list(config["c_range"])
        c_seq_lengths = [2**c for c in c_values]
        
        # Evaluate both implementations
        flash_attn_bandwidths = [evaluate_flash_attn_decode_seq(config, c) for c in c_values]
        flashinfer_bandwidths = [evaluate_flashinfer_decode_seq(config, c) for c in c_values]
        
        # Plot
        ax = axs1[idx]
        ax.plot(c_seq_lengths, flash_attn_bandwidths, "b-o", label="FlashAttention-3")
        ax.plot(c_seq_lengths, flashinfer_bandwidths, "r-x", label="FlashInfer")
        ax.set_xscale("log", base=2)
        ax.set_title(model_display_names[idx])
        ax.set_xlabel("Context Length")
        ax.set_ylabel("Memory Bandwidth (GB/s)")
        ax.grid(True, which="both")
        ax.legend()
    
    fig1.suptitle("Decode Attention Memory Bandwidth - Varying Context Length", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("decode_performance_seq.png")
    plt.close()
    
    # 2. Varying batch size with fixed sequence length
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    batch_sizes = 2 ** np.arange(7)  # 2^0 to 2^6
    context_len = 1024
    
    for idx, model_name in enumerate(models):
        config = model_configs[model_name]
        
        # Evaluate both implementations
        flash_attn_bandwidths = [
            evaluate_flash_attn_decode_batch(config, b, context_len) for b in batch_sizes
        ]
        flashinfer_bandwidths = [
            evaluate_flashinfer_decode_batch(config, b, context_len) for b in batch_sizes
        ]
        
        # Plot
        ax = axs2[idx]
        ax.plot(batch_sizes, flash_attn_bandwidths, "b-o", label="FlashAttention-3")
        ax.plot(batch_sizes, flashinfer_bandwidths, "r-x", label="FlashInfer")
        ax.set_xscale("log", base=2)
        ax.set_title(model_display_names[idx])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Memory Bandwidth (GB/s)")
        ax.grid(True, which="both")
        ax.legend()
    
    fig2.suptitle("Decode Attention Memory Bandwidth - Varying Batch Size", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("decode_performance_batch.png")
    plt.close()
    
    # 3. Varying page size with fixed batch size and sequence length
    fig3, axs3 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    page_sizes = [1, 2, 4, 8, 16]
    batch_size = 128
    context_len = 1024
    
    for idx, model_name in enumerate(models):
        config = model_configs[model_name]
        
        # Evaluate both implementations
        flash_attn_bandwidths = evaluate_flash_attn_decode_pagesize(
            config, page_sizes, batch_size, context_len
        )
        flashinfer_bandwidths = evaluate_flashinfer_decode_pagesize(
            config, page_sizes, batch_size, context_len
        )
        
        # Plot
        ax = axs3[idx]
        ax.plot(page_sizes, flash_attn_bandwidths, "b-o", label="FlashAttention-3")
        ax.plot(page_sizes, flashinfer_bandwidths, "r-x", label="FlashInfer")
        ax.set_title(model_display_names[idx])
        ax.set_xlabel("Page Size")
        ax.set_ylabel("Memory Bandwidth (GB/s)")
        ax.set_xticks(page_sizes)
        ax.grid(True)
        ax.legend()
    
    fig3.suptitle("Decode Attention Memory Bandwidth - Varying Page Size", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("decode_performance_pagesize.png")
    plt.close()

if __name__ == "__main__":
    plot_decode_performance() 