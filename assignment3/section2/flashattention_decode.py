from einops import rearrange
import torch
import numpy as np
import matplotlib.pyplot as plt
from flash_attn_interface import *
import flashinfer
import time


def get_model_config(model_name):
    if model_name == "llama2-7B":
        return {
            "hidden_size": 4096,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "p_range": range(7, 13),  # 2^7 to 2^12
        }
    elif model_name in ["llama3-8B", "llama3-70B"]:
        return {
            "hidden_size": 4096 if model_name == "llama3-8B" else 8192,
            "num_heads": 32 if model_name == "llama3-8B" else 64,
            "num_kv_heads": 8 if model_name == "llama3-8B" else 16,
            "head_dim": 128,
            "p_range": range(7, 16),  # 2^7 to 2^15
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_attention_tensors(batch_size, seq_len, num_heads, head_dim):
    """Create q, k, v tensors for attention computation.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        include_batch: Whether to include batch dimension (True for FlashAttention, False for FlashInfer)

    Returns:
        Tuple of (q, k, v) tensors
    """
    # Shape: [batch_size, seq_len, num_heads, head_dim]
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
    )

    return q, k, v


def do_bench(fn, warmup=1, rep=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # cuda event
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def bench_flash_attention(
    model_config,
    seq_len,
    page_size=16,
    batch_size=1,
):
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    num_kv_heads = model_config["num_kv_heads"]
    headdim_v = head_dim
    headdim = head_dim
    seq_len = 2**seq_len
    seqlen_q = seq_len

    print(
        f"\n{batch_size=}, {seq_len=}, {seqlen_q=}, {num_heads=}, {num_kv_heads=}, {head_dim=}, {headdim_v=}, {page_size=}"
    )
    cache_seqlens = torch.tensor([seq_len] * batch_size, device="cuda", dtype=torch.int)
    q = torch.randn(
        batch_size, seqlen_q, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    v_cache = torch.randn(
        batch_size, seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
    )
    k_cache = torch.randn(
        batch_size, seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
    )
    if page_size is not None:
        assert seq_len % page_size == 0
        k_cache, v_cache = [
            rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size)
            for x in [k_cache, v_cache]
        ]
        page_table = rearrange(
            torch.arange(
                batch_size * seq_len // page_size, device="cuda", dtype=torch.int32
            ),
            "(b s) -> b s",
            s=seq_len // page_size,
        )
    else:
        page_table = None

    # Precomputing this saves ~2us
    scheduler_metadata = get_scheduler_metadata(
        batch_size,
        seqlen_q,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        cache_seqlens,
        q.dtype,
        headdim_v=headdim_v,
        page_size=page_size,
        causal=True,
    )
    # scheduler_metadata = None
    # breakpoint()
    fn0 = lambda: flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        causal=True,
        scheduler_metadata=scheduler_metadata,
    )
    # time.sleep(1)  # to avoid power throttlingF
    # Time in ms
    t0 = do_bench(fn0, warmup=3, rep=10)
    # exit(0)

    # should_run_flashmla = attn_variant == "mla" and page_size == 64 and flash_mla_with_kvcache is not None
    should_run_flashmla = False
    torch.manual_seed(0)

    total_seqlen = (
        seq_len * batch_size if cache_seqlens is None else cache_seqlens.sum().item()
    )
    
    # Calculate memory I/O in bytes (16-bit = 2 bytes per element)
    # For decode case (seqlen_q tokens attending to KV cache)
    bytes_per_element = 2  # float16 = 2 bytes
    
    # 1. Read from KV cache (K and V are separate)
    kv_cache_read = total_seqlen * num_kv_heads * head_dim * 2  # K and V
    
    # 2. Read query
    q_read = batch_size * seqlen_q * num_heads * head_dim
    
    # 3. Write output
    output_write = batch_size * seqlen_q * num_heads * head_dim
    
    # Total memory I/O in bytes
    mem_io_bytes = (kv_cache_read + q_read + output_write) * bytes_per_element
    
    # Return memory bandwidth in GB/s
    return mem_io_bytes / (t0 * 1e-3) / 1e9  # t0 is in ms, convert to seconds


def bench_flashinfer(model_config, seq_len, batch_size=1, page_size=16):
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    num_kv_heads = model_config["num_kv_heads"]  # Use the model's actual kv_heads

    seq_len = 2**seq_len
    
    print(
        f"\n{batch_size=}, {seq_len=}, {num_heads=}, {num_kv_heads=}, {head_dim=}, {page_size=}"
    )

    try:
        # For batch processing, we use BatchDecodeWithPagedKVCacheWrapper
        # Allocate workspace buffer (256MB to ensure enough space)
        workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

        # Create the wrapper with the correct layout
        decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )

        # Make sure sequence length is valid for the page size
        pages_per_seq = (seq_len + page_size - 1) // page_size
        max_num_pages = batch_size * pages_per_seq

        # Create paged KV indptr - indicates where each sequence starts in the paged KV cache
        paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        for i in range(1, batch_size + 1):
            paged_kv_indptr[i] = i * pages_per_seq

        # Create indices for the paged KV cache
        paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")

        # Handle partial pages
        last_page_len = seq_len % page_size
        if last_page_len == 0:
            last_page_len = page_size
        paged_kv_last_page_len = torch.full(
            (batch_size,), last_page_len, dtype=torch.int32, device="cuda"
        )

        # Create query tensor - shape: [batch_size * seq_len, num_heads, head_dim]
        # For decode, we use a single token per sequence
        seqlen_q = 1  # Single token decode
        q = torch.randn(
            batch_size * seqlen_q, num_heads, head_dim, dtype=torch.float16, device="cuda"
        )  # Use single token per sequence for decode

        # Create KV cache - shape: [num_pages, 2, page_size, num_kv_heads, head_dim]
        # First dim is num_pages, second is K (0) and V (1)
        kv_cache = torch.randn(
            max_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device="cuda",
        )

        # Plan the operation
        decode_wrapper.plan(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            page_size,
        )

        fn0 = lambda: decode_wrapper.run(q, kv_cache)
        
        # Time in ms
        t0 = do_bench(fn0, warmup=3, rep=10)

        # Calculate memory I/O in bytes (16-bit = 2 bytes per element)
        # For decode case (1 token attending to KV cache)
        bytes_per_element = 2  # float16 = 2 bytes
        
        # 1. Read from KV cache (K and V are separate)
        kv_cache_read = batch_size * seq_len * num_kv_heads * head_dim * 2  # K and V
        
        # 2. Read query
        q_read = batch_size * seqlen_q * num_heads * head_dim
        
        # 3. Write output
        output_write = batch_size * seqlen_q * num_heads * head_dim
        
        # Total memory I/O in bytes
        mem_io_bytes = (kv_cache_read + q_read + output_write) * bytes_per_element
        
        # Return memory bandwidth in GB/s
        return mem_io_bytes / (t0 * 1e-3) / 1e9
    
    except Exception as e:
        print(f"Error in FlashInfer benchmark: {e}")
        # Return a placeholder value to allow the plotting to continue
        return 0.0


def plot_performance():
    # Sequence lengths (powers of 2)
    p_llama2 = 2 ** np.arange(7, 13)  # 2^7 to 2^12
    p_llama3 = 2 ** np.arange(7, 16)  # 2^7 to 2^15

    # Get real TFLOPs data
    llama2_config = get_model_config("llama2-7B")
    llama3_8b_config = get_model_config("llama3-8B")
    llama3_70b_config = get_model_config("llama3-70B")

    # Evaluate FlashAttention-3
    llama2_flashattn = [bench_flash_attention(llama2_config, p) for p in range(7, 13)]
    llama3_8b_flashattn = [
        bench_flash_attention(llama3_8b_config, p) for p in range(7, 16)
    ]
    llama3_70b_flashattn = [
        bench_flash_attention(llama3_70b_config, p) for p in range(7, 16)
    ]

    # Evaluate FlashInfer
    llama2_flashinfer = [bench_flashinfer(llama2_config, p) for p in range(7, 13)]
    llama3_8b_flashinfer = [bench_flashinfer(llama3_8b_config, p) for p in range(7, 16)]
    llama3_70b_flashinfer = [
        bench_flashinfer(llama3_70b_config, p) for p in range(7, 16)
    ]

    # Plotting setup
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    models = ["LLaMA2-7B", "LLaMA3-8B", "LLaMA3-70B"]

    # LLaMA2-7B plot
    axs[0].plot(p_llama2, llama2_flashattn, label="FlashAttention-3", marker="o")
    axs[0].plot(p_llama2, llama2_flashinfer, label="FlashInfer", marker="x")
    axs[0].set_xscale("log", base=2)
    axs[0].set_title(models[0])
    axs[0].set_xlabel("p (sequence length)")
    axs[0].set_ylabel("Memory Bandwidth (GB/s)")
    axs[0].set_xticks(p_llama2)
    axs[0].set_xticklabels([str(p) for p in p_llama2])
    axs[0].legend()
    axs[0].grid(True, which="both")

    # LLaMA3-8B plot
    axs[1].plot(p_llama3, llama3_8b_flashattn, label="FlashAttention-3", marker="o")
    axs[1].plot(p_llama3, llama3_8b_flashinfer, label="FlashInfer", marker="x")
    axs[1].set_xscale("log", base=2)
    axs[1].set_title(models[1])
    axs[1].set_xlabel("p (sequence length)")
    axs[1].set_xticks(p_llama3)
    axs[1].set_xticklabels([str(p) for p in p_llama3])
    axs[1].legend()
    axs[1].grid(True, which="both")

    # LLaMA3-70B plot
    axs[2].plot(p_llama3, llama3_70b_flashattn, label="FlashAttention-3", marker="o")
    axs[2].plot(p_llama3, llama3_70b_flashinfer, label="FlashInfer", marker="x")
    axs[2].set_xscale("log", base=2)
    axs[2].set_title(models[2])
    axs[2].set_xlabel("p (sequence length)")
    axs[2].set_xticks(p_llama3)
    axs[2].set_xticklabels([str(p) for p in p_llama3])
    axs[2].legend()
    axs[2].grid(True, which="both")

    # Overall figure title and layout
    fig.suptitle("Decode Attention Memory Bandwidth per Layer", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("decode_attention_performance_seq.png")
    plt.close()

    # Plot 2: Varying batch size with fixed sequence length
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Batch sizes from 2^0 to 2^6
    batch_sizes = [2 ** i for i in range(7)]
    batch_size_logs = np.arange(7)  # log2 of batch sizes

    # Evaluate FlashAttention-3
    llama2_flashattn_batch = [
        bench_flash_attention(llama2_config, 10, batch_size=b) for b in batch_sizes
    ]
    llama3_8b_flashattn_batch = [
        bench_flash_attention(llama3_8b_config, 10, batch_size=b) for b in batch_sizes
    ]
    llama3_70b_flashattn_batch = [
        bench_flash_attention(llama3_70b_config, 10, batch_size=b)
        for b in batch_sizes
    ]

    # Evaluate FlashInfer
    llama2_flashinfer_batch = [
        bench_flashinfer(llama2_config, 10, batch_size=b) for b in batch_sizes
    ]
    llama3_8b_flashinfer_batch = [
        bench_flashinfer(llama3_8b_config, 10, batch_size=b) for b in batch_sizes
    ]
    llama3_70b_flashinfer_batch = [
        bench_flashinfer(llama3_70b_config, 10, batch_size=b) for b in batch_sizes
    ]

    # LLaMA2-7B plot
    axs2[0].plot(
        batch_size_logs, llama2_flashattn_batch, label="FlashAttention-3", marker="o"
    )
    axs2[0].plot(
        batch_size_logs, llama2_flashinfer_batch, label="FlashInfer", marker="x"
    )
    axs2[0].set_title(models[0])
    axs2[0].set_xlabel("log₂(batch_size)")
    axs2[0].set_ylabel("Memory Bandwidth (GB/s)")
    axs2[0].set_xticks(batch_size_logs)
    axs2[0].set_xticklabels([str(b) for b in batch_sizes])
    axs2[0].legend()
    axs2[0].grid(True, which="both")

    # LLaMA3-8B plot
    axs2[1].plot(
        batch_size_logs, llama3_8b_flashattn_batch, label="FlashAttention-3", marker="o"
    )
    axs2[1].plot(
        batch_size_logs, llama3_8b_flashinfer_batch, label="FlashInfer", marker="x"
    )
    axs2[1].set_title(models[1])
    axs2[1].set_xlabel("log₂(batch_size)")
    axs2[1].set_xticks(batch_size_logs)
    axs2[1].set_xticklabels([str(b) for b in batch_sizes])
    axs2[1].legend()
    axs2[1].grid(True, which="both")

    # LLaMA3-70B plot
    axs2[2].plot(
        batch_size_logs,
        llama3_70b_flashattn_batch,
        label="FlashAttention-3",
        marker="o",
    )
    axs2[2].plot(
        batch_size_logs, llama3_70b_flashinfer_batch, label="FlashInfer", marker="x"
    )
    axs2[2].set_title(models[2])
    axs2[2].set_xlabel("log₂(batch_size)")
    axs2[2].set_xticks(batch_size_logs)
    axs2[2].set_xticklabels([str(b) for b in batch_sizes])
    axs2[2].legend()
    axs2[2].grid(True, which="both")

    # Overall figure title and layout
    fig2.suptitle(
        "Decode Attention Memory Bandwidth per Layer (Batch Size Variation)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("decode_attention_performance_batch.png")
    plt.close()


if __name__ == "__main__":
    plot_performance()

    # llama2_config = get_model_config("llama2-7B")
    # bench_flash_attention(llama2_config, 10, batch_size=1)
