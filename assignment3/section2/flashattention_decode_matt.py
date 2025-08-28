from einops import rearrange
import torch
import numpy as np
import matplotlib.pyplot as plt
from flash_attn_interface import *
import flashinfer
import time
from flashinfer.decode import PosEncodingMode

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
    seqlen,
    page_size=16,
    batch_size=1,
    num_splits=0,
    has_qv=False,
):
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    num_kv_heads = model_config["num_kv_heads"]
    headdim_v = head_dim
    headdim = head_dim
    seqlen_q = seqlen

    cache_seqlens = torch.tensor([seqlen] * batch_size, device="cuda", dtype=torch.int)
    q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(batch_size, seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

    if page_size is not None:
        assert seqlen % page_size == 0
        k_cache, v_cache = [
            rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k_cache, v_cache]
        ]
        page_table = rearrange(
            torch.arange(batch_size * seqlen // page_size, device="cuda", dtype=torch.int32),
            "(b s) -> b s",
            s=seqlen // page_size,
        )
    else:
        page_table = None

    scheduler_metadata = get_scheduler_metadata(
        batch_size, seqlen_q, seqlen, num_heads, num_kv_heads, head_dim, cache_seqlens,
        q.dtype, headdim_v=headdim_v, page_size=page_size, causal=True,
    )

    fn = lambda: flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens=cache_seqlens,
        num_splits=num_splits, qv=None, page_table=page_table,
        causal=True, scheduler_metadata=scheduler_metadata,
    )
    time.sleep(1)
    ms = do_bench(fn, warmup=1, rep=10)  # returns milliseconds

    # Calculate total memory IO
    total_seqlen = seqlen * batch_size
    bytes_q = q.numel() * 2  # fp16 = 2 bytes
    bytes_k = total_seqlen * num_kv_heads * head_dim * 2
    bytes_v = total_seqlen * num_kv_heads * head_dim * 2
    bytes_out = batch_size * seqlen_q * num_heads * head_dim * 2

    total_bytes = bytes_q + bytes_k + bytes_v + bytes_out  # total memory read+write per run

    gbps = total_bytes / (ms / 1000.0) / 1e9  # GB/s

    return gbps

def bench_flashinfer(model_config, seq_len, batch_size, page_size=16):
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    num_kv_heads = num_heads

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer,"NHD") 
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device="cuda")

    q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
    kv_cache = torch.randn(1, batch_size, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")

    decode_wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        0,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size
    )

    for _ in range(3):
        _ = decode_wrapper.run(q, kv_cache[0])
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(10):
        _ = decode_wrapper.run(q, kv_cache[0])
    torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time

    bytes_q = q.numel() * 2
    bytes_kv = kv_cache.numel() * 2
    bytes_out = (batch_size * num_heads * head_dim) * 2
    total_bytes = bytes_q + bytes_kv + bytes_out

    gbps = total_bytes / elapsed / 1e9

    return gbps

def plot_page_size_variation(model_name):
    config = get_model_config(model_name)
    page_sizes = [1, 2, 4, 8, 16]
    batch_size = 128
    c = 1024

    flash_attn_results = []
    flashinfer_results = []

    for page_size in page_sizes:
        gbps_flash_attn = bench_flash_attention(config, c, page_size=page_size, batch_size=batch_size)
        gbps_flashinfer = bench_flashinfer(config, c, batch_size=batch_size, page_size=page_size)
        flash_attn_results.append(gbps_flash_attn)
        flashinfer_results.append(gbps_flashinfer)

    plt.figure(figsize=(8,6))
    plt.plot(page_sizes, flash_attn_results, label="FlashAttention-3", marker="o")
    plt.plot(page_sizes, flashinfer_results, label="FlashInfer", marker="x")
    plt.xlabel("Page Size")
    plt.ylabel("Memory Bandwidth Utilization (GB/s)")
    plt.title(f"{model_name} Page Size Sweep")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"page_size_variation_{model_name}.png")
    plt.close()

def plot_performance():
    # Sequence lengths (powers of 2)
    p_llama2 = 2 ** np.arange(7, 13)  # 2^7 to 2^12
    p_llama3 = 2 ** np.arange(7, 16)  # 2^7 to 2^15

    # Get real TFLOPs data
    llama2_config = get_model_config("llama2-7B")
    llama3_8b_config = get_model_config("llama3-8B")
    llama3_70b_config = get_model_config("llama3-70B")

    # Evaluate FlashAttention-3
    llama2_flashattn = [bench_flash_attention(llama2_config, 2 ** p) for p in range(7, 13)]
    llama3_8b_flashattn = [bench_flash_attention(llama3_8b_config, 2 ** p) for p in range(7, 16)]
    llama3_70b_flashattn = [bench_flash_attention(llama3_70b_config, 2 ** p) for p in range(7, 16)]

    # Evaluate FlashInfer
    llama2_flashinfer = [bench_flashinfer(llama2_config, seq_len=1, batch_size=(2 ** p)) for p in range(7, 13)]
    llama3_8b_flashinfer = [bench_flashinfer(llama3_8b_config, seq_len=1, batch_size=(2 ** p)) for p in range(7, 16)]
    llama3_70b_flashinfer = [bench_flashinfer(llama3_70b_config, seq_len=1, batch_size=(2 ** p)) for p in range(7, 16)] 

    # Plotting setup
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    models = ["LLaMA2-7B", "LLaMA3-8B", "LLaMA3-70B"]

    # LLaMA2-7B plot
    axs[0].plot(p_llama2, llama2_flashattn, label="FlashAttention-3", marker="o")
    axs[0].plot(p_llama2, llama2_flashinfer, label="FlashInfer", marker="x")
    axs[0].set_xscale("log", base=2)
    axs[0].set_title(models[0])
    axs[0].set_xlabel("p (sequence length)")
    axs[0].set_ylabel("Memory Bandwidth Utilization (GB/s)")
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
    fig.suptitle("Memory Bandwidth Utilization", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("attention_performance_seq.png")
    plt.close()

    # Plot 2: Varying batch size with fixed sequence length
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Batch sizes from 2^0 to 2^6
    batch_sizes = 2 ** np.arange(7)
    batch_size_logs = np.arange(7)  # log2 of batch sizes

    # Evaluate FlashAttention-3
    llama2_flashattn_batch = [
        bench_flash_attention(llama2_config, 1024, batch_size=b) for b in batch_sizes
    ]
    llama3_8b_flashattn_batch = [
        bench_flash_attention(llama3_8b_config, 1024, batch_size=b) for b in batch_sizes
    ]
    llama3_70b_flashattn_batch = [
        bench_flash_attention(llama3_70b_config, 1024, batch_size=b) for b in batch_sizes
    ]

    # Evaluate FlashInfer
    llama2_flashinfer_batch = [
        bench_flashinfer(llama2_config, 1024, batch_size=b) for b in batch_sizes
    ]
    llama3_8b_flashinfer_batch = [
        bench_flashinfer(llama3_8b_config, 1024, batch_size=b) for b in batch_sizes
    ]
    llama3_70b_flashinfer_batch = [
        bench_flashinfer(llama3_70b_config, 1024, batch_size=b) for b in batch_sizes
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
    axs2[0].set_ylabel("Memory Bandwidth Utilization (GB/s)")
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

    # Overall figure title and layout for batch size variation
    fig2.suptitle(
        "Memory Bandwidth Utilization (Batch Size Sweep)",
        fontsize=16,
    )
    axs2[0].set_xlabel("Batch Size")
    axs2[1].set_xlabel("Batch Size")
    axs2[2].set_xlabel("Batch Size")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("attention_performance_batch.png")
    plt.close()


if __name__ == "__main__":
    plot_performance()
    plot_page_size_variation("llama2-7B")
    plot_page_size_variation("llama3-8B")
    plot_page_size_variation("llama3-70B")