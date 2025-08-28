import torch
import numpy as np
import matplotlib.pyplot as plt
from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_func
import flashinfer
import time


def get_model_config(model_name):
    if model_name == "llama2-7B":
        return {
            "hidden_size": 4096,
            "num_heads": 32,
            "head_dim": 128,
            "p_range": range(7, 13),  # 2^7 to 2^12
        }
    elif model_name in ["llama3-8B", "llama3-70B"]:
        return {
            "hidden_size": 4096 if model_name == "llama3-8B" else 8192,
            "num_heads": 32 if model_name == "llama3-8B" else 64,
            "head_dim": 128,
            "p_range": range(7, 16),  # 2^7 to 2^15
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")



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


def create_attention_tensors(
    batch_size, seq_len, num_heads, head_dim, include_batch=True
):
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
    if include_batch:
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
    else:
        # Shape: [seq_len, num_heads, head_dim]
        q = torch.randn(
            seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
        )
        k = torch.randn(
            seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
        )

    return q, k, v


def evaluate_flash_attn_seq(model_config, p):
    batch_size = 1
    seq_len = 2**p
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]

    # Create input tensors
    q, k, v = create_attention_tensors(
        batch_size, seq_len, num_heads, head_dim, include_batch=True
    )

    fn0 = lambda: flash_attn_func(q, k, v, causal=False)

    time = do_bench(fn0, warmup=3, rep=10)

    # Calculate TFlops
    total_ops = 2 * seq_len * seq_len * num_heads * head_dim * batch_size
    tflops = (total_ops * 10) / time / 1e12

    return tflops


def evaluate_flashinfer_seq(model_config, p):
    batch_size = 1
    seq_len = 2**p
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]

    # For sequence length test, we just use single prefill
    # Create input tensors
    q, k, v = create_attention_tensors(
        batch_size, seq_len, num_heads, head_dim, include_batch=False
    )

    # Warmup
    for _ in range(3):
        _ = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=False,
        )
    torch.cuda.synchronize()

    fn0 = lambda: flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=False,
    )

    time = do_bench(fn0, warmup=3, rep=10)

    # Calculate TFlops
    total_ops = 2 * seq_len * seq_len * num_heads * head_dim * batch_size
    tflops = (total_ops * 10) / time / 1e12

    return tflops


def evaluate_flash_attn_batch(model_config, batch_size):
    seq_len = 1024  # Fixed sequence length
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]

    # Create input tensors
    q, k, v = create_attention_tensors(
        batch_size, seq_len, num_heads, head_dim, include_batch=True
    )

    # Warmup
    for _ in range(3):
        _ = flash_attn_func(q, k, v, causal=False)
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(10):
        _ = flash_attn_func(q, k, v, causal=False)
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate TFlops
    total_ops = 2 * seq_len * seq_len * num_heads * head_dim * batch_size
    tflops = (total_ops * 10) / (end_time - start_time) / 1e12

    return tflops


def evaluate_flashinfer_batch(model_config, batch_size):
    seq_len = 1024  # Fixed sequence length
    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_heads"]
    head_dim = model_config["head_dim"]
    num_kv_heads = num_heads  # Assuming same number of KV heads as Q heads for simplicity

    # For batch processing, we use BatchPrefillWithPagedKVCacheWrapper as per documentation

    # Allocate workspace buffer (128MB)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    # Create the wrapper
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )

    # Set up page size and max pages
    page_size = 16
    max_num_pages = (batch_size * seq_len + page_size - 1) // page_size

    # Create QO indptr (batch pointers)
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    for i in range(1, batch_size + 1):
        qo_indptr[i] = i * seq_len

    # Create paged KV indptr and indices
    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    pages_per_seq = (seq_len + page_size - 1) // page_size
    for i in range(1, batch_size + 1):
        paged_kv_indptr[i] = i * pages_per_seq

    paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")

    # Last page length (for potentially partial pages)
    last_page_len = seq_len % page_size
    if last_page_len == 0:
        last_page_len = page_size
    paged_kv_last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32, device="cuda")

    # Create input tensors
    q = torch.randn(batch_size * seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    kv_cache = torch.randn(1, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")

    # Plan the operation
    prefill_wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
    )

    # Warmup
    for _ in range(3):
        _ = prefill_wrapper.run(q, kv_cache[0])
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.time()
    for _ in range(10):
        _ = prefill_wrapper.run(q, kv_cache[0])
    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate TFlops
    total_ops = 2 * seq_len * seq_len * num_heads * head_dim * batch_size
    tflops = (total_ops * 10) / (end_time - start_time) / 1e12

    return tflops


def plot_performance():
    # Sequence lengths (powers of 2)
    p_llama2 = 2 ** np.arange(7, 13)  # 2^7 to 2^12
    p_llama3 = 2 ** np.arange(7, 16)  # 2^7 to 2^15

    # Get real TFLOPs data
    llama2_config = get_model_config("llama2-7B")
    llama3_8b_config = get_model_config("llama3-8B")
    llama3_70b_config = get_model_config("llama3-70B")

    # Evaluate FlashAttention-3
    llama2_flashattn = [evaluate_flash_attn_seq(llama2_config, p) for p in range(7, 13)]
    llama3_8b_flashattn = [
        evaluate_flash_attn_seq(llama3_8b_config, p) for p in range(7, 16)
    ]
    llama3_70b_flashattn = [
        evaluate_flash_attn_seq(llama3_70b_config, p) for p in range(7, 16)
    ]

    # Evaluate FlashInfer
    llama2_flashinfer = [
        evaluate_flashinfer_seq(llama2_config, p) for p in range(7, 13)
    ]
    llama3_8b_flashinfer = [
        evaluate_flashinfer_seq(llama3_8b_config, p) for p in range(7, 16)
    ]
    llama3_70b_flashinfer = [
        evaluate_flashinfer_seq(llama3_70b_config, p) for p in range(7, 16)
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
    axs[0].set_ylabel("Compute Utilization (TFLOPs)")
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
    fig.suptitle("Prefill Attention Compute Utilization per Layer", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("prefill_attention_performance_seq.png")
    plt.close()

    # Plot 2: Varying batch size with fixed sequence length
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Batch sizes from 2^0 to 2^6
    batch_sizes = 2 ** np.arange(7)
    batch_size_logs = np.arange(7)  # log2 of batch sizes

    # Evaluate FlashAttention-3
    llama2_flashattn_batch = [
        evaluate_flash_attn_batch(llama2_config, b) for b in batch_sizes
    ]
    llama3_8b_flashattn_batch = [
        evaluate_flash_attn_batch(llama3_8b_config, b) for b in batch_sizes
    ]
    llama3_70b_flashattn_batch = [
        evaluate_flash_attn_batch(llama3_70b_config, b) for b in batch_sizes
    ]

    # Evaluate FlashInfer
    llama2_flashinfer_batch = [
        evaluate_flashinfer_batch(llama2_config, b) for b in batch_sizes
    ]
    llama3_8b_flashinfer_batch = [
        evaluate_flashinfer_batch(llama3_8b_config, b) for b in batch_sizes
    ]
    llama3_70b_flashinfer_batch = [
        evaluate_flashinfer_batch(llama3_70b_config, b) for b in batch_sizes
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
    axs2[0].set_ylabel("Compute Utilization (TFLOPs)")
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
        "Prefill Attention Compute Utilization per Layer (Batch Size Variation)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("prefill_attention_performance_batch.png")
    plt.close()


if __name__ == "__main__":
    plot_performance()
