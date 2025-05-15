import torch
import flashinfer
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import flash_attn_interface

def profile_prefill_performance(
    model_sizes: Dict[str, Dict[str, int]],
    batch_size: int,
    p_ranges: Dict[str, List[int]],
    num_warmups: int = 3,
    num_iters: int = 10
):
    # Initialize results storage
    results = {model: {"fa3": [], "flashinfer": []} for model in model_sizes}
    
    for model_name, config in model_sizes.items():
        num_heads = config["num_heads"]
        head_dim = config["head_dim"]
        p_values = p_ranges[model_name]
        
        for p in p_values:
            # Create test inputs
            q = torch.randn(batch_size, p, num_heads, head_dim).half().cuda()
            k = torch.randn(batch_size, p, num_heads, head_dim).half().cuda()
            v = torch.randn(batch_size, p, num_heads, head_dim).half().cuda()
            
            # FlashAttention-3
            with torch.inference_mode():
                for _ in range(num_warmups):
                    _ = flash_attn_interface.flash_attn_func(q, k, v)
                torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(num_iters):
                    _ = flash_attn_interface.flash_attn_func(q, k, v)
                end.record()
                torch.cuda.synchronize()
                time_fa3 = start.elapsed_time(end) / num_iters
                
            # FlashInfer
            with torch.inference_mode():
                workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
                
                for _ in range(num_warmups):
                    wrapper.run(q, (k, v))
                torch.cuda.synchronize()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(num_iters):
                    wrapper.run(q, (k, v))
                end.record()
                torch.cuda.synchronize()
                time_flashinfer = start.elapsed_time(end) / num_iters
            
            # Calculate TFLOPS
            tflops_fa3 = (4 * p * num_heads * head_dim * p) / (time_fa3 * 1e-3) / 1e12
            tflops_flashinfer = (4 * p * num_heads * head_dim * p) / (time_flashinfer * 1e-3) / 1e12
            
            results[model_name]["fa3"].append(tflops_fa3)
            results[model_name]["flashinfer"].append(tflops_flashinfer)
    
    return results

def plot_performance(results: Dict, x_label: str, y_label: str):
    models = list(results.keys())
    fig, axs = plt.subplots(1, len(models), figsize=(15, 5))
    
    for idx, model in enumerate(models):
        ax = axs[idx]
        p_values = list(results[model]["fa3"].keys())
        
        ax.plot(p_values, results[model]["fa3"], label="FlashAttention-3", 
                linestyle="--", marker="o")
        ax.plot(p_values, results[model]["flashinfer"], label="FlashInfer", 
                linestyle="-", marker="s")
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{model} Performance")
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("attention_performance.png")

# Configuration
model_configs = {
    "llama2-7B": {"num_heads": 32, "head_dim": 128},
    "llama3-8B": {"num_heads": 32, "head_dim": 128},
    "llama3-70B": {"num_heads": 64, "head_dim": 128}
}

p_ranges = {
    "llama2-7B": [2**i for i in range(7, 13)],
    "llama3-8B": [2**i for i in range(7, 16)],
    "llama3-70B": [2**i for i in range(7, 16)]
}

# Run profiling
prefill_results = profile_prefill_performance(model_configs, 1, p_ranges)

# Plot results
plot_performance(prefill_results, r"$\log_2(p)$", "Compute Utilization (TFlops)")
