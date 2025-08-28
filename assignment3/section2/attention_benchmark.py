import os
import sys
import time
from pathlib import Path
import math
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import flashinfer
import flash_attn_interface

# Model configurations
MODEL_CONFIGS = {
    "llama2-7B": {
        "path": "/model/Llama-2-7b-hf",
        "head_dim": 128,
        "num_qo_heads": 32,
        "num_kv_heads": 32,
        "layers": 32,
        "prefill_lengths": [2**i for i in range(7, 13)],  # 2^7 to 2^12
    },
    "llama3-8B": {
        "path": "/model/Meta-Llama-3-8B-Instruct",
        "head_dim": 128,
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "layers": 32,
        "prefill_lengths": [2**i for i in range(7, 16)],  # 2^7 to 2^15
    },
    "llama3-70B": {
        "path": "/model/Meta-Llama-3-70B-Instruct",  # Update if path is different
        "head_dim": 128,
        "num_qo_heads": 64,
        "num_kv_heads": 8,
        "layers": 80,
        "prefill_lengths": [2**i for i in range(7, 16)],  # 2^7 to 2^15
    }
}

# Batch sizes for experiments with fixed sequence length
BATCH_SIZES = [2**i for i in range(0, 7)]  # 2^0 to 2^6

# Page sizes for paged attention experiments
PAGE_SIZES = [1, 2, 4, 8, 16]

class AttentionProfiler:
    """
    Profiler for measuring performance of attention mechanisms.
    """
    def __init__(self, model_name: str):
        """
        Initialize the profiler for a specific model.

        Args:
            model_name: Name of the model configuration to use
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")
        
        self.model_config = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        
        # Create workspace for FlashInfer
        workspace_bytes = 256 << 20  # 256 MiB
        self._fi_workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        
        # Using synthetic data only - no need to load real model weights
        self.has_weights = False
        print(f"Initializing profiler for {model_name} with synthetic data")
    
    def _create_synthetic_data(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create synthetic QKV data for benchmarking.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            q, k, v tensors
        """
        head_dim = self.model_config["head_dim"]
        num_qo_heads = self.model_config["num_qo_heads"]
        num_kv_heads = self.model_config["num_kv_heads"]
        
        # Create synthetic data in the shape expected by attention implementations
        q = torch.randn((batch_size, seq_len, num_qo_heads, head_dim), 
                        dtype=torch.float16, device="cuda")
        k = torch.randn((batch_size, seq_len, num_kv_heads, head_dim), 
                        dtype=torch.float16, device="cuda")
        v = torch.randn((batch_size, seq_len, num_kv_heads, head_dim), 
                        dtype=torch.float16, device="cuda")
        
        return q, k, v
    
    def _calculate_flops(self, batch_size: int, seq_len: int) -> float:
        """
        Calculate theoretical FLOPs for an attention operation.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            FLOPs count for the operation
        """
        head_dim = self.model_config["head_dim"]
        num_qo_heads = self.model_config["num_qo_heads"]
        
        # FLOPs for QK^T (bmm) + softmax + attention (BV)
        flops_per_token = 2 * num_qo_heads * seq_len * head_dim  # QK^T
        flops_per_token += 2 * num_qo_heads * seq_len  # softmax
        flops_per_token += 2 * num_qo_heads * seq_len * head_dim  # AV
        
        total_flops = batch_size * seq_len * flops_per_token
        
        return total_flops
    
    def _calculate_memory_bandwidth(self, batch_size: int, seq_len: int, page_size: Optional[int] = None) -> float:
        """
        Calculate theoretical memory bandwidth for an attention operation.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            page_size: Page size for paged attention (if applicable)
            
        Returns:
            Memory bandwidth in bytes
        """
        head_dim = self.model_config["head_dim"]
        num_qo_heads = self.model_config["num_qo_heads"]
        num_kv_heads = self.model_config["num_kv_heads"]
        
        # Size of data types
        dtype_size = 2  # float16 = 2 bytes
        
        # For normal attention, total bytes accessed is (in simplified form):
        # q: B×S×H_q×D
        # k: B×S×H_k×D
        # v: B×S×H_k×D
        # out: B×S×H_q×D
        
        if page_size is None:
            # Regular attention, all KV values read once
            total_bytes = (
                batch_size * seq_len * num_qo_heads * head_dim * dtype_size +  # q
                batch_size * seq_len * num_kv_heads * head_dim * dtype_size +  # k
                batch_size * seq_len * num_kv_heads * head_dim * dtype_size +  # v
                batch_size * seq_len * num_qo_heads * head_dim * dtype_size   # output
            )
        else:
            # Paged attention - each token needs to read whole KV cache
            # Bytes accessed depends on how we organize the paged KV cache
            # Simplified estimate for read bandwidth
            kv_cache_size = seq_len * num_kv_heads * head_dim * dtype_size  # Size in bytes of KV cache
            
            # For decode attention, we need to read the entire KV cache for each token
            # Q/output is just single token per batch
            total_bytes = (
                batch_size * 1 * num_qo_heads * head_dim * dtype_size +  # q (single token per batch)
                kv_cache_size +  # k - whole KV cache
                kv_cache_size +  # v - whole KV cache
                batch_size * 1 * num_qo_heads * head_dim * dtype_size   # output (single token per batch)
            )
        
        return total_bytes
    
    def benchmark_flashattn3_prefill(self, batch_size: int, seq_len: int) -> float:
        """
        Benchmark FlashAttention-3 for prefill attention.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Compute utilization in TFLOPs
        """
        from flash_attn_interface import flash_attn_func
        
        q, k, v = self._create_synthetic_data(batch_size, seq_len)
        
        # Transpose from (B, S, H, D) to (B, S, H*D)
        q_flat = q.reshape(batch_size, seq_len, -1)
        k_flat = k.reshape(batch_size, seq_len, -1)
        v_flat = v.reshape(batch_size, seq_len, -1)
        
        # Transpose again for FlashAttention expected format: (B, S, H*D) -> (B, H, S, D)
        q_flashattn = q_flat.view(batch_size, seq_len, self.model_config["num_qo_heads"], 
                                self.model_config["head_dim"]).transpose(1, 2)
        k_flashattn = k_flat.view(batch_size, seq_len, self.model_config["num_kv_heads"], 
                                self.model_config["head_dim"]).transpose(1, 2)
        v_flashattn = v_flat.view(batch_size, seq_len, self.model_config["num_kv_heads"], 
                                self.model_config["head_dim"]).transpose(1, 2)
        
        # Apply ROPE if needed - simplify for benchmarking
        
        # Warmup
        for _ in range(5):
            _ = flash_attn_func(q_flashattn, k_flashattn, v_flashattn)
        
        # Benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        num_iters = 20
        start_event.record()
        for _ in range(num_iters):
            _ = flash_attn_func(q_flashattn, k_flashattn, v_flashattn)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000 / num_iters  # in seconds
        
        flops = self._calculate_flops(batch_size, seq_len)
        tflops = flops / elapsed_time / 1e12  # convert to TFLOPs
        
        return tflops
    
    def benchmark_flashinfer_prefill(self, batch_size: int, seq_len: int) -> float:
        """
        Benchmark FlashInfer for prefill attention.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Compute utilization in TFLOPs
        """
        q, k, v = self._create_synthetic_data(batch_size, seq_len)
        
        # Prepare indptr tensor for ragged batch handling
        indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        for i in range(batch_size):
            indptr[i + 1] = indptr[i] + seq_len
        
        # Initialize prefill wrapper
        prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            self._fi_workspace, kv_layout="HND"
        )
        
        # Plan the operation
        prefill_wrapper.plan(
            qo_indptr=indptr,
            kv_indptr=indptr,
            num_qo_heads=self.model_config["num_qo_heads"],
            num_kv_heads=self.model_config["num_kv_heads"],
            head_dim_qk=self.model_config["head_dim"],
            pos_encoding_mode="NONE",  # Simplified for benchmarking
            causal=True
        )
        
        # Warmup
        for _ in range(5):
            _ = prefill_wrapper.run(q, k, v)
        
        # Benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        num_iters = 20
        start_event.record()
        for _ in range(num_iters):
            _ = prefill_wrapper.run(q, k, v)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000 / num_iters  # in seconds
        
        flops = self._calculate_flops(batch_size, seq_len)
        tflops = flops / elapsed_time / 1e12  # convert to TFLOPs
        
        return tflops
    
    def benchmark_flashattn3_decode(self, batch_size: int, seq_len: int, page_size: int = 16) -> float:
        """
        Benchmark FlashAttention-3 for decode (paged) attention.
        
        Args:
            batch_size: Batch size
            seq_len: Context length (KV cache size)
            page_size: Page size for paged attention
            
        Returns:
            Memory bandwidth utilization in GB/s
        """
        from flash_attn_interface import flash_attn_with_kvcache
        
        # Create synthetic query (just 1 token per batch)
        head_dim = self.model_config["head_dim"]
        num_qo_heads = self.model_config["num_qo_heads"]
        num_kv_heads = self.model_config["num_kv_heads"]
        
        # Create query tensor (B, H, 1, D)
        q = torch.randn((batch_size, num_qo_heads, 1, head_dim), 
                      dtype=torch.float16, device="cuda")
        
        # Create KV cache tensors (B, H, S, D)
        k_cache = torch.randn((batch_size, num_kv_heads, seq_len, head_dim), 
                            dtype=torch.float16, device="cuda")
        v_cache = torch.randn((batch_size, num_kv_heads, seq_len, head_dim), 
                            dtype=torch.float16, device="cuda")
        
        # Simulate paged attention with FlashAttention
        # Since FlashAttention doesn't directly support paged attention, we'll use normal attention
        # for benchmarking memory bandwidth, which should be comparable
        
        # Warmup
        for _ in range(5):
            _ = flash_attn_with_kvcache(
                q, k_cache, v_cache, 
                cache_seqlens=torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)
            )
        
        # Benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        num_iters = 20
        start_event.record()
        for _ in range(num_iters):
            _ = flash_attn_with_kvcache(
                q, k_cache, v_cache, 
                cache_seqlens=torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)
            )
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000 / num_iters  # in seconds
        
        memory_bytes = self._calculate_memory_bandwidth(batch_size, seq_len, page_size)
        bandwidth_gb_s = memory_bytes / elapsed_time / 1e9  # convert to GB/s
        
        return bandwidth_gb_s
    
    def benchmark_flashinfer_decode(self, batch_size: int, seq_len: int, page_size: int = 16) -> float:
        """
        Benchmark FlashInfer for decode (paged) attention.
        
        Args:
            batch_size: Batch size
            seq_len: Context length (KV cache size)
            page_size: Page size for paged attention
            
        Returns:
            Memory bandwidth utilization in GB/s
        """
        head_dim = self.model_config["head_dim"]
        num_qo_heads = self.model_config["num_qo_heads"]
        num_kv_heads = self.model_config["num_kv_heads"]
        
        # Setup paged KV cache
        # Calculate number of pages needed
        num_pages = (seq_len + page_size - 1) // page_size
        
        # Create paged KV tensors (N, H, P, D) where N is total pages
        # and organize by "HND" layout (head-page-dim)
        num_total_pages = batch_size * num_pages
        k_paged = torch.randn((num_total_pages, num_kv_heads, page_size, head_dim), 
                            dtype=torch.float16, device="cuda")
        v_paged = torch.randn((num_total_pages, num_kv_heads, page_size, head_dim), 
                            dtype=torch.float16, device="cuda")
        
        # Create single query token per batch (B, H, D)
        q = torch.randn((batch_size, num_qo_heads, head_dim), 
                      dtype=torch.float16, device="cuda")
        
        # Setup paged KV cache metadata
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        for i in range(batch_size):
            kv_indptr[i + 1] = kv_indptr[i] + num_pages
        
        # Create sequential page indices
        kv_indices = torch.arange(num_total_pages, dtype=torch.int32, device="cuda")
        
        # Set last page length for each request (if last page is partial)
        last_page_offset = seq_len % page_size
        last_page_len = torch.full(
            (batch_size,), 
            page_size if last_page_offset == 0 else last_page_offset, 
            dtype=torch.int32, device="cuda"
        )
        
        # Initialize decode wrapper
        decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._fi_workspace, "HND", use_tensor_cores=True
        )
        
        # Plan the operation
        decode_wrapper.plan(
            indptr=kv_indptr,
            indices=kv_indices,
            last_page_len=last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            pos_encoding_mode="NONE",  # Simplified for benchmarking
            page_size=page_size,
            rope_theta=10000.0,  # Not used with NONE encoding
        )
        
        # Warmup
        for _ in range(5):
            _ = decode_wrapper.run(q, (k_paged, v_paged))
        
        # Benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        num_iters = 20
        start_event.record()
        for _ in range(num_iters):
            _ = decode_wrapper.run(q, (k_paged, v_paged))
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000 / num_iters  # in seconds
        
        memory_bytes = self._calculate_memory_bandwidth(batch_size, seq_len, page_size)
        bandwidth_gb_s = memory_bytes / elapsed_time / 1e9  # convert to GB/s
        
        return bandwidth_gb_s

def profile_prefill_seq_length():
    """
    Profile prefill attention performance with varying sequence lengths.
    - Batch size = 1
    - Sequence lengths vary by model
    - Plot compute utilization (TFLOPs)
    """
    models = ["llama2-7B", "llama3-8B", "llama3-70B"]
    batch_size = 1
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, model_name in enumerate(models):
        profiler = AttentionProfiler(model_name)
        seq_lengths = MODEL_CONFIGS[model_name]["prefill_lengths"]
        
        # Results for plotting
        log_seq_lengths = [math.log2(seq_len) for seq_len in seq_lengths]
        flash_attn_tflops = []
        flashinfer_tflops = []
        
        print(f"Profiling {model_name} with varying sequence lengths...")
        
        for seq_len in seq_lengths:
            print(f"  Sequence length: {seq_len}")
            
            # Benchmark FlashAttention-3
            flash_attn_tflop = profiler.benchmark_flashattn3_prefill(batch_size, seq_len)
            flash_attn_tflops.append(flash_attn_tflop)
            print(f"    FlashAttention-3: {flash_attn_tflop:.2f} TFLOPs")
            
            # Benchmark FlashInfer
            flashinfer_tflop = profiler.benchmark_flashinfer_prefill(batch_size, seq_len)
            flashinfer_tflops.append(flashinfer_tflop)
            print(f"    FlashInfer: {flashinfer_tflop:.2f} TFLOPs")
        
        # Plot results
        ax = axes[i]
        ax.plot(log_seq_lengths, flash_attn_tflops, 'o-', label='FlashAttention-3')
        ax.plot(log_seq_lengths, flashinfer_tflops, 's--', label='FlashInfer')
        ax.set_title(f"{model_name}")
        ax.set_xlabel('log₂(sequence length)')
        ax.set_ylabel('Compute Utilization (TFLOPs)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('prefill_seq_length_perf.png')
    plt.close()
    
    print("Prefill sequence length profiling completed.")

def profile_prefill_batch_size():
    """
    Profile prefill attention performance with varying batch sizes.
    - Fixed sequence length p = 1024
    - Batch sizes: 2^0 to 2^6
    - Plot compute utilization (TFLOPs)
    """
    models = ["llama2-7B", "llama3-8B", "llama3-70B"]
    seq_length = 1024
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, model_name in enumerate(models):
        profiler = AttentionProfiler(model_name)
        
        # Results for plotting
        log_batch_sizes = [math.log2(bs) for bs in BATCH_SIZES]
        flash_attn_tflops = []
        flashinfer_tflops = []
        
        print(f"Profiling {model_name} with varying batch sizes...")
        
        for batch_size in BATCH_SIZES:
            print(f"  Batch size: {batch_size}")
            
            # Benchmark FlashAttention-3
            flash_attn_tflop = profiler.benchmark_flashattn3_prefill(batch_size, seq_length)
            flash_attn_tflops.append(flash_attn_tflop)
            print(f"    FlashAttention-3: {flash_attn_tflop:.2f} TFLOPs")
            
            # Benchmark FlashInfer
            flashinfer_tflop = profiler.benchmark_flashinfer_prefill(batch_size, seq_length)
            flashinfer_tflops.append(flashinfer_tflop)
            print(f"    FlashInfer: {flashinfer_tflop:.2f} TFLOPs")
        
        # Plot results
        ax = axes[i]
        ax.plot(log_batch_sizes, flash_attn_tflops, 'o-', label='FlashAttention-3')
        ax.plot(log_batch_sizes, flashinfer_tflops, 's--', label='FlashInfer')
        ax.set_title(f"{model_name}")
        ax.set_xlabel('log₂(batch size)')
        ax.set_ylabel('Compute Utilization (TFLOPs)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('prefill_batch_size_perf.png')
    plt.close()
    
    print("Prefill batch size profiling completed.")

def profile_decode_seq_length():
    """
    Profile decode attention performance with varying sequence lengths.
    - Batch size = 1
    - Page size = 16
    - Sequence lengths vary by model
    - Plot memory bandwidth utilization (GB/s)
    """
    models = ["llama2-7B", "llama3-8B", "llama3-70B"]
    batch_size = 1
    page_size = 16
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, model_name in enumerate(models):
        profiler = AttentionProfiler(model_name)
        seq_lengths = MODEL_CONFIGS[model_name]["prefill_lengths"]
        
        # Results for plotting
        log_seq_lengths = [math.log2(seq_len) for seq_len in seq_lengths]
        flash_attn_bw = []
        flashinfer_bw = []
        
        print(f"Profiling {model_name} decode with varying sequence lengths...")
        
        for seq_len in seq_lengths:
            print(f"  Sequence length: {seq_len}")
            
            # Benchmark FlashAttention-3
            flash_attn_bandwidth = profiler.benchmark_flashattn3_decode(batch_size, seq_len, page_size)
            flash_attn_bw.append(flash_attn_bandwidth)
            print(f"    FlashAttention-3: {flash_attn_bandwidth:.2f} GB/s")
            
            # Benchmark FlashInfer
            flashinfer_bandwidth = profiler.benchmark_flashinfer_decode(batch_size, seq_len, page_size)
            flashinfer_bw.append(flashinfer_bandwidth)
            print(f"    FlashInfer: {flashinfer_bandwidth:.2f} GB/s")
        
        # Plot results
        ax = axes[i]
        ax.plot(log_seq_lengths, flash_attn_bw, 'o-', label='FlashAttention-3')
        ax.plot(log_seq_lengths, flashinfer_bw, 's--', label='FlashInfer')
        ax.set_title(f"{model_name}")
        ax.set_xlabel('log₂(sequence length)')
        ax.set_ylabel('Memory Bandwidth (GB/s)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('decode_seq_length_perf.png')
    plt.close()
    
    print("Decode sequence length profiling completed.")

def profile_decode_batch_size():
    """
    Profile decode attention performance with varying batch sizes.
    - Fixed sequence length c = 1024
    - Page size = 16
    - Batch sizes: 2^0 to 2^6
    - Plot memory bandwidth utilization (GB/s)
    """
    models = ["llama2-7B", "llama3-8B", "llama3-70B"]
    seq_length = 1024
    page_size = 16
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, model_name in enumerate(models):
        profiler = AttentionProfiler(model_name)
        
        # Results for plotting
        log_batch_sizes = [math.log2(bs) for bs in BATCH_SIZES]
        flash_attn_bw = []
        flashinfer_bw = []
        
        print(f"Profiling {model_name} decode with varying batch sizes...")
        
        for batch_size in BATCH_SIZES:
            print(f"  Batch size: {batch_size}")
            
            # Benchmark FlashAttention-3
            flash_attn_bandwidth = profiler.benchmark_flashattn3_decode(batch_size, seq_length, page_size)
            flash_attn_bw.append(flash_attn_bandwidth)
            print(f"    FlashAttention-3: {flash_attn_bandwidth:.2f} GB/s")
            
            # Benchmark FlashInfer
            flashinfer_bandwidth = profiler.benchmark_flashinfer_decode(batch_size, seq_length, page_size)
            flashinfer_bw.append(flashinfer_bandwidth)
            print(f"    FlashInfer: {flashinfer_bandwidth:.2f} GB/s")
        
        # Plot results
        ax = axes[i]
        ax.plot(log_batch_sizes, flash_attn_bw, 'o-', label='FlashAttention-3')
        ax.plot(log_batch_sizes, flashinfer_bw, 's--', label='FlashInfer')
        ax.set_title(f"{model_name}")
        ax.set_xlabel('log₂(batch size)')
        ax.set_ylabel('Memory Bandwidth (GB/s)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('decode_batch_size_perf.png')
    plt.close()
    
    print("Decode batch size profiling completed.")

def profile_decode_page_size():
    """
    Profile decode attention performance with varying page sizes.
    - Batch size = 128
    - Sequence length c = 1024
    - Page sizes: 1, 2, 4, 8, 16
    - Plot memory bandwidth utilization (GB/s)
    """
    models = ["llama2-7B", "llama3-8B", "llama3-70B"]
    batch_size = 128
    seq_length = 1024
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, model_name in enumerate(models):
        profiler = AttentionProfiler(model_name)
        
        # Results for plotting
        flash_attn_bw = []
        flashinfer_bw = []
        
        print(f"Profiling {model_name} decode with varying page sizes...")
        
        for page_size in PAGE_SIZES:
            print(f"  Page size: {page_size}")
            
            # Benchmark FlashAttention-3
            flash_attn_bandwidth = profiler.benchmark_flashattn3_decode(batch_size, seq_length, page_size)
            flash_attn_bw.append(flash_attn_bandwidth)
            print(f"    FlashAttention-3: {flash_attn_bandwidth:.2f} GB/s")
            
            # Benchmark FlashInfer
            flashinfer_bandwidth = profiler.benchmark_flashinfer_decode(batch_size, seq_length, page_size)
            flashinfer_bw.append(flashinfer_bandwidth)
            print(f"    FlashInfer: {flashinfer_bandwidth:.2f} GB/s")
        
        # Plot results
        ax = axes[i]
        ax.plot(PAGE_SIZES, flash_attn_bw, 'o-', label='FlashAttention-3')
        ax.plot(PAGE_SIZES, flashinfer_bw, 's--', label='FlashInfer')
        ax.set_title(f"{model_name}")
        ax.set_xlabel('Page Size')
        ax.set_ylabel('Memory Bandwidth (GB/s)')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('decode_page_size_perf.png')
    plt.close()
    
    print("Decode page size profiling completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile attention performance")
    parser.add_argument('--all', action='store_true', help='Run all profiling functions')
    parser.add_argument('--prefill-seq', action='store_true', help='Profile prefill with varying sequence lengths')
    parser.add_argument('--prefill-batch', action='store_true', help='Profile prefill with varying batch sizes')
    parser.add_argument('--decode-seq', action='store_true', help='Profile decode with varying sequence lengths')
    parser.add_argument('--decode-batch', action='store_true', help='Profile decode with varying batch sizes')
    parser.add_argument('--decode-page', action='store_true', help='Profile decode with varying page sizes')
    parser.add_argument('--models', nargs='+', choices=MODEL_CONFIGS.keys(), default=list(MODEL_CONFIGS.keys()),
                        help='Models to profile (default: all)')
    
    args = parser.parse_args()
    
    # If no specific profiling is requested, run all
    if not any([args.all, args.prefill_seq, args.prefill_batch, args.decode_seq, args.decode_batch, args.decode_page]):
        args.all = True
    
    if args.all or args.prefill_seq:
        profile_prefill_seq_length()
    
    if args.all or args.prefill_batch:
        profile_prefill_batch_size()
    
    if args.all or args.decode_seq:
        profile_decode_seq_length()
    
    if args.all or args.decode_batch:
        profile_decode_batch_size()
    
    if args.all or args.decode_page:
        profile_decode_page_size() 