Transformers Lab
================
A grab-bag of small assignments exploring how modern LLM inference works and how to tune it for speed. The code focuses on manual implementations of decoder blocks, KV-cache strategies, scheduling, and GPU kernels, mostly using the Llama 3 8B weights stored at `/model/Meta-Llama-3-8B-Instruct`.

Repository layout
-----------------
- `assignment2/`: Hand-rolled Llama decoder with different prefill/generation strategies. Includes:
  - `no_kv.py`: Baseline generation without KV cache, timing written to `no_kv_times.csv`.
  - `different_prefill.py`, `uniform_prefill.py`, `single_batch.py`: Compare batched prefill patterns and KV reuse; CSVs and `prefill_comparison.png` plot wall times vs batch size.
  - `graph1.py`, `graph2.py`: Helpers to visualize the CSV timing outputs.
- `assignment3/section2/`: Benchmarks FlashAttention vs FlashInfer.
  - `flashattention_single_layer.py`: TFlops vs sequence length and vs batch size on CUDA using flash-attn bindings; produces `attention_performance_seq.png` and `attention_performance_batch.png`.
  - `flashattention_decode.py`: Decoding-time comparison of FlashAttention and FlashInfer in a single layer.
- `assignment3/section3/`: FlashInfer pipeline experiment with a paged KV cache.
  - `flashinfer_pipeline.py`: Implements a simple allocator (`DistKVPool`/`DistKVCache`), prompt/generation loop, and uses FlashInfer wrappers to prefill and decode many requests.
  - `helper.py`: Shared tokenizer/weight loading utilities and RoPE.
- `assignment4/Section1/`: Request schedulers that interleave many generations.
  - `continous_*`: Continuous batching scheduler and engine.
  - `chunked_*`: Chunked scheduling variant to improve throughput with limited KV memory.
- `assignment4/Section2/`: Tensor-parallel and collective communication profiling.
  - `transformer-w3l1.py`: Single-GPU reference run of the decoder.
  - `transformer-tp2.py`: Two-way tensor parallel forward pass (uses NCCL init).
  - `profile_allreduce.py`: Measures NCCL allreduce bandwidth vs batch size and plots `profile_allreduce.png`.
- `rms_norm/` and `silu/`: CUDA/C++ and Triton stubs for custom kernels.
- `host_GPU/`: Minimal CUDA playground for copying GPU memory from host.

Dependencies
------------
- Python: `torch`, `transformers`, `safetensors`, `tqdm`, `matplotlib`, `flash-attn`, `flashinfer`, `triton`.
- GPU with CUDA and (for distributed parts) NCCL available.
- Llama weights placed under `/model/Meta-Llama-3-8B-Instruct` (paths are hard-coded in multiple scripts).

How to run
----------
- KV-cache vs prefill experiments (write CSVs/PNGs):  
  `python assignment2/no_kv.py` or `python assignment2/different_prefill.py`
- FlashAttention vs FlashInfer TFlops:  
  `python assignment3/section2/flashattention_single_layer.py`
- FlashInfer pipelined batch decode:  
  `python assignment3/section3/flashinfer_pipeline.py`
- Scheduler experiments:  
  `python assignment4/Section1/continous_main.py` or `python assignment4/Section1/chunked_main.py`
- Tensor-parallel profile:  
  `python assignment4/Section2/profile_allreduce.py`

Notes
-----
- Scripts assume a CUDA device; some require two GPUs for NCCL tests.  
- Many files are educational stubs: see TODOs in `silu/` and `rms_norm/` if extending custom kernels.
