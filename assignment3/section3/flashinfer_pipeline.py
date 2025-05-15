from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import torch
import flashinfer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
#  Project utilities (local module)
# ---------------------------------------------------------------------------
# helper.py must live one directory above this file
sys.path.append(str(Path(__file__).resolve().parent.parent))
from helper import WeightManager, extract_model_weights  # noqa: E402


# ---------------------------------------------------------------------------
#  Low-level data structures: paged KV-cache & per-request view
# ---------------------------------------------------------------------------
class DistKVPool:
    """Global *paged* KV-cache ("HND" = head-page-dim layout)."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        capacity: int,
        page_size: int,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.page_size = page_size

        # Simple free-list allocator ------------------------------------------------
        self._free_pages: set[int] = set(range(capacity))

        # Backing storage tensors: (L, N, H, P, D) where
        #   L = num transformer layers
        #   N = total pages in the pool
        kv_shape = (
            num_layers,
            capacity,
            num_kv_heads,
            page_size,
            head_dim,
        )
        self.k_datas = torch.empty(kv_shape, dtype=torch.float16, device="cuda")
        self.v_datas = torch.empty_like(self.k_datas)

    # Free-list helpers -----------------------------------------------------------
    @property
    def num_free_pages(self) -> int:  # noqa: D401  (short property)
        """Number of unallocated pages left in the pool."""
        return len(self._free_pages)

    def alloc_page(self) -> int:
        """Pop a page index off the free list (O(1))."""
        return self._free_pages.pop()

    def free_page(self, idx: int) -> None:
        """Return *idx* back to the pool."""
        assert idx not in self._free_pages, "double-free detected"
        self._free_pages.add(idx)


class DistKVCache:
    """Light-weight *view* of a request's KV pages (no real storage)."""

    def __init__(self, pool: DistKVPool):
        self._pool = pool
        self._indices: list[int] = []  # page indices owned by this request
        self._seqlen: int = 0  # total tokens stored so far
        self.page_size = pool.page_size

    # Convenience properties -----------------------------------------------------
    @property
    def seqlen(self) -> int:
        return self._seqlen

    @property
    def indices(self) -> list[int]:
        return self._indices

    @property
    def last_page_offset(self) -> int:
        """Number of tokens already present in the *last* page (0-based)."""
        if self._seqlen == 0:
            return 0
        remainder = self._seqlen % self.page_size
        return self.page_size if remainder == 0 else remainder

    # Allocation / release -------------------------------------------------------
    def allocate_tokens(self, num_tokens: int) -> None:
        """Grow the cache so it can hold *num_tokens* additional tokens."""
        assert num_tokens > 0, "must allocate a positive number of tokens"

        # Tokens that still fit into the *current* (possibly partial) page --------
        room_in_last = (
            self.page_size - self.last_page_offset
        ) % self.page_size  # 0 when last page is full

        remaining = max(0, num_tokens - room_in_last)
        pages_needed = (remaining + self.page_size - 1) // self.page_size

        for _ in range(pages_needed):
            self._indices.append(self._pool.alloc_page())

        self._seqlen += num_tokens

    def release(self) -> None:
        """Return all pages back to the global pool (when request finishes)."""
        for idx in self._indices:
            self._pool.free_page(idx)
        self._indices.clear()
        self._seqlen = 0


# ---------------------------------------------------------------------------
#  Helpers to convert a *list* of DistKVCache into FlashInfer ragged metadata
# ---------------------------------------------------------------------------


def build_kv_metadata(kvs: List[DistKVCache]):
    """Return (indptr, indices, last_page_len) - all torch.cuda tensors."""
    kv_indptr: List[int] = [0]
    kv_indices: List[int] = []
    kv_last_page_len: List[int] = []

    for kv in kvs:
        # Append all page indices from this KV cache
        kv_indices.extend(kv.indices)
        # Update indptr with cumulative count of pages
        kv_indptr.append(kv_indptr[-1] + len(kv.indices))
        # Append the length of the last page for this request
        kv_last_page_len.append(kv.last_page_offset)

    device = "cuda"
    return (
        torch.tensor(kv_indptr, dtype=torch.int32, device=device),
        torch.tensor(kv_indices, dtype=torch.int32, device=device),
        torch.tensor(kv_last_page_len, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
#  Simple *request* wrapper (prompt + generation buffer)
# ---------------------------------------------------------------------------
class Request:
    def __init__(self, req_id: int, prompt_ids: torch.Tensor, target_len: int):
        self.request_id = req_id
        self.prompt_token_ids = prompt_ids  # (prompt_len,)
        self.output_length = target_len
        # History buffer (prompt + generated tokens will be appended here)
        self.output_token_ids = prompt_ids.clone()

    # Convenience --------------------------------------------------------------
    @property
    def prompt_length(self) -> int:
        return self.prompt_token_ids.size(0)

    @property
    def current_length(self) -> int:
        return self.output_token_ids.size(0)


# ---------------------------------------------------------------------------
#  Generation *engine*
# ---------------------------------------------------------------------------
class Engine:
    """A minimal Llama-3-8B engine using FlashInfer for attention."""

    # ---------------------------------------------------------------------
    #  Initialisation
    # ---------------------------------------------------------------------
    def __init__(self, profile_time=False) -> None:
        # ---- model hyper-parameters --------------------------------------
        self.weight_path = "/model/Meta-Llama-3-8B-Instruct"
        self.head_dim = 128
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.layers = 32
        
        # Profiling flag
        self.profile_time = profile_time

        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_path)

        # ---- load weights -------------------------------------------------
        wm = WeightManager()
        wm.load_from_safe_tensor(self.weight_path)
        self.weights = extract_model_weights(wm.weight_map, self.layers)

        # ---- global paged KV-cache ---------------------------------------
        self.page_size = 16
        self.max_pages = 20_000  # total pages in the pool (across *all* layers)
        self.pool = DistKVPool(
            num_layers=self.layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            capacity=self.max_pages,
            page_size=self.page_size,
        )

        # Mapping: request-id -> DistKVCache
        self.kv_cache_map: Dict[int, DistKVCache] = {}

        # FlashInfer workspace (single allocation for the whole run)
        workspace_bytes = 128 << 20  # 128 MiB
        self._fi_workspace = torch.empty(
            workspace_bytes, dtype=torch.uint8, device="cuda"
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._fi_workspace, "HND"
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._fi_workspace, "HND", use_tensor_cores=True
        )
        
        # Operation timing dictionary
        self.last_op_times = {}

    # Enable/disable profiling
    def set_profiling(self, enabled: bool) -> None:
        """Enable or disable time profiling."""
        self.profile_time = enabled
        if enabled:
            self.last_op_times = {}
        
    # Helper method to create CUDA events for timing if profiling is enabled
    def _maybe_create_events(self):
        """Create and return CUDA events if profiling is enabled, otherwise return None, None."""
        if self.profile_time:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            return start_event, end_event
        return None, None
        
    # Helper method to record start event if profiling is enabled
    def _maybe_record_start(self, start_event):
        """Record start event if profiling is enabled."""
        if self.profile_time and start_event is not None:
            start_event.record()
            
    # Helper method to record end event and calculate time if profiling is enabled
    def _maybe_record_end(self, start_event, end_event, time_key):
        """Record end event and store elapsed time if profiling is enabled."""
        if self.profile_time and start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            self.last_op_times[time_key] = start_event.elapsed_time(end_event) / 1000.0

    # ---------------------------------------------------------------------
    #  One *step* (mixed prefill + decode) over an *arbitrary* request batch
    # ---------------------------------------------------------------------
    def run(self, requests: List[Request], num_decode_req: int = 0):
        """Run *one* transformer step for ``requests``.

        Parameters
        ----------
        requests : List[Request]
            Full list of requests to be processed this step.
        num_decode_req : int, default=0
            Number of *decode* requests (the **first** N in *requests*).
            Those will feed only their **last** token; the rest are prefills.
        """
        # Reset timing dictionary if profiling is enabled
        if self.profile_time:
            self.last_op_times = {}
        
        with torch.inference_mode():
            # ----------------------------------------------------------------
            # 1) Build ragged *input* tensor and its CSR *indptr*
            # ----------------------------------------------------------------
            start_event, end_event = self._maybe_create_events()
            self._maybe_record_start(start_event)
            
            pieces: List[torch.Tensor] = []
            indptr: List[int] = [0]

            for idx, req in enumerate(requests):
                if idx < num_decode_req:  # decode - feed only *last* token
                    pieces.append(req.output_token_ids[-1:])
                    indptr.append(indptr[-1] + 1)
                else:  # prefill - feed *whole* prompt
                    pieces.append(req.prompt_token_ids)
                    indptr.append(indptr[-1] + req.prompt_length)

            input_tensor = torch.cat(pieces).to("cuda")
            indptr_tensor = torch.tensor(indptr, dtype=torch.int32, device="cuda")
            
            self._maybe_record_end(start_event, end_event, "input_preparation")

            # ----------------------------------------------------------------
            # 2) Create KV cache for prefill requests in kv_cache_map
            # ----------------------------------------------------------------
            start_event, end_event = self._maybe_create_events()
            self._maybe_record_start(start_event)
            
            if num_decode_req == 0:
                for req in requests:
                    # Create new KV cache for this request if it doesn't exist already
                    if req.request_id not in self.kv_cache_map:
                        self.kv_cache_map[req.request_id] = DistKVCache(self.pool)

            # Store current sequence lengths before allocation
            seq_lens_before = [self.kv_cache_map[r.request_id].seqlen for r in requests]
            seq_lens_before_t = torch.tensor(
                seq_lens_before, dtype=torch.int32, device="cuda"
            )

            # ----------------------------------------------------------------
            # 3) Reserve allocate pages for all requests if needed using allocate_tokens function
            # ----------------------------------------------------------------

            for req in requests:
                if num_decode_req == 0:
                    self.kv_cache_map[req.request_id].allocate_tokens(req.prompt_length)
                else:
                    self.kv_cache_map[req.request_id].allocate_tokens(1)

            seq_lens_after = [self.kv_cache_map[r.request_id].seqlen for r in requests]
            seq_lens_after_t = torch.tensor(
                seq_lens_after, dtype=torch.int32, device="cuda"
            )

            # Build paged-KV metadata **after** the append -------------------
            kv_indptr, kv_indices, kv_last_page_len = build_kv_metadata(
                [self.kv_cache_map[r.request_id] for r in requests]
            )
            
            self._maybe_record_end(start_event, end_event, "kv_cache_allocation")

            rope_theta = 500000.0

            # ----------------------------------------------------------------
            # 4) Plan FlashInfer execution for batch
            # ----------------------------------------------------------------
            start_event, end_event = self._maybe_create_events()
            self._maybe_record_start(start_event)
            
            if num_decode_req == 0:
                # plan prefill wrapper
                self.prefill_wrapper.plan(
                    qo_indptr=indptr_tensor[num_decode_req:],
                    paged_kv_indptr=kv_indptr,
                    paged_kv_indices=kv_indices,
                    paged_kv_last_page_len=kv_last_page_len,
                    num_qo_heads=self.num_qo_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim_qk=self.head_dim,
                    pos_encoding_mode="ROPE_LLAMA",
                    page_size=self.page_size,
                    rope_theta=rope_theta,
                )
            else:
                # plan decode wrapper
                self.decode_wrapper.plan(
                    indptr=kv_indptr,
                    indices=kv_indices,
                    last_page_len=kv_last_page_len,
                    num_qo_heads=self.num_qo_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    pos_encoding_mode="ROPE_LLAMA",
                    page_size=self.page_size,
                    rope_theta=rope_theta,
                )
            
            self._maybe_record_end(start_event, end_event, "flashinfer_planning")

            # ----------------------------------------------------------------
            # 5) Forward pass through all *transformer* layers
            # ----------------------------------------------------------------
            start_event, end_event = self._maybe_create_events()
            self._maybe_record_start(start_event)
            
            hidden = self.weights["embedding"][input_tensor]
            
            self._maybe_record_end(start_event, end_event, "embedding_lookup")

            layer_times = {}
            
            for layer in range(self.layers):
                if self.profile_time:
                    layer_start = torch.cuda.Event(enable_timing=True)
                    layer_end = torch.cuda.Event(enable_timing=True)
                    layer_start.record()
                    
                    # === Self-attention sub-layer ==================================
                    attn_start = torch.cuda.Event(enable_timing=True)
                    attn_end = torch.cuda.Event(enable_timing=True)
                    attn_start.record()
                    
                    # QKV projection
                    qkv_start = torch.cuda.Event(enable_timing=True)
                    qkv_end = torch.cuda.Event(enable_timing=True)
                    qkv_start.record()
                
                # === Self-attention sub-layer ==================================
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_attn_in = (hidden / rms).to(torch.float16) * self.weights[
                    "layernormAttn_weight"
                ][layer]

                k = ln_attn_in.matmul(
                    self.weights["self_attn_k_proj_weight"][layer].T
                ).view(-1, self.num_kv_heads, self.head_dim)
                v = ln_attn_in.matmul(
                    self.weights["self_attn_v_proj_weight"][layer].T
                ).view(-1, self.num_kv_heads, self.head_dim)
                q = ln_attn_in.matmul(
                    self.weights["self_attn_q_proj_weight"][layer].T
                ).view(-1, self.num_qo_heads, self.head_dim)
                
                if self.profile_time:
                    qkv_end.record()
                    qkv_end.synchronize()
                    qkv_proj_time = qkv_start.elapsed_time(qkv_end) / 1000.0
                    
                    # ---- Rotary positional embedding ---------------------------
                    rope_start = torch.cuda.Event(enable_timing=True)
                    rope_end = torch.cuda.Event(enable_timing=True)
                    rope_start.record()
                
                # ---- Rotary positional embedding ---------------------------
                flashinfer.apply_rope_inplace(
                    q, k, indptr_tensor, offsets=seq_lens_before_t, rope_theta=rope_theta
                )
                
                if self.profile_time:
                    rope_end.record()
                    rope_end.synchronize()
                    rope_time = rope_start.elapsed_time(rope_end) / 1000.0

                    # ---- Append new tokens to *paged* KV-cache ------------------
                    kv_append_start = torch.cuda.Event(enable_timing=True)
                    kv_append_end = torch.cuda.Event(enable_timing=True)
                    kv_append_start.record()
                
                # ---- Append new tokens to *paged* KV-cache ------------------
                # Get batch indices and positions for appending to KV cache
                batch_indices, positions = flashinfer.get_batch_indices_positions(
                    indptr_tensor, seq_lens=seq_lens_after_t, nnz=q.size(0)
                )

                # Append new KV tokens to paged cache
                flashinfer.append_paged_kv_cache(
                    append_key=k,
                    append_value=v,
                    paged_kv_cache=(self.pool.k_datas[layer], self.pool.v_datas[layer]),
                    kv_indices=kv_indices,
                    kv_indptr=kv_indptr,
                    kv_last_page_len=kv_last_page_len,
                    batch_indices=batch_indices,
                    positions=positions,
                    kv_layout="HND",
                )
                
                if self.profile_time:
                    kv_append_end.record()
                    kv_append_end.synchronize()
                    kv_append_time = kv_append_start.elapsed_time(kv_append_end) / 1000.0

                    # ---- Attention itself --------------------------------------
                    attn_op_start = torch.cuda.Event(enable_timing=True)
                    attn_op_end = torch.cuda.Event(enable_timing=True)
                    attn_op_start.record()
                
                # ---- Attention itself --------------------------------------
                # run prefill and decode wrappers
                attn_out = None
                if num_decode_req == 0:
                    attn_out = self.prefill_wrapper.run(
                        q, (self.pool.k_datas[layer], self.pool.v_datas[layer])
                    )
                else:
                    attn_out = self.decode_wrapper.run(
                        q, (self.pool.k_datas[layer], self.pool.v_datas[layer])
                    )
                
                attn_out = attn_out.reshape(attn_out.size(0), -1)
                
                if self.profile_time:
                    attn_op_end.record()
                    attn_op_end.synchronize()
                    attn_op_time = attn_op_start.elapsed_time(attn_op_end) / 1000.0
                    
                    # Residual connection
                    o_proj_start = torch.cuda.Event(enable_timing=True)
                    o_proj_end = torch.cuda.Event(enable_timing=True)
                    o_proj_start.record()
                
                # Residual connection
                hidden = (
                    attn_out.matmul(self.weights["o_proj_weight"][layer].T) + hidden
                )
                
                if self.profile_time:
                    o_proj_end.record()
                    o_proj_end.synchronize()
                    o_proj_time = o_proj_start.elapsed_time(o_proj_end) / 1000.0
                    
                    attn_end.record()
                    attn_end.synchronize()
                    attn_total_time = attn_start.elapsed_time(attn_end) / 1000.0

                    # === FFN sub-layer ==========================================
                    ffn_start = torch.cuda.Event(enable_timing=True)
                    ffn_end = torch.cuda.Event(enable_timing=True)
                    ffn_start.record()
                
                # === FFN sub-layer ==========================================
                rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
                ln_ffn_in = (hidden / rms).to(torch.float16) * self.weights[
                    "layernormFFN_weight"
                ][layer]

                up = ln_ffn_in.matmul(self.weights["up_proj_weight"][layer].T)
                gate = ln_ffn_in.matmul(self.weights["gate_proj_weight"][layer].T)
                hidden = (up * torch.nn.functional.silu(gate)).matmul(
                    self.weights["down_proj_weight"][layer].T
                ) + hidden
                
                if self.profile_time:
                    ffn_end.record()
                    ffn_end.synchronize()
                    ffn_time = ffn_start.elapsed_time(ffn_end) / 1000.0
                    
                    # Total layer time
                    layer_end.record()
                    layer_end.synchronize()
                    layer_total_time = layer_start.elapsed_time(layer_end) / 1000.0
                    
                    # Store timing info for this layer
                    layer_times[f"layer_{layer}"] = {
                        "total": layer_total_time,
                        "attention": {
                            "total": attn_total_time,
                            "qkv_proj": qkv_proj_time,
                            "rope": rope_time,
                            "kv_append": kv_append_time,
                            "attention_op": attn_op_time,
                            "o_proj": o_proj_time
                        },
                        "ffn": ffn_time
                    }

            # Store average times across all layers if profiling is enabled
            if self.profile_time and layer_times:
                num_layers = self.layers
                self.last_op_times["transformer_layers"] = {
                    "layer_time": sum(layer_times[f"layer_{i}"]["total"] for i in range(num_layers)),
                    "attention_time": sum(layer_times[f"layer_{i}"]["attention"]["total"] for i in range(num_layers)),
                    "ffn_time": sum(layer_times[f"layer_{i}"]["ffn"] for i in range(num_layers)),
                    "detailed_layers": layer_times
                }

            # ----------------------------------------------------------------
            # 6) Final language-model head ----------------------------------
            start_event, end_event = self._maybe_create_events()
            self._maybe_record_start(start_event)
            
            rms = torch.sqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
            logits = (
                (hidden / rms).to(torch.float16)
                * self.weights["model_layernorm_weight"]
            ).matmul(self.weights["lm_head_weight"].T)

            sample_ids = torch.argmax(logits, dim=-1)

            # Extract *new* token for each request (last token of each row)
            last_token_indices = (indptr_tensor[1:] - 1).long()
            
            self._maybe_record_end(start_event, end_event, "lm_head")
            
            return sample_ids[last_token_indices].cpu()

    # ---------------------------------------------------------------------
    #  Full batched *generation* loop (prefill + iterative decode)
    # ---------------------------------------------------------------------
    def generate_batched(self, prompts: List[str], rounds: int = 20):
        print(">>> starting batched generation ({} rounds)".format(rounds))

        # Build *Request* objects ------------------------------------------------
        requests: List[Request] = []
        for idx, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
            requests.append(Request(idx, prompt_ids, rounds))

        # ---- 1) Prefill pass ---------------------------------------------------
        prefill_outputs = self.run(requests, num_decode_req=0)
        print("prefill pass finished - appending first generated token …")

        for i in range(len(requests)):
            new_tok = prefill_outputs[i].unsqueeze(0)
            requests[i].output_token_ids = torch.cat(
                [requests[i].output_token_ids, new_tok], dim=0
            )

        # You do not need to support adding new request on the fly for this assignment, but if you want to, you can uncomment the following lines
        # requests.append(Request(999, self.tokenizer("Today is", return_tensors="pt").input_ids[0], rounds))
        # # ---- 1.5) Prefill pass for the new request --------------------------
        # prefill_outputs = self.run(requests, num_decode_req=len(requests) - 1)
        # for i in range(len(requests) - 1):
        #     new_tok = prefill_outputs[i].unsqueeze(0)
        #     requests[i].output_token_ids = torch.cat(
        #         [requests[i].output_token_ids, new_tok], dim=0
        #     )

        # ---- 2) Iterative decode passes ---------------------------------------
        for _ in range(rounds - 1):
            decode_outputs = self.run(requests, num_decode_req=len(requests))
            for i in range(len(requests)):
                new_tok = decode_outputs[i].unsqueeze(0)
                requests[i].output_token_ids = torch.cat(
                    [requests[i].output_token_ids, new_tok], dim=0
                )

        # ---- 3) Decode back to text and return -------------------------------
        return [
            self.tokenizer.decode(r.output_token_ids, skip_special_tokens=True)
            for r in requests
        ]

    def clean_kv_cache(self):
        for req in self.kv_cache_map:
            self.kv_cache_map[req].release()
        
        self.kv_cache_map = {}


import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import math
from typing import Dict, List, Tuple


def profile_decode_length():
    """
    Profile performance with varying decode lengths.
    - Batch size: 128
    - Prefill length: 1024
    - Decode lengths: 2^5 to 2^10
    """
    print("Profiling with varying decode lengths...")
    
    # Configuration
    batch_size = 128
    prefill_length = 1024
    decode_lengths = [2**i for i in range(5, 11)]  # 32, 64, 128, 256, 512, 1024
    
    # Initialize engine with profiling enabled
    engine = Engine(profile_time=True)
    
    # Generate input prompts
    base_prompt = "Hello " * (prefill_length // 5)  # Approximate length
    prompts = [base_prompt for _ in range(batch_size)]
    
    results = []
    
    for decode_length in decode_lengths:
        print(f"Testing decode length: {decode_length}")
        
        # Measure prefill time
        start_prefill = time.time()
        requests = []
        for idx, prompt in enumerate(prompts):
            prompt_ids = engine.tokenizer(prompt, return_tensors="pt").input_ids[0]
            requests.append(Request(idx, prompt_ids, decode_length))
        
        prefill_outputs = engine.run(requests, num_decode_req=0)
        for i in range(len(requests)):
            new_tok = prefill_outputs[i].unsqueeze(0)
            requests[i].output_token_ids = torch.cat(
                [requests[i].output_token_ids, new_tok], dim=0
            )
        end_prefill = time.time()
        prefill_time = end_prefill - start_prefill
        
        # Collect prefill operation timings
        prefill_op_times = engine.last_op_times.copy() if hasattr(engine, 'last_op_times') else {}
        
        # Measure decode time
        start_decode = time.time()
        for decode_idx in range(decode_length - 1):
            decode_outputs = engine.run(requests, num_decode_req=len(requests))
            for i in range(len(requests)):
                new_tok = decode_outputs[i].unsqueeze(0)
                requests[i].output_token_ids = torch.cat(
                    [requests[i].output_token_ids, new_tok], dim=0
                )
                
            # Store operation times from the last decode iteration
            if decode_idx == decode_length - 2:
                last_decode_op_times = engine.last_op_times.copy() if hasattr(engine, 'last_op_times') else {}
                
        end_decode = time.time()
        decode_time = end_decode - start_decode
        
        # Find the operation that takes the longest time in the last decode cycle
        longest_op = None
        longest_time = 0
        
        # Find operations with the longest time in the last decode cycle
        if last_decode_op_times:
            # Handle the special case of transformer layers
            if 'transformer_layers' in last_decode_op_times:
                layer_times = last_decode_op_times['transformer_layers']
                if 'detailed_layers' in layer_times:
                    # Find the longest operation across all layers
                    for layer, layer_data in layer_times['detailed_layers'].items():
                        # Check attention operations
                        for op, op_time in layer_data['attention'].items():
                            # Skip total time
                            if op == "total":
                                continue

                            if op_time > longest_time:
                                longest_time = op_time
                                longest_op = f"attention.{op}"
                        
                        # Check FFN operation
                        if layer_data['ffn'] > longest_time:
                            longest_time = layer_data['ffn']
                            longest_op = "ffn"
            
            # For non-transformer layer operations
            for op, op_time in last_decode_op_times.items():
                if op != 'transformer_layers' and isinstance(op_time, (int, float)) and op_time > longest_time:
                    longest_time = op_time
                    longest_op = op
        
        total_time = prefill_time + decode_time
        
        results.append({
            "decode_length": decode_length,
            "log_decode_length": math.log2(decode_length),
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": total_time,
            "prefill_op_times": prefill_op_times,
            "last_decode_op_times": last_decode_op_times,
            "longest_op": longest_op,
            "longest_op_time": longest_time
        })

        engine.clean_kv_cache()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    x = [r["log_decode_length"] for r in results]
    plt.plot(x, [r["prefill_time"] for r in results], 'o-', label='Prefill Time')
    plt.plot(x, [r["decode_time"] for r in results], 'o-', label='Decode Time')
    plt.plot(x, [r["total_time"] for r in results], 'o-', label='Total Time')
    plt.xlabel('Log2(Decode Length)')
    plt.ylabel('Time (seconds)')
    plt.title('Performance vs Decode Length (Batch Size=128, Prefill Length=1024)')
    plt.legend()
    plt.grid(True)
    plt.savefig('decode_length_performance.png')
    
    # Plot longest operation per decode length
    plt.figure(figsize=(12, 6))
    ops = {}
    for r in results:
        if 'last_decode_op_times' in r and 'transformer_layers' in r['last_decode_op_times']:
            layer_data = r['last_decode_op_times']['transformer_layers']
            
            # Extract key operations
            if 'attention_time' in layer_data:
                ops.setdefault('attention_time', []).append(layer_data['attention_time'])
            if 'ffn_time' in layer_data:
                ops.setdefault('ffn_time', []).append(layer_data['ffn_time'])
            if 'layer_time' in layer_data:
                ops.setdefault('layer_time', []).append(layer_data['layer_time'])
            
    # Plot each operation
    for op_name, times in ops.items():
        if len(times) == len(x):  # Ensure we have data for all decode lengths
            plt.plot(x, times, 'o-', label=op_name)
    
    plt.xlabel('Log2(Decode Length)')
    plt.ylabel('Time (seconds)')
    plt.title('Last Decode Cycle - Key Operations')
    plt.legend()
    plt.grid(True)
    plt.savefig('decode_length_operations.png')
    
    # Create a detailed plot of operation time per decode length
    plt.figure(figsize=(14, 8))
    
    # Extract all operation times from the transformer layers
    # Instead of using just the first layer as representative, average across all layers
    detailed_ops = {}
    
    for r_idx, r in enumerate(results):
        decode_len = r["decode_length"]
        if 'last_decode_op_times' in r and 'transformer_layers' in r['last_decode_op_times']:
            layer_times = r['last_decode_op_times']['transformer_layers']
            
            # Process all layers and calculate averages
            if 'detailed_layers' in layer_times:
                # Initialize counters for averaging
                layer_op_sums = {}
                layer_op_counts = {}
                
                # Sum up times across all layers
                for layer_name, layer_data in layer_times['detailed_layers'].items():
                    # Process attention operations
                    if 'attention' in layer_data:
                        for op_name, op_time in layer_data['attention'].items():
                            if op_name != 'total':  # Skip total to avoid redundancy
                                op_key = f"attention.{op_name}"
                                layer_op_sums[op_key] = layer_op_sums.get(op_key, 0) + op_time
                                layer_op_counts[op_key] = layer_op_counts.get(op_key, 0) + 1
                    
                    # Process FFN operation
                    if 'ffn' in layer_data:
                        layer_op_sums['ffn'] = layer_op_sums.get('ffn', 0) + layer_data['ffn']
                        layer_op_counts['ffn'] = layer_op_counts.get('ffn', 0) + 1
                
                # Calculate averages and store in detailed_ops
                for op_key, op_sum in layer_op_sums.items():
                    count = layer_op_counts.get(op_key, 1)  # Avoid division by zero
                    detailed_ops.setdefault(op_key, [None] * len(decode_lengths))
                    detailed_ops[op_key][r_idx] = op_sum
            
            # Add high-level operations
            for op_name, op_time in r['last_decode_op_times'].items():
                if op_name != 'transformer_layers' and isinstance(op_time, (int, float)):
                    detailed_ops.setdefault(op_name, [None] * len(decode_lengths))
                    detailed_ops[op_name][r_idx] = op_time
    
    # Sort operations by category for better visualization
    def get_sort_key(op_name):
        # Define the order of attention operations
        attention_order = {
            'attention.qkv_proj': 0,
            'attention.rope': 1,
            'attention.kv_append': 2,
            'attention.attention_op': 3,
            'attention.o_proj': 4
        }
        
        # If it's an attention operation, sort by the predefined order
        if op_name in attention_order:
            return (0, attention_order[op_name])
        
        # Sort other operations by their category
        for idx, key in enumerate(['ffn', 'input', 'kv', 'embedding', 'lm_head', 'flashinfer']):
            if key in op_name.lower():
                return (1, idx)
        
        # Anything else goes at the end
        return (2, 0)
    
    # Define colors for different operation types
    color_map = {
        'attention.qkv_proj': 'royalblue',
        'attention.rope': 'cornflowerblue',
        'attention.kv_append': 'steelblue',
        'attention.attention_op': 'darkblue',
        'attention.o_proj': 'mediumblue',
        'ffn': 'green',
        'input': 'red',
        'kv': 'purple',
        'embedding': 'orange',
        'lm_head': 'brown',
        'flashinfer': 'cyan'
    }
    
    # Assign colors to operations
    op_colors = {}
    for op in detailed_ops.keys():
        # First check for exact matches
        if op in color_map:
            op_colors[op] = color_map[op]
        else:
            # Then check for category matches
            color_assigned = False
            for key, color in color_map.items():
                # Skip attention.* keys for substring matching
                if key.startswith('attention.'):
                    continue
                if key in op.lower():
                    op_colors[op] = color
                    color_assigned = True
                    break
            # Default color if no match
            if not color_assigned:
                op_colors[op] = 'gray'
    
    sorted_ops = sorted(detailed_ops.keys(), key=get_sort_key)
    
    # Plot each operation
    for op_name in sorted_ops:
        op_times = detailed_ops[op_name]
        if all(x is not None for x in op_times):  # Only plot if we have data for all decode lengths
            plt.plot(x, op_times, 'o-', label=op_name, color=op_colors.get(op_name, 'gray'))
    
    plt.xlabel('Log2(Decode Length)')
    plt.ylabel('Time (seconds)')
    plt.title('Detailed Operation Times per Decode Length')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('decode_length_detailed_operations.png')
    
    # Bar chart of operations at longest decode length
    plt.figure(figsize=(12, 8))
    
    # Get the last result (longest decode length)
    last_result = results[-1]
    op_times = {}
    
    if 'last_decode_op_times' in last_result and 'transformer_layers' in last_result['last_decode_op_times']:
        layer_times = last_result['last_decode_op_times']['transformer_layers']
        
        # Find the first layer in detailed_layers
        if 'detailed_layers' in layer_times:
            # Sum across all layers
            op_times_sum = {}
            
            # Iterate through all layers
            for layer_key, layer_data in layer_times['detailed_layers'].items():
                # Add attention operations
                if 'attention' in layer_data:
                    for op_name, op_time in layer_data['attention'].items():
                        if op_name != 'total':
                            op_times_sum[f"attention.{op_name}"] = op_times_sum.get(f"attention.{op_name}", 0) + op_time
                
                # Add FFN operation
                if 'ffn' in layer_data:
                    op_times_sum['ffn'] = op_times_sum.get('ffn', 0) + layer_data['ffn']
            
            # Add summed operations to op_times
            for op_name, op_time in op_times_sum.items():
                op_times[op_name] = op_time
        
        # Add high-level operations
        for op_name, op_time in last_result['last_decode_op_times'].items():
            if op_name != 'transformer_layers' and isinstance(op_time, (int, float)):
                op_times[op_name] = op_time
    
    # Plot bar chart of operation times
    if op_times:
        # Sort by time (descending)
        sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)
        names, times = zip(*sorted_ops)
        
        # Create bar colors using the same color scheme as the line plot
        bar_colors = []
        for name in names:
            # Check for exact matches first
            if name in color_map:
                bar_colors.append(color_map[name])
            else:
                # Then check for category matches
                color_found = False
                for key, color in color_map.items():
                    # Skip attention.* keys for substring matching
                    if key.startswith('attention.'):
                        continue
                    if key in name.lower():
                        bar_colors.append(color)
                        color_found = True
                        break
                # Default color if no match
                if not color_found:
                    bar_colors.append('gray')
        
        plt.barh(names, times, color=bar_colors)
        plt.xlabel('Time (seconds)')
        plt.title(f'Operation Times at Decode Length = {last_result["decode_length"]}')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.savefig('longest_decode_operations_bar.png')
    
    # Save results to CSV and JSON
    with open('decode_length_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['decode_length', 'log_decode_length', 'prefill_time', 'decode_time', 'total_time', 'longest_op', 'longest_op_time'])
        for r in results:
            writer.writerow([r['decode_length'], r['log_decode_length'], r['prefill_time'], r['decode_time'], r['total_time'], r['longest_op'], r['longest_op_time']])
    
    # Save detailed results to JSON
    import json
    with open('decode_length_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze bottleneck
    prefill_avg = sum(r["prefill_time"] for r in results) / len(results)
    decode_avg = sum(r["decode_time"] for r in results) / len(results)
    
    if prefill_avg > decode_avg:
        bottleneck = "Prefill phase"
    else:
        bottleneck = "Decode phase"
    
    # Find most common longest operation
    op_counter = {}
    for r in results:
        op = r.get("longest_op", "unknown")
        op_counter[op] = op_counter.get(op, 0) + 1
    
    most_common_bottleneck = max(op_counter.items(), key=lambda x: x[1])[0] if op_counter else "unknown"
    
    print(f"Decode length profiling completed.")
    print(f"Key bottleneck: {bottleneck}")
    print(f"Most common bottleneck operation in last decode cycle: {most_common_bottleneck}")
    print(f"Created detailed operation timing plots: decode_length_detailed_operations.png and longest_decode_operations_bar.png")


def profile_prefill_length():
    """
    Profile performance with varying prefill lengths.
    - Batch size: 1
    - Prefill lengths: 2^8 to 2^16
    - Decode length: Fixed (e.g., 20)
    """
    print("Profiling with varying prefill lengths...")
    
    # Configuration
    batch_size = 1
    prefill_lengths = [2**i for i in range(8, 17)]  # 256 to 65536
    decode_length = 20  # Fixed decode length
    
    # Initialize engine with profiling enabled
    engine = Engine(profile_time=True)
    
    results = []
    
    for prefill_length in prefill_lengths:
        print(f"Testing prefill length: {prefill_length}")
        
        # Generate input prompt
        base_prompt = "Hello " * (prefill_length // 5)  # Approximate length
        prompts = [base_prompt for _ in range(batch_size)]
        
        # Measure prefill time with operation breakdown
        start_prefill = time.time()
        requests = []
        for idx, prompt in enumerate(prompts):
            prompt_ids = engine.tokenizer(prompt, return_tensors="pt").input_ids[0]
            requests.append(Request(idx, prompt_ids, decode_length))
        
        prefill_outputs = engine.run(requests, num_decode_req=0)
        for i in range(len(requests)):
            new_tok = prefill_outputs[i].unsqueeze(0)
            requests[i].output_token_ids = torch.cat(
                [requests[i].output_token_ids, new_tok], dim=0
            )
        end_prefill = time.time()
        prefill_time = end_prefill - start_prefill
        
        # Get operation times from engine
        prefill_op_times = engine.last_op_times.copy() if hasattr(engine, 'last_op_times') else {}
        
        # Identify dominant operations in prefill
        dominant_op = None
        dominant_time = 0
        
        if prefill_op_times:
            # Handle transformer layers separately
            if 'transformer_layers' in prefill_op_times:
                layer_times = prefill_op_times['transformer_layers']
                if 'attention_time' in layer_times and layer_times['attention_time'] > dominant_time:
                    dominant_time = layer_times['attention_time']
                    dominant_op = 'attention'
                if 'ffn_time' in layer_times and layer_times['ffn_time'] > dominant_time:
                    dominant_time = layer_times['ffn_time']
                    dominant_op = 'ffn'
            
            # Check other top-level operations
            for op, time_value in prefill_op_times.items():
                if op != 'transformer_layers' and isinstance(time_value, (int, float)) and time_value > dominant_time:
                    dominant_time = time_value
                    dominant_op = op
        
        # Measure decode time (for completeness)
        start_decode = time.time()
        for _ in range(decode_length - 1):
            decode_outputs = engine.run(requests, num_decode_req=len(requests))
            for i in range(len(requests)):
                new_tok = decode_outputs[i].unsqueeze(0)
                requests[i].output_token_ids = torch.cat(
                    [requests[i].output_token_ids, new_tok], dim=0
                )
        end_decode = time.time()
        decode_time = end_decode - start_decode
        
        total_time = prefill_time + decode_time
        
        results.append({
            "prefill_length": prefill_length,
            "log_prefill_length": math.log2(prefill_length),
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": total_time,
            "prefill_op_times": prefill_op_times,
            "dominant_op": dominant_op,
            "dominant_op_time": dominant_time
        })

        engine.clean_kv_cache()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    x = [r["log_prefill_length"] for r in results]
    plt.plot(x, [r["prefill_time"] for r in results], 'o-', label='Prefill Time')
    plt.plot(x, [r["total_time"] for r in results], 'o-', label='Total Time')
    plt.xlabel('Log2(Prefill Length)')
    plt.ylabel('Time (seconds)')
    plt.title('Performance vs Prefill Length (Batch Size=1)')
    plt.legend()
    plt.grid(True)
    plt.savefig('prefill_length_performance.png')
    
    # Plot prefill operation breakdown
    plt.figure(figsize=(12, 6))
    op_times = {}
    
    # Extract operation times
    for r in results:
        if 'prefill_op_times' in r and 'transformer_layers' in r['prefill_op_times']:
            layer_data = r['prefill_op_times']['transformer_layers']
            # Extract key operations
            if 'attention_time' in layer_data:
                op_times.setdefault('Attention', []).append(layer_data['attention_time'])
            if 'ffn_time' in layer_data:
                op_times.setdefault('FFN', []).append(layer_data['ffn_time'])
        
        # Add other top-level operations
        for op, time_value in r.get('prefill_op_times', {}).items():
            if op != 'transformer_layers' and isinstance(time_value, (int, float)):
                op_times.setdefault(op, []).append(time_value)
    
    # Plot each operation type
    for op, times in op_times.items():
        if len(times) == len(x):  # Make sure we have data for all prefill lengths
            plt.plot(x, times, 'o-', label=op)
    
    plt.xlabel('Log2(Prefill Length)')
    plt.ylabel('Time (seconds)')
    plt.title('Prefill Operation Breakdown')
    plt.legend()
    plt.grid(True)
    plt.savefig('prefill_breakdown.png')
    
    # Save results to CSV and JSON
    with open('prefill_length_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prefill_length', 'log_prefill_length', 'prefill_time', 'decode_time', 'total_time', 'dominant_op', 'dominant_op_time'])
        for r in results:
            writer.writerow([r['prefill_length'], r['log_prefill_length'], r['prefill_time'], r['decode_time'], r['total_time'], r['dominant_op'], r['dominant_op_time']])
    
    # Save detailed results to JSON
    import json
    with open('prefill_length_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze dominant operation across different prefill lengths
    op_counter = {}
    for r in results:
        op = r.get("dominant_op", "unknown")
        op_counter[op] = op_counter.get(op, 0) + 1
    
    most_common_op = max(op_counter.items(), key=lambda x: x[1])[0] if op_counter else "unknown"
    
    print("Prefill length profiling completed.")
    print(f"Most common dominant operation across prefill lengths: {most_common_op}")
    
    # Analyze if dominance changes with length
    short_prefill = [r["dominant_op"] for r in results[:len(results)//2]]
    long_prefill = [r["dominant_op"] for r in results[len(results)//2:]]
    
    short_counter = {}
    for op in short_prefill:
        short_counter[op] = short_counter.get(op, 0) + 1
    
    long_counter = {}
    for op in long_prefill:
        long_counter[op] = long_counter.get(op, 0) + 1
    
    short_dominant = max(short_counter.items(), key=lambda x: x[1])[0] if short_counter else "unknown"
    long_dominant = max(long_counter.items(), key=lambda x: x[1])[0] if long_counter else "unknown"
    
    if short_dominant != long_dominant:
        print(f"Dominant operation changes with prefill length:")
        print(f"  Short prefill dominant: {short_dominant}")
        print(f"  Long prefill dominant: {long_dominant}")


def profile_batch_size():
    """
    Profile performance with varying batch sizes.
    - Batch sizes: 2^0 to 2^10
    - Prefill length: 128
    - Decode length: 128
    """
    print("Profiling with varying batch sizes...")
    
    # Configuration
    batch_sizes = [2**i for i in range(0, 11)]  # 1 to 1024
    prefill_length = 128
    decode_length = 128
    
    # Initialize engine with profiling enabled
    engine = Engine(profile_time=False)
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Generate input prompts
        base_prompt = "Hello " * (prefill_length // 5)  # Approximate length
        prompts = [base_prompt for _ in range(batch_size)]
        
        # Measure total time
        start_total = time.time()
        
        # Prefill phase
        requests = []
        for idx, prompt in enumerate(prompts):
            prompt_ids = engine.tokenizer(prompt, return_tensors="pt").input_ids[0]
            requests.append(Request(idx, prompt_ids, decode_length))
        
        prefill_outputs = engine.run(requests, num_decode_req=0)
        prefill_op_times = engine.last_op_times.copy() if hasattr(engine, 'last_op_times') else {}
        
        for i in range(len(requests)):
            new_tok = prefill_outputs[i].unsqueeze(0)
            requests[i].output_token_ids = torch.cat(
                [requests[i].output_token_ids, new_tok], dim=0
            )
        
        # Decode phase
        decode_op_times = []
        for _ in range(decode_length - 1):
            decode_outputs = engine.run(requests, num_decode_req=len(requests))
            decode_op_times.append(engine.last_op_times.copy() if hasattr(engine, 'last_op_times') else {})
            
            for i in range(len(requests)):
                new_tok = decode_outputs[i].unsqueeze(0)
                requests[i].output_token_ids = torch.cat(
                    [requests[i].output_token_ids, new_tok], dim=0
                )
        
        end_total = time.time()
        total_time = end_total - start_total
        
        # Calculate throughput (tokens/second)
        total_tokens = batch_size * (prefill_length + decode_length)
        throughput = total_tokens / total_time
        
        results.append({
            "batch_size": batch_size,
            "log_batch_size": math.log2(batch_size),
            "total_time": total_time,
            "throughput": throughput,
            "prefill_op_times": prefill_op_times,
            "decode_op_times": decode_op_times[-1] if decode_op_times else {}  # Last decode cycle
        })

        engine.clean_kv_cache()
    
    # Find where performance saturates
    # We'll define saturation as the point where doubling batch size gives less than 20% improvement
    saturation_point = None
    saturation_throughput = None
    
    for i in range(1, len(results)):
        prev_throughput = results[i-1]["throughput"]
        curr_throughput = results[i]["throughput"]
        improvement = (curr_throughput - prev_throughput) / prev_throughput
        
        if improvement < 0.5:  # Less than 20% improvement
            saturation_point = results[i]["batch_size"]
            saturation_throughput = curr_throughput
            break
    
    # Plot time results
    plt.figure(figsize=(10, 6))
    x = [r["log_batch_size"] for r in results]
    plt.plot(x, [r["total_time"] for r in results], 'o-')
    if saturation_point:
        saturation_index = batch_sizes.index(saturation_point)
        plt.axvline(x=x[saturation_index], color='r', linestyle='--', 
                   label=f'Saturation at batch={saturation_point}')
    plt.xlabel('Log2(Batch Size)')
    plt.ylabel('Time (seconds)')
    plt.title('End-to-End Time vs Batch Size (Prefill=128, Decode=128)')
    plt.grid(True)
    plt.legend()
    plt.savefig('batch_size_time.png')
    
    # Plot throughput results
    plt.figure(figsize=(10, 6))
    plt.plot(x, [r["throughput"] for r in results], 'o-')
    if saturation_point:
        saturation_index = batch_sizes.index(saturation_point)
        plt.axvline(x=x[saturation_index], color='r', linestyle='--',
                   label=f'Saturation at batch={saturation_point}')
    plt.xlabel('Log2(Batch Size)')
    plt.ylabel('Throughput (tokens/second)')
    plt.title('Throughput vs Batch Size (Prefill=128, Decode=128)')
    plt.grid(True)
    plt.legend()
    plt.savefig('batch_size_throughput.png')
    
    # Save results to CSV and JSON
    with open('batch_size_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch_size', 'log_batch_size', 'total_time', 'throughput'])
        for r in results:
            writer.writerow([r['batch_size'], r['log_batch_size'], r['total_time'], r['throughput']])
    
    # Save detailed results to JSON
    import json
    with open('batch_size_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Batch size profiling completed.")
    if saturation_point:
        print(f"Performance saturates at batch size {saturation_point} with throughput {saturation_throughput:.2f} tokens/second")
    else:
        print("No clear saturation point detected within the tested batch sizes")



# ---------------------------------------------------------------------------
#  Entry-point (debug / standalone execution)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    profile_decode_length()
    profile_prefill_length()
    profile_batch_size()
