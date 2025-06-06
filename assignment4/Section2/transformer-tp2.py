import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer
from tqdm import tqdm

from helper import WeightManager, apply_rope, extract_model_weights

def init_distributed(rank, world_size):
    """Initialize distributed process group for tensor parallelism"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # change if needed
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def load_model_weights():
    """Load and return all model weights"""
    # === Model Parameters ===
    # Path to the pretrained model weights
    weight_path = "/model/Meta-Llama-3-8B-Instruct"

    # === Load Weights ===
    # Load tokenizer and model weights
    # Tokenizer helps convert strings to token IDs
    # WeightManager loads the weights from safetensors

    # Load tokenizer for text encoding/decoding
    tokenizer = AutoTokenizer.from_pretrained("/model/Meta-Llama-3-8B-Instruct")
    weight_manager = WeightManager()
    weight_manager.load_from_safe_tensor(weight_path)
    weights = extract_model_weights(weight_manager.weight_map, layers)

    # Extract and name all necessary model weights for easier access
    model_weights = {
        "embedding": weights["embedding"],
        "layernormAttn_weight": weights["layernormAttn_weight"],
        "self_attn_q_proj_weight": weights["self_attn_q_proj_weight"],
        "self_attn_k_proj_weight": weights["self_attn_k_proj_weight"],
        "self_attn_v_proj_weight": weights["self_attn_v_proj_weight"],
        "o_proj_weight": weights["o_proj_weight"],
        "layernormFFN_weight": weights["layernormFFN_weight"],
        "up_proj_weight": weights["up_proj_weight"],
        "gate_proj_weight": weights["gate_proj_weight"],
        "down_proj_weight": weights["down_proj_weight"],
        "model_layernorm_weight": weights["model_layernorm_weight"],
        "lm_head_weight": weights["lm_head_weight"]
    }
    
    return tokenizer, model_weights

# === Model Parameters ===
# LLaMA-3 8B has 32 transformer layers
layers = 32
num_qo_heads = 32   # Number of query/output heads per transformer block
head_dim = 128       # Dimension per attention head
num_kv_heads = 8     # Number of key/value heads
hidden_dim = 4096    # Total hidden dimension of transformer
intermediate_dim = 14336  # Intermediate dimension for MLP (FFN)

# === Run One Iteration with Tensor Parallelism ===
def run_one_iteration(input_ids, rank, world_size, model_weights):
    # Extract model weights
    embedding = model_weights["embedding"]
    layernormAttn_weight = model_weights["layernormAttn_weight"]
    self_attn_q_proj_weight = model_weights["self_attn_q_proj_weight"]
    self_attn_k_proj_weight = model_weights["self_attn_k_proj_weight"]
    self_attn_v_proj_weight = model_weights["self_attn_v_proj_weight"]
    o_proj_weight = model_weights["o_proj_weight"]
    layernormFFN_weight = model_weights["layernormFFN_weight"]
    up_proj_weight = model_weights["up_proj_weight"]
    gate_proj_weight = model_weights["gate_proj_weight"]
    down_proj_weight = model_weights["down_proj_weight"]
    model_layernorm_weight = model_weights["model_layernorm_weight"]
    lm_head_weight = model_weights["lm_head_weight"]
    
    # Convert input token IDs to tensor and move to current device
    input_tensor = torch.tensor(input_ids, dtype=torch.int32, device=f"cuda:{rank}")
    hidden_state = embedding.to(f'cuda:{rank}')[input_tensor]  # Embed input tokens

    # Define local dimensions for tensor parallelism
    local_q_heads = num_qo_heads // world_size
    local_kv_heads = num_kv_heads // world_size
    local_hidden_dim = hidden_dim // world_size
    local_intermediate_dim = intermediate_dim // world_size

    for layer in range(layers):
        # --- Attention Block ---
        rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = hidden_state / rms
        x = normalized_x * layernormAttn_weight[layer]
        
        # --- Part 1 Implement the attention block ---

        # TODO: generate the q, k, v vectors
        # assuming that the weights of q_proj, k_proj, v_proj are split in a column-wise (head) manner
        # hint: use rank, local_q_heads, local_kv_heads, head_dim to figure out the correct slice
        # hint: to debug, compare the intermediate outputs with the original implementation in transformer-w3l1.py
        
        # Column-wise split for q, k, v projections (split by heads)
        q_start = rank * local_q_heads * head_dim
        q_end = (rank + 1) * local_q_heads * head_dim
        kv_start = rank * local_kv_heads * head_dim
        kv_end = (rank + 1) * local_kv_heads * head_dim
        
        # Transpose weights first, then slice (weights are stored as output_dim x input_dim)
        q = x.matmul(self_attn_q_proj_weight[layer].t()[:, q_start:q_end])
        k = x.matmul(self_attn_k_proj_weight[layer].t()[:, kv_start:kv_end])
        v = x.matmul(self_attn_v_proj_weight[layer].t()[:, kv_start:kv_end]) 

        # Apply rotary position embeddings
        apply_rope(q, output=q, head_dim=head_dim, offset=0)
        apply_rope(k, output=k, head_dim=head_dim, offset=0)

        # Reshape into multi-head format and replicate KV for grouped attention
        sub_q = q.view(-1, local_q_heads, head_dim)
        sub_k = k.view(-1, local_kv_heads, head_dim).repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
        sub_v = v.view(-1, local_kv_heads, head_dim).repeat_interleave(num_qo_heads // num_kv_heads, dim=1)

        # Transpose for batch matmul in attention computation
        sub_q_t = sub_q.permute(1, 0, 2)
        sub_k_t = sub_k.permute(1, 0, 2)
        scores = torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))

        # Causal mask to prevent attending to future tokens
        causal_mask = torch.tril(torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        v_t = sub_v.permute(1, 0, 2)
        attn_output = torch.matmul(attn_weights, v_t).permute(1, 0, 2).reshape(-1, local_q_heads * head_dim)

        # TODO: generate the o_proj_local vector
        # assuming that the weights of o_proj are split in a row-wise manner
        # hint: use rank, local_hidden_dim to figure out the correct slice
        # Row-wise split for output projection
        # Each GPU handles partial attention output, slice input dimension of weight matrix
        attn_start = rank * local_q_heads * head_dim
        attn_end = (rank + 1) * local_q_heads * head_dim
        o_proj_local = attn_output.matmul(o_proj_weight[layer].t()[attn_start:attn_end, :])
        
        # TODO: perform the all-reduce operation
        # hint: use dist.all_reduce
        # All-reduce to sum results from all ranks
        dist.all_reduce(o_proj_local, op=dist.ReduceOp.SUM) 
        
        o_proj_residual = o_proj_local + hidden_state  # Add residual

        # --- Part 2 Implement the feedforward block ---
        
        rms = torch.sqrt(torch.mean(o_proj_residual ** 2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = o_proj_residual / rms
        ffn_input = normalized_x.to(torch.float16) * layernormFFN_weight[layer]

        # TODO: generate the up_local and gate_local vectors
        # assuming that the weights of up_proj and gate_proj are split in a column-wise manner
        # hint: use rank, local_intermediate_dim to figure out the correct slice
        # Column-wise split for up and gate projections
        up_start = rank * local_intermediate_dim
        up_end = (rank + 1) * local_intermediate_dim
        
        up_local = ffn_input.matmul(up_proj_weight[layer].t()[:, up_start:up_end])
        gate_local = ffn_input.matmul(gate_proj_weight[layer].t()[:, up_start:up_end]) 

        # SwiGLU activation (SiLU * linear)
        activation_output = up_local * F.silu(gate_local)

        # TODO: generate the down_local vector
        # assuming that the weights of down_proj are split in a row-wise manner
        # Row-wise split for down projection
        # Each GPU handles partial FFN output, slice input dimension of weight matrix
        ffn_start = rank * local_intermediate_dim
        ffn_end = (rank + 1) * local_intermediate_dim
        down_local = activation_output.matmul(down_proj_weight[layer].t()[ffn_start:ffn_end, :])

        # TODO: perform the all-reduce operation
        # All-reduce to sum results from all ranks
        dist.all_reduce(down_local, op=dist.ReduceOp.SUM)

        # Add residual
        hidden_state = down_local + o_proj_residual

    # Final layer norm and projection to vocabulary space
    rms = torch.sqrt(torch.mean(hidden_state ** 2, dim=-1, keepdim=True) + 1e-5)
    normalized_x = hidden_state / rms
    model_output = normalized_x.to(torch.float16) * model_layernorm_weight

    logits = model_output.matmul(lm_head_weight.t())  # Final linear layer to get logits
    return torch.argmax(logits, dim=-1)[-1].item()  # Return the ID of the last token predicted

# === Text Generation ===
def generate(tokenizer, model_weights, rank, world_size):
    input_string = "The University of Michigan is a"
    input_ids = tokenizer.encode(input_string)
    output_ids = input_ids.copy()
    for _ in range(100):
        new_token = run_one_iteration(output_ids, rank, world_size, model_weights)
        output_ids.append(new_token)
    if rank == 0:
        print("\nOutput:", tokenizer.decode(output_ids, skip_special_tokens=True))

def run_worker(rank, world_size, return_dict):
    """Worker function for each GPU process"""
    init_distributed(rank, world_size)
    
    # Load model weights (each process loads independently)
    tokenizer, model_weights = load_model_weights()
    
    # Warm up
    for i in tqdm(range(10), desc=f"Warmup GPU {rank}"):
        generate(tokenizer, model_weights, rank, world_size)
    dist.barrier()
    
    # Timing runs
    start_time = time.time()
    for i in tqdm(range(10), desc=f"Timing GPU {rank}"):
        generate(tokenizer, model_weights, rank, world_size)
    end_time = time.time()
    
    if rank == 0:
        avg_time = (end_time - start_time) / 10
        print(f"Average time taken: {avg_time} seconds")
        return_dict['avg_time'] = avg_time
    
    dist.destroy_process_group()

# === Entry ===
if __name__ == '__main__':
    # TODO: set CUDA_VISIBLE_DEVICES to 2 GPUs before running the code
    # e.g. export CUDA_VISIBLE_DEVICES=2,3
    
    world_size = 2
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run_worker, args=(rank, world_size, return_dict))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    if 'avg_time' in return_dict:
        print(f"\nFinal result - Average time taken: {return_dict['avg_time']} seconds")
