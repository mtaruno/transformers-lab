import torch
from transformers import AutoTokenizer
import sys
sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights


class Engine:
    """
    A class to manage the generation engine.
    """
    def __init__(self):
        ########################################
        # Model Configuration Parameters
        ########################################
        self.weight_path = "/model/Meta-Llama-3-8B-Instruct"
        self.head_dim = 128         # Dimensionality of each attention head
        self.num_qo_heads = 32      # Total number of query/output heads
        self.num_kv_heads = 8       # Total number of key/value heads
        self.layers = 32            # Number of transformer layers

        # Load the tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained("/model/Meta-Llama-3-8B-Instruct")

        # Initialize and load model weights using the helper module
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(self.weight_path)

        # Extract all required model weights from the weight_map
        self.weights = extract_model_weights(weight_manager.weight_map, self.layers)
        
        self.kv_cache = {}
    
    def run(self, input_ids, prefill=True):
        ########################################
        # Already implemented
        ########################################
        input_tensor = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
        hidden_state = self.weights["embedding"][input_tensor]
        seq_len = hidden_state.shape[0]

        # Initialize KV cache if it's a prefill run
        if prefill:
            self.kv_cache = {}
            for layer in range(self.layers):
                self.kv_cache[layer] = {"k": None, "v": None}
            self.seq_pos = 0  # Reset sequence position in prefill mode

        # Current token positions for correct masking
        current_pos = torch.arange(self.seq_pos, self.seq_pos + seq_len, device="cuda")
        self.seq_pos += seq_len  # Update for next call

        for current_layer in range(self.layers):
            # --- Self-Attention Block ---
            rms = torch.sqrt(torch.mean(hidden_state**2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = hidden_state / rms
            x = (
                normalized_x.to(torch.float16)
                * self.weights["layernormAttn_weight"][current_layer]
            )

            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())

            # Apply RoPE to query and key using the helper function
            offset = 0 if prefill else self.kv_cache[current_layer]["k"].shape[0]
            apply_rope(q, output=q, head_dim=self.head_dim, offset=offset)
            apply_rope(k, output=k, head_dim=self.head_dim, offset=offset)

            scale = 1.0 / (self.head_dim**0.5)
            group_size = self.num_qo_heads // self.num_kv_heads

            sub_q = q.view(
                -1, self.num_qo_heads, self.head_dim
            )  # (seq_len, num_qo_heads, head_dim)
            sub_k = k.view(
                -1, self.num_kv_heads, self.head_dim
            )  # (seq_len, num_kv_heads, head_dim)
            sub_v = v.view(
                -1, self.num_kv_heads, self.head_dim
            )  # (seq_len, num_kv_heads, head_dim)

            # Update KV cache
            if prefill:
                self.kv_cache[current_layer]["k"] = sub_k
                self.kv_cache[current_layer]["v"] = sub_v
            else:
                self.kv_cache[current_layer]["k"] = torch.cat(
                    [self.kv_cache[current_layer]["k"], sub_k], dim=0
                )
                self.kv_cache[current_layer]["v"] = torch.cat(
                    [self.kv_cache[current_layer]["v"], sub_v], dim=0
                )

            # Use the cached KV values
            sub_k = self.kv_cache[current_layer]["k"]
            sub_v = self.kv_cache[current_layer]["v"]

            n_q = sub_q.shape[0]  # Number of query tokens (current batch)
            n_k = sub_k.shape[
                0
            ]  # Number of key tokens (includes history from KV cache)

            sub_k = sub_k.repeat_interleave(
                group_size, dim=1
            )  # (seq_len, num_qo_heads, head_dim)
            sub_v = sub_v.repeat_interleave(
                group_size, dim=1
            )  # (seq_len, num_qo_heads, head_dim)

            sub_q_t = sub_q.permute(1, 0, 2)  # (num_qo_heads, seq_len, head_dim)
            sub_k_t = sub_k.permute(1, 0, 2)  # (num_qo_heads, seq_len, head_dim)

            scores = (
                torch.matmul(sub_q_t, sub_k_t.transpose(-2, -1)) * scale
            )  # (num_qo_heads, seq_len, seq_len)

            # Create proper causal mask for KV caching
            # When using KV cache, each query position should attend only to cached keys
            # up to and including its corresponding position
            if prefill:
                # In prefill mode, standard causal mask (lower triangular)
                causal_mask = torch.tril(
                    torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)
                )

                scores = scores.masked_fill(
                    ~causal_mask.unsqueeze(0), float("-inf")
                )  # (num_qo_heads, seq_len, seq_len)

            attn_weights = torch.softmax(scores, dim=-1)

            v_t = sub_v.permute(1, 0, 2)  # (num_qo_heads, seq_len, head_dim)
            attn_output = torch.matmul(
                attn_weights, v_t
            )  # (num_qo_heads, seq_len, head_dim)
            attn_output = attn_output.permute(
                1, 0, 2
            )  # (seq_len, num_qo_heads, head_dim)

            attn_output = attn_output.reshape(
                -1, self.num_qo_heads * self.head_dim
            )  # (seq_len, num_qo_heads * head_dim)
            prefill_output = (
                attn_output.matmul(self.weights["o_proj_weight"][current_layer].t())
                + hidden_state
            )

            # --- Feed-Forward Network (FFN) Block ---
            rms = torch.sqrt(torch.mean(prefill_output**2, dim=-1, keepdim=True) + 1e-5)
            normalized_x = prefill_output / rms
            layernormFFN_output = (
                normalized_x.to(torch.float16)
                * self.weights["layernormFFN_weight"][current_layer]
            )

            up_proj_output = layernormFFN_output.matmul(
                self.weights["up_proj_weight"][current_layer].t()
            )
            gate_proj_output = layernormFFN_output.matmul(
                self.weights["gate_proj_weight"][current_layer].t()
            )

            activation_output = up_proj_output * torch.nn.functional.silu(
                gate_proj_output
            )
            hidden_state = (
                activation_output.matmul(
                    self.weights["down_proj_weight"][current_layer].t()
                )
                + prefill_output
            )

        # --- Final Layer Normalization and Output Projection ---
        rms = torch.sqrt(torch.mean(hidden_state**2, dim=-1, keepdim=True) + 1e-5)
        normalized_x = hidden_state / rms
        model_output = (
            normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]
        )
        logits = model_output.matmul(self.weights["lm_head_weight"].t())

        sample_output = torch.argmax(logits, dim=1)
        return sample_output[-1].item()
    
    def generate_batched(self, input_string, rounds=20):
        input_ids_list = self.tokenizer(input_string, return_tensors="pt", padding=False).input_ids
        print("Input String:", input_string)

        print("Token IDs:", input_ids_list)
        output_ids_list = input_ids_list  

        new_token = self.run(output_ids_list)
        print("New Token Shape:", new_token.shape)
        output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        for round in range(rounds - 1):
            print(f"Round {round}")
            new_token = self.run(output_ids_list[:, -1:], prefill=False)
            output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        output_text = self.tokenizer.batch_decode(output_ids_list, skip_special_tokens=True)
        return output_text

########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hi, who are you?"
    input_string_list = [input_string for _ in range(10)]
    another_input_string = "Hi, how are you?"
    for _ in range(10):
        input_string_list.append(another_input_string)
    engine = Engine()
    output_text = engine.generate_batched(input_string_list, rounds=20)
    print("Generated Text:", output_text)