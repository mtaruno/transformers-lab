import torch
from transformers import AutoTokenizer
import sys

sys.path.append("../")  # Adjust the path to import the helper module
from helper import WeightManager, apply_rope, extract_model_weights
import time
import csv


class Engine:
    """
    A class to manage the generation engine.
    """

    def __init__(self):
        ########################################
        # Model Configuration Parameters
        ########################################
        self.weight_path = "/model/Meta-Llama-3-8B-Instruct"
        self.head_dim = 128  # Dimensionality of each attention head
        self.num_qo_heads = 32  # Total number of query/output heads
        self.num_kv_heads = 8  # Total number of key/value heads
        self.layers = 32  # Number of transformer layers

        # Load the tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/model/Meta-Llama-3-8B-Instruct"
        )

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
        input_tensor = input_ids.to(device="cuda")  # (batch_size, seq_len)
        hidden_state = self.weights["embedding"][
            input_tensor
        ]  # (batch_size, seq_len, hidden_dim)
        batch_size, seq_len = hidden_state.shape[:2]

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
            offset = 0 if prefill else self.kv_cache[current_layer]["k"].shape[1]

            # Apply RoPE to each sequence in the batch
            for i in range(batch_size):
                apply_rope(q[i], output=q[i], head_dim=self.head_dim, offset=offset)
                apply_rope(k[i], output=k[i], head_dim=self.head_dim, offset=offset)

            scale = 1.0 / (self.head_dim**0.5)
            group_size = self.num_qo_heads // self.num_kv_heads

            # Reshape for attention computation
            sub_q = q.view(batch_size, seq_len, self.num_qo_heads, self.head_dim)
            sub_k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            sub_v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

            # Update KV cache
            if prefill:
                self.kv_cache[current_layer]["k"] = sub_k
                self.kv_cache[current_layer]["v"] = sub_v
            else:
                self.kv_cache[current_layer]["k"] = torch.cat(
                    [self.kv_cache[current_layer]["k"], sub_k], dim=1
                )
                self.kv_cache[current_layer]["v"] = torch.cat(
                    [self.kv_cache[current_layer]["v"], sub_v], dim=1
                )

            # Use the cached KV values
            sub_k = self.kv_cache[current_layer][
                "k"
            ]  # (batch_size, total_seq_len, num_kv_heads, head_dim)
            sub_v = self.kv_cache[current_layer][
                "v"
            ]  # (batch_size, total_seq_len, num_kv_heads, head_dim)

            n_q = seq_len  # Number of query tokens (current batch)
            n_k = sub_k.shape[
                1
            ]  # Number of key tokens (includes history from KV cache)

            # Repeat KV heads to match QO heads
            sub_k = sub_k.repeat_interleave(
                group_size, dim=2
            )  # (batch_size, total_seq_len, num_qo_heads, head_dim)
            sub_v = sub_v.repeat_interleave(
                group_size, dim=2
            )  # (batch_size, total_seq_len, num_qo_heads, head_dim)

            # Reshape for attention computation
            sub_q = sub_q.transpose(
                1, 2
            )  # (batch_size, num_qo_heads, seq_len, head_dim)
            sub_k = sub_k.transpose(
                1, 2
            )  # (batch_size, num_qo_heads, total_seq_len, head_dim)
            sub_v = sub_v.transpose(
                1, 2
            )  # (batch_size, num_qo_heads, total_seq_len, head_dim)

            scores = (
                torch.matmul(sub_q, sub_k.transpose(-2, -1)) * scale
            )  # (batch_size, num_qo_heads, seq_len, total_seq_len)

            # Create proper causal mask for KV caching
            if prefill:
                # In prefill mode, standard causal mask (lower triangular)
                causal_mask = torch.tril(
                    torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)
                )

                scores = scores.masked_fill(
                    ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            attn_weights = torch.softmax(
                scores, dim=-1
            )  # (batch_size, num_qo_heads, seq_len, total_seq_len)
            attn_output = torch.matmul(
                attn_weights, sub_v
            )  # (batch_size, num_qo_heads, seq_len, head_dim)

            # Reshape attention output
            attn_output = attn_output.transpose(
                1, 2
            )  # (batch_size, seq_len, num_qo_heads, head_dim)
            attn_output = attn_output.reshape(
                batch_size, seq_len, -1
            )  # (batch_size, seq_len, hidden_dim)

            # Project output and add residual connection
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
        logits = model_output.matmul(
            self.weights["lm_head_weight"].t()
        )  # (batch_size, seq_len, vocab_size)

        # Get the last token's logits for each sequence in the batch
        sample_output = torch.argmax(logits[:, -1, :], dim=-1)  # (batch_size,)
        # make sure the output has dimention (batch_size, )
        sample_output = sample_output.unsqueeze(1)
        return sample_output

    def generate_batched(self, input_string, rounds=20):
        input_ids_list = self.tokenizer(
            input_string, return_tensors="pt", padding=False
        ).input_ids

        output_ids_list = input_ids_list
        output_ids_list = output_ids_list.to(device="cuda")

        new_token = self.run(output_ids_list)
        output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        for round in range(rounds - 1):
            # print(f"Round {round}")
            new_token = self.run(output_ids_list[:, -1:], prefill=False)
            output_ids_list = torch.cat((output_ids_list, new_token), dim=1)

        output_text = self.tokenizer.batch_decode(
            output_ids_list, skip_special_tokens=True
        )
        return output_text


########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hello " * 510
    output_length = 128
    engine = Engine()

    times = {}

    for batch_size_pow in range(0, 1):
        batch_size = 2**batch_size_pow

        input_string_list = [input_string for _ in range(batch_size)]
        start = time.time()
        output_text = engine.generate_batched(input_string_list, rounds=output_length)
        end = time.time()
        times[batch_size] = end - start
        print(f"Batch size: {batch_size}, Time taken: {end - start} seconds")

    print(times)

    with open("uniform_prefill.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "time"])
        for batch_size, time in times.items():
            writer.writerow([batch_size, time])
