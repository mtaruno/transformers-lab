from itertools import accumulate
from numpy import concat, require
import torch
from transformers import AutoTokenizer
import sys
import time
import csv

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

        # input_ids is a list of input_ids
        # input_ids[i] is the input_ids for the i-th sequence
        batch_size = len(input_ids)
        seq_len = [input_ids[i].shape[0] for i in range(batch_size)]

        batch_offsets = torch.tensor([0] + list(accumulate(seq_len)), device="cuda")
        input_tensor = torch.cat(input_ids, dim=0).to(
            device="cuda"
        )

        hidden_state = self.weights["embedding"][
            input_tensor
        ]  # (batch_size, seq_len, hidden_dim)

        # Initialize KV cache if it's a prefill run
        if prefill:
            self.kv_cache = {}
            for layer in range(self.layers):
                # each batch has its own kv cache
                self.kv_cache[layer] = {
                    "k": None,
                    "v": None,
                }
            self.seq_pos = [
                0 for _ in range(batch_size)
            ]  # Reset sequence position in prefill mode

        # Current token positions for correct masking
        for i in range(batch_size):
            self.seq_pos[i] += seq_len[i]

        for current_layer in range(self.layers):
            normalized_x = torch.empty_like(hidden_state)

            for i in range(batch_size):
                # --- Self-Attention Block ---
                rms = torch.sqrt(
                    torch.mean(
                        hidden_state[batch_offsets[i] : batch_offsets[i + 1]] ** 2,
                        dim=-1,
                        keepdim=True,
                    )
                    + 1e-5
                )
                normalized_x[batch_offsets[i] : batch_offsets[i + 1]] = (
                    hidden_state[batch_offsets[i] : batch_offsets[i + 1]] / rms
                )

            x = (
                normalized_x.to(torch.float16)
                * self.weights["layernormAttn_weight"][current_layer]
            )

            k = x.matmul(self.weights["self_attn_k_proj_weight"][current_layer].t())
            v = x.matmul(self.weights["self_attn_v_proj_weight"][current_layer].t())
            q = x.matmul(self.weights["self_attn_q_proj_weight"][current_layer].t())

            # Apply RoPE to each sequence in the batch
            for i in range(batch_size):
                offset = 0 if prefill else self.kv_cache[current_layer]["k"][i].shape[0]
                apply_rope(
                    q[batch_offsets[i] : batch_offsets[i + 1]],
                    output=q[batch_offsets[i] : batch_offsets[i + 1]],
                    head_dim=self.head_dim,
                    offset=offset,
                )
                apply_rope(
                    k[batch_offsets[i] : batch_offsets[i + 1]],
                    output=k[batch_offsets[i] : batch_offsets[i + 1]],
                    head_dim=self.head_dim,
                    offset=offset,
                )

            scale = 1.0 / (self.head_dim**0.5)
            group_size = self.num_qo_heads // self.num_kv_heads

            # Reshape for attention computation
            all_sub_q = q.view(q.shape[0], self.num_qo_heads, self.head_dim)
            all_sub_k = k.view(k.shape[0], self.num_kv_heads, self.head_dim)
            all_sub_v = v.view(v.shape[0], self.num_kv_heads, self.head_dim)

            # Update KV cache       
            if prefill:
                self.kv_cache[current_layer]["k"] = all_sub_k
                self.kv_cache[current_layer]["v"] = all_sub_v
            else:
                self.kv_cache[current_layer]["k"] = torch.cat(
                    [
                        self.kv_cache[current_layer]["k"],
                        all_sub_k,
                    ],
                    dim=0,
                )
                self.kv_cache[current_layer]["v"] = torch.cat(
                    [
                        self.kv_cache[current_layer]["v"],
                        all_sub_v,
                    ],
                    dim=0,
                )

            all_sub_k = self.kv_cache[current_layer]["k"]
            all_sub_v = self.kv_cache[current_layer]["v"]

            for i in range(batch_size):

                sub_q = all_sub_q[batch_offsets[i] : batch_offsets[i + 1]]
                sub_k = all_sub_k[batch_offsets[i] : batch_offsets[i + 1]]
                sub_v = all_sub_v[batch_offsets[i] : batch_offsets[i + 1]]

                # Number of query tokens (current batch)
                n_q = seq_len[i]
                # Number of key tokens (includes history from KV cache)
                n_k = sub_k.shape[0]

                # Repeat KV heads to match QO heads
                sub_k = sub_k.repeat_interleave(
                    group_size, dim=1
                )  # (total_seq_len, num_qo_heads, head_dim)
                sub_v = sub_v.repeat_interleave(
                    group_size, dim=1
                )  # (total_seq_len, num_qo_heads, head_dim)

                sub_q_t = sub_q.permute(1, 0, 2)  # (num_qo_heads, seq_len, head_dim)
                sub_k_t = sub_k.permute(1, 0, 2)  # (num_qo_heads, seq_len, head_dim)

                # Compute attention scores
                scores = (
                    sub_q_t @ sub_k_t.transpose(-2, -1)
                ) * scale  # (num_qo_heads, seq_len, seq_len)

                # Create proper causal mask for KV caching
                if prefill:
                    # In prefill mode, standard causal mask (lower triangular)
                    causal_mask = torch.tril(
                        torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device)
                    )

                    scores = scores.masked_fill(
                        ~causal_mask.unsqueeze(0), float("-inf")
                    )

                attn_weights = torch.softmax(
                    scores, dim=-1
                )  # (batch_size, num_qo_heads, seq_len, total_seq_len)

                sub_v_t = sub_v.permute(1, 0, 2)  # (num_qo_heads, seq_len, head_dim)

                attn_output = torch.matmul(
                    attn_weights, sub_v_t
                )  # (num_qo_heads, seq_len, head_dim)

                attn_output = attn_output.permute(1, 0, 2)

                attn_output = attn_output.reshape(
                    -1, self.num_qo_heads * self.head_dim
                )  # (seq_len, num_qo_heads * head_dim)

                # Project output and add residual connection
                prefill_output = (
                    attn_output.matmul(self.weights["o_proj_weight"][current_layer].t())
                    + hidden_state[batch_offsets[i] : batch_offsets[i + 1]]
                )

                # --- Feed-Forward Network (FFN) Block ---
                rms = torch.sqrt(
                    torch.mean(prefill_output**2, dim=-1, keepdim=True) + 1e-5
                )
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
                hidden_state[batch_offsets[i] : batch_offsets[i + 1]] = (
                    activation_output.matmul(
                        self.weights["down_proj_weight"][current_layer].t()
                    )
                    + prefill_output
                )

        sample_output = torch.empty(batch_size, dtype=torch.int64, device=hidden_state.device)

        for i in range(batch_size):
            # --- Final Layer Normalization and Output Projection ---
            rms = torch.sqrt(
                torch.mean(
                    hidden_state[batch_offsets[i] : batch_offsets[i + 1]] ** 2,
                    dim=-1,
                    keepdim=True,
                )
                + 1e-5
            )
            normalized_x = hidden_state[batch_offsets[i] : batch_offsets[i + 1]] / rms
            model_output = (
                normalized_x.to(torch.float16) * self.weights["model_layernorm_weight"]
            )
            logits = model_output.matmul(
                self.weights["lm_head_weight"].t()
            )  # (batch_size, seq_len, vocab_size)

            # Get the last token's logits for each sequence in the batch
            sample_output[i] = torch.argmax(logits, dim=1)[-1].item()
            # make sure the output has dimention (batch_size, )
        return sample_output.to(device="cpu")

    def generate_batched(self, input_string, rounds=20):
        input_ids_list = []
        for input_string in input_string:
            input_ids = self.tokenizer(input_string, return_tensors="pt").input_ids[0]
            input_ids_list.append(input_ids)

        output_ids_list = input_ids_list
        new_token = self.run(input_ids_list)
        for i in range(len(input_ids_list)):
            output_ids_list[i] = torch.cat(
                (output_ids_list[i], new_token[i : i + 1]), dim=0
            )

        for round in range(rounds - 1):
            # print(f"Round {round}")
            input_ids_list = []
            for output_ids in output_ids_list:
                input_ids_list.append(output_ids[-1:])
            new_token = self.run(input_ids_list, prefill=False)

            for i in range(len(input_ids_list)):
                output_ids_list[i] = torch.cat(
                    (output_ids_list[i], new_token[i : i + 1]), dim=0
                )
        output_text_list = []
        for output_ids in output_ids_list:
            output_text_list.append(
                self.tokenizer.decode(output_ids, skip_special_tokens=True)
            )
        return output_text_list


########################################
# Main Loop: Text Generation
########################################
if __name__ == "__main__":
    input_string = "Hello " * 510
    output_length = 128
    engine = Engine()

    times = {}

    for batch_size_pow in range(0, 7):
        batch_size = 2 ** batch_size_pow
        start = time.time()
        input_string_list = [input_string for _ in range(batch_size)]

        output_text = engine.generate_batched(input_string_list, rounds=output_length)
        end = time.time()
        times[batch_size] = end - start
        print(f"Batch size: {batch_size}, Time taken: {end - start} seconds")

    print(times)

    with open("different_prefill.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "time"])
        for batch_size, time in times.items():
            writer.writerow([batch_size, time])
