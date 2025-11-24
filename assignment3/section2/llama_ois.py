import pandas as pd

# Lengths from 2^7 to 2^15
lengths = [2**i for i in range(7, 16)]

# Prefill OI formulas
def prefill_oi(model, p):
    if model == "LLaMA 2–7B":
        return p / 3
    elif model == "LLaMA 3–8B":
        return (2 * p) / 3
    elif model == "LLaMA 3–70B":
        return (4 * p) / 5

# Decode OI formulas
def decode_oi(model, c):
    if model == "LLaMA 2–7B":
        return c / (c + 0.5)
    elif model == "LLaMA 3–8B":
        return c / (c + 1)
    elif model == "LLaMA 3–70B":
        return (2 * c) / (c + 2)

models = ["LLaMA 2–7B", "LLaMA 3–8B", "LLaMA 3–70B"]

# Prepare DataFrames
prefill_df = pd.DataFrame(index=models, columns=lengths, dtype=float)
decode_df = pd.DataFrame(index=models, columns=lengths, dtype=float)

for model in models:
    for L in lengths:
        prefill_df.at[model, L] = prefill_oi(model, L)
        decode_df.at[model, L] = decode_oi(model, L)

# Display results
pd.set_option('display.precision', 4)
print("=== Prefill Attention OI (FLOPs/byte) ===")
print(prefill_df)
print("\n=== Decode Attention OI (FLOPs/byte) ===")
print(decode_df)