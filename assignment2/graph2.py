import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
uniform_df = pd.read_csv('uniform_prefill.csv')
different_df = pd.read_csv('different_prefill.csv')

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data with markers and lines
plt.plot(uniform_df['batch_size'], uniform_df['time'], 'o-', label='Uniform Prefill', color='blue')
plt.plot(different_df['batch_size'], different_df['time'], 's-', label='Different Prefill', color='red')

# Set x-axis to logarithmic scale
plt.xscale('log', base=2)

# Add labels and title
plt.xlabel('Batch Size')
plt.ylabel('Time (ms)')
plt.title('Comparison of Uniform vs Different Prefill Time')
plt.grid(True, which="both", ls="--", alpha=0.5)

# Add x-ticks for batch sizes
plt.xticks(uniform_df['batch_size'], [str(int(x)) for x in uniform_df['batch_size']])

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('prefill_comparison.png', dpi=300)
plt.show()