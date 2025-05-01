import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the CSV files
no_kv_data = pd.read_csv('no_kv_times.csv')
single_batch_data = pd.read_csv('single_batch.csv')

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot both datasets
plt.plot(no_kv_data['output_length'], no_kv_data['time'], marker='o', linestyle='-', label='No KV Cache')
plt.plot(single_batch_data['output_length'], single_batch_data['time'], marker='s', linestyle='-', label='Single Batch')

# Add labels and title
plt.xlabel('Output Length')
plt.ylabel('Time (seconds)')
plt.title('Comparison of No KV Cache vs Single Batch Processing Time')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Set the y-axis to start from 0
plt.ylim(bottom=0)

# Save the figure
plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.tight_layout()
plt.show()