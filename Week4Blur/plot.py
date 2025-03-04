import matplotlib.pyplot as plt

# Function to parse result.txt
def parse_results(file_path):
    sizes = []
    times = []
    memories = []
    speedups = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Start after the header (skip first 4 lines)
        for line in lines[4:]:
            # Skip footer line
            if line.startswith('+'):
                continue
            # Split by '|' and strip whitespace
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) == 4:  # Ensure valid data row
                size = parts[0]           # e.g., "1792x1792"
                time = float(parts[1])    # e.g., "14.88"
                memory = float(parts[2])  # e.g., "37632"
                speedup = float(parts[3]) # e.g., "1.30"
                
                sizes.append(size)
                times.append(time)
                memories.append(memory)
                speedups.append(speedup)
    
    return sizes, times, memories, speedups

# Path to your result.txt file
file_path = "result.txt"

# Parse the data
sizes, times, memories, speedups = parse_results(file_path)

# Plot 1: Time (ms) vs Memory (KB)
plt.figure(figsize=(10, 6))
plt.plot(memories, times, marker='o', linestyle='-', color='b', label='Time (ms)')
plt.axvline(x=32768, color='r', linestyle='--', label='32 MiB L3 Cache (32,768 KB)')
plt.xlabel('Memory (KB)')
plt.ylabel('Time (ms)')
plt.title('Execution Time vs. Memory Usage')
plt.grid(True)
plt.legend()
plt.xscale('log')  # Log scale for better visibility of smaller sizes
plt.yscale('log')  # Log scale for time to see exponential growth
plt.xticks(memories, rotation=45)  # Show memory values on x-axis
plt.tight_layout()
plt.savefig('time_vs_memory.png')  # Save the plot
plt.show()

# Plot 2: Speedup vs Memory (KB)
plt.figure(figsize=(10, 6))
plt.plot(memories, speedups, marker='o', linestyle='-', color='g', label='Speedup')
plt.axvline(x=32768, color='r', linestyle='--', label='32 MiB L3 Cache (32,768 KB)')
plt.axhline(y=1.0, color='k', linestyle='--', label='Ideal Scaling (1.0)')
plt.xlabel('Memory (KB)')
plt.ylabel('Speedup')
plt.title('Speedup vs. Memory Usage')
plt.grid(True)
plt.legend()
plt.xscale('log')  # Log scale for memory
plt.xticks(memories, rotation=45)
plt.tight_layout()
plt.savefig('speedup_vs_memory.png')  # Save the plot
plt.show()