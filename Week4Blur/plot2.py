import matplotlib.pyplot as plt

# Function to parse result.txt and compute time per pixel
def parse_results(file_path):
    sizes = []
    times = []
    memories = []
    pixels = []
    time_per_pixel = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Skip header (first 4 lines)
        for line in lines[4:]:
            # Skip footer line
            if line.startswith('+'):
                continue
            # Split by '|' and strip whitespace
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) == 4:  # Ensure valid data row
                size_str = parts[0]       # e.g., "1792x1792"
                time = float(parts[1])    # e.g., "14.88"
                memory = float(parts[2])  # e.g., "37632"
                # Extract width and height from size (e.g., "1792x1792" -> 1792 * 1792)
                width, height = map(int, size_str.split('x'))
                pixel_count = width * height
                
                sizes.append(size_str)
                times.append(time)
                memories.append(memory)
                pixels.append(pixel_count)
                time_per_pixel.append(time / pixel_count * 1000)  # ms per pixel (x1000 for scale)
    
    return sizes, times, memories, pixels, time_per_pixel

# Path to result.txt file
file_path = "result.txt"

# Parse the data
sizes, times, memories, pixels, time_per_pixel = parse_results(file_path)

# Plot: Time per Pixel (ms/pixel) vs Memory (KB)
plt.figure(figsize=(10, 6))
plt.plot(memories, time_per_pixel, marker='o', linestyle='-', color='b', label='Time per Pixel (ms/pixel x 10³)')
plt.axvline(x=32768, color='r', linestyle='--', label='32 MiB L3 Cache (32,768 KB)')
plt.xlabel('Memory (KB)')
plt.ylabel('Time per Pixel (ms/pixel x 10³)')
plt.title('Execution Time per Pixel vs. Memory Usage')
plt.grid(True)
plt.legend()
plt.xscale('log')  # Log scale for memory to spread out smaller values
# plt.yscale('log')  # Optional: log scale for y-axis; comment out for linear
plt.xticks(memories, rotation=45)  # Show memory values on x-axis
plt.tight_layout()
plt.savefig('time_per_pixel_vs_memory.png')  # Save the plot
plt.show()