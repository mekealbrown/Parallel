import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_matrix_mult_roofline(sizes, serial_times, vectorized_times, peak_gflops=None):
    """
    Create a roofline plot specifically for matrix multiplication analysis
    
    Parameters:
    sizes: list of matrix sizes (N for NxN matrices)
    serial_times: list of execution times for serial implementation
    vectorized_times: list of execution times for vectorized implementation
    peak_gflops: Optional peak performance in GFLOPS (will be estimated if None)
    """
    # Calculate GFLOPS for each implementation
    # Matrix multiplication requires 2*N^3 floating point operations
    serial_gflops = []
    vectorized_gflops = []
    operational_intensities = []
    
    for i, N in enumerate(sizes):
        # Calculate FLOPs (2 operations per multiply-add, N^3 multiply-adds)
        flops = 2 * N * N * N
        
        # Convert to GFLOPS
        serial_gflops.append(flops / (serial_times[i] * 1e9))
        vectorized_gflops.append(flops / (vectorized_times[i] * 1e9))
        
        # Calculate operational intensity (FLOPS/byte)
        # Each element is accessed once, requiring 8 bytes (double precision)
        bytes_accessed = 3 * N * N * 8  # Input matrices A, B and output matrix C
        operational_intensities.append(flops / bytes_accessed)

    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the measured points
    plt.scatter(operational_intensities, serial_gflops, c='red', s=100, 
                label='Serial Implementation', marker='o')
    plt.scatter(operational_intensities, vectorized_gflops, c='green', s=100,
                label='Vectorized Implementation', marker='^')
    
    # Add connecting lines between implementations for same matrix size
    for i in range(len(sizes)):
        plt.plot([operational_intensities[i], operational_intensities[i]], 
                 [serial_gflops[i], vectorized_gflops[i]], 'k--', alpha=0.3)
        # Add matrix size annotation
        plt.annotate(f'{sizes[i]}x{sizes[i]}', 
                    (operational_intensities[i], vectorized_gflops[i]),
                    xytext=(10, 10), textcoords='offset points')

    # Calculate and plot roofline
    x = np.logspace(np.log10(min(operational_intensities)*0.5), 
                    np.log10(max(operational_intensities)*2), 1000)
    
    # Estimate peak performance if not provided
    if peak_gflops is None:
        peak_gflops = max(vectorized_gflops) * 1.2  # 20% higher than best measured
    
    # Estimate memory bandwidth from the memory-bound region
    memory_bandwidth = max([g/oi for g, oi in zip(vectorized_gflops, operational_intensities)])
    
    memory_bound = memory_bandwidth * x
    compute_bound = np.full_like(x, peak_gflops)
    roofline = np.minimum(memory_bound, compute_bound)
    
    plt.loglog(x, roofline, 'b-', linewidth=2, label='Roofline')
    
    # Formatting
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Operational Intensity (FLOPS/Byte)')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Roofline Analysis of Matrix Multiplication Implementations')
    plt.legend()
    
    # Add peak performance annotation
    plt.axhline(y=peak_gflops, color='b', linestyle='--', alpha=0.5)
    plt.text(min(operational_intensities), peak_gflops*1.1, 
             f'Peak Performance: {peak_gflops:.1f} GFLOPS')
    
    return plt

# Example usage with your data
df = pd.read_csv('matrix_performance.csv')

# Create separate lists for serial and vectorized implementations
sizes = df[df['implementation'] == 'serial']['size'].tolist()
serial_times = df[df['implementation'] == 'serial']['time'].tolist()
vectorized_times = df[df['implementation'] == 'vectorized']['time'].tolist()

# Create the roofline plot
plot = create_matrix_mult_roofline(sizes, serial_times, vectorized_times)
plt.show()