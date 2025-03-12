---

# Gaussian Blur in C with AVX2 Optimization

A high-performance Gaussian blur implementation in C, leveraging AVX2 SIMD instructions to achieve a **500x speedup** over a naive box blur (kernel size 60). This project uses a two-pass separable approach with sliding windows, processing **8 pixels per cycle**.

---

## What It Does

- **Blurs Images Fast**: Takes an input image, applies a Gaussian blur, and outputs the result in PNG, JPG, or BMP format.
- **Highly Optimized**: Combines a separable two-pass algorithm with AVX2 vectorization for blazing-fast throughput.

---

## Key Features

- **500x Speedup**: Outperforms a naive 4-nested-loop box blur (kernel=60) by ~500x, benchmarked on a [6-Core AMD Ryzen 5 3600].
- **AVX2 SIMD**: Processes 8 pixels simultaneously using 256-bit registers, vectorizing RGB sums in a sliding window.
- **Two-Pass Separable**: Splits the blur into horizontal and vertical passes, reducing complexity from **O(n²)** to **O(n)** per pixel.
- **Sliding Window**: Dynamically updates sums by adding/subtracting edge pixels, eliminating redundant calculations.
- **Benchmarked**: Includes a cache limit tester to profile performance across image sizes (64x64 to 8192x8192).

---

## How to Run It

### Prerequisites

- **C Compiler**: GCC or any compiler with AVX2 support.
- **CMake**: Ensure you have CMake installed on your system
- **OpenMP**: Ensure you have OpenMP installed on your system 
- **CPU**: Must support AVX2 (e.g., Intel Haswell or later, AMD Zen or later).
- **STB Image Libraries**: `stb_image.h` and `stb_image_write.h` are included in the project.

### Build

```bash
mkdir build
cd build
cmake ..
make
```

### Usage

```bash
./blur_all <input_image> <output_image> <kernel_size>
```

#### Example

```bash
./blur_all ../images/input.png ../images/output.png 10
```

- `<input_image>`: Path to your image (PNG, JPG, BMP).
- `<output_image>`: Output file path (same formats supported).
- `<kernel_size>`: Blur radius (e.g., 10).

### Outputs

- **Blurred Image**: Saved at `<output_image>`.
- **Console Table**: Performance comparison (unoptimized vs. optimized vs. SIMD).
- **result.txt**: Cache benchmark results (time, memory, speedup).

---

## Performance Highlights

- **Unoptimized**: Naive 4-loop blur, painfully slow.
- **Optimized (non-SIMD)**: Separable + sliding window, much faster.
- **SIMD (AVX2)**: ~500x speedup over naive (kernel=60).

### Sample Benchmark

```
=== Blur Function Performance ===
| Version              | Time (ms)  | Speedup  |
|----------------------|------------|----------|
| Unoptimized          | 47589.06   | 1.00     |
| Optimized (non-SIMD) | 410.79     | 115.85   |
| Optimized (SIMD)     | 89.85      | 529.64   |
```

**Note**: Exact times vary by hardware. Run it on your system!

---

## Code Breakdown

- **`blur_image`**: Naive baseline—4 loops, **O(kernel²)** per pixel.
- **`blur_image_opt`**: Separable two-pass with sliding window, **O(kernel)** per pixel.
- **`blur_image_opt_simd`**: AVX2-accelerated, 8 pixels/cycle, same complexity but vectorized.
- **`benchmark_cache_limit`**: Profiles slowdowns as image size approaches cache limits and spills to RAM.
- **`test_blur_performance`**: Times all three versions for comparison.

### Key Optimization Example

```c
__m256i result_r = _mm256_add_epi32(result_r, _mm256_and_si256(reg, _mm256_set1_epi32(0xFF)));
```

- Masks and sums 8 red channels in one instruction—repeated for green and blue.

---

## About

Built during my self-study in **Parallel Computing** (Spring 2025). Questions or feedback? Email me at: **brownmekeal@gmail.com**.

---
