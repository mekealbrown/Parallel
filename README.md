# Parallel Computing (Spring 2025)

Welcome to my project portfolio for **Parallel Computing**. This repository showcases my exploration of parallel programming techniques, optimizations, and performance analysis, developed as part of my self-study at Lipscomb University. The goal? To harness the power of modern hardware through parallelism and vectorization—making code faster, smarter, and more efficient.

---

## About the Project

This portfolio is a collection of assignments, experiments, and optimizations crafted to deepen my understanding of parallel computing concepts. From multi-threading to SIMD vectorization, I’ve tackled real-world problems with a focus on performance and scalability.

### Learning Objectives

- Learn parallel programming paradigms (e.g., OpenMP, SIMD).
- Optimize algorithms for speed and resource efficiency.
- Analyze performance bottlenecks (cache, memory, CPU utilization).
- Apply vectorization techniques using modern instruction sets (e.g., AVX2).
- Benchmark and compare sequential vs. parallel implementations.

---

## Featured Project: Gaussian Blur with AVX2

### Overview

A high-performance Gaussian blur implementation in C, achieving a **500x speedup** over a naive box blur (kernel size 60) using AVX2 SIMD instructions. It leverages a separable two-pass approach with sliding windows, processing **8 pixels per cycle**.

### Highlights

- **500x Speedup**: Crushes a naive 4-loop blur, benchmarked on my Ryzen 5 3600.
- **AVX2 SIMD**: Vectorizes RGB sums with 256-bit registers.
- **Two-Pass Separable**: Reduces complexity from **O(n²)** to **O(n)** per pixel.
- **Sliding Window**: Dynamically updates sums for efficiency.
- **Benchmarks**: Profiles performance across image sizes (64x64 to 8192x8192).

### Build & Run

```bash
mkdir build
cd build
cmake ..
make
./blur_all <input_image> <output_image> <kernel_size>
```


### Example:
```bash
./blur_all ../images/input.png ../images/output.png 10
```

See the Gaussian Blur README for full details!

## Other Projects

### Matrix Multiplication with OpenMP 

Parallelized a matrix multiply, achieving a max 6.11% speedup on 11 threads.


## Tools and Technology

- **Languages**: C (primary), Python(analysis)
- **Parallel Frameworks**: OpenMP for multi-threading, AVX2 for SIMD.
- **Build System**: CMake for cross-platform compatibility, Make.
- **External Libraries**: STB Image for image I/O.
- **Hardware**: 6-Core AMD Ryzen 5 3600 with AVX2 support, 16Gi DDR4 RAM.

## Performance Insights

Across my projects, I’ve learned that parallelism isn’t just about throwing more threads or vector instructions at a problem—it’s about understanding the hardware.
### Key takeaways:

- **Cache is King**: Performance cliffs emerge when data exceeds cache size.
- **Vectorization Pays Off**: AVX2 shines when data is aligned and operations are uniform.
- **Overhead Matters**: Too many threads can slow things down—balance is critical.

## How to Explore

#### Clone the Repo:
```bash
git clone <repository-url>
cd Parallel
```

## About Me

I’m Mekeal Brown, a student passionate about squeezing every ounce of performance out of code. This portfolio reflects my journey through Parallel Computing in Spring 2025. Questions, feedback, or collaboration ideas? Reach me at brownmekeal@gmail.com.

## Acknowledgments

Dr. Dwayne Towell: For inspiring this deep dive into parallelism.
xAI’s Grok: For helping refine my READMEs with style and clarity.
Open Source Community: For tools like CMake and STB Image.
