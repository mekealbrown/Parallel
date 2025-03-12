#Gaussian Blur in C with AVX2 Optimization

##A high-performance Gaussian blur implementation in C, leveraging AVX2 SIMD instructions to achieve a 500x speedup over a naive box blur (kernel size 60). This project uses a two-pass separable approach with sliding windows, processing 8 pixels per cycle.

##What It Does

    Blurs images fast: Takes an input image, applies a Gaussian blur, and outputs the result in PNG, JPG, or BMP.
    Optimized: Combines a separable two-pass algorithm with AVX2 vectorization for fast throughput.

##Key Features

    500x Speedup: Crushes a naive 4-nested-loop box blur (kernel=60) by a factor of roughly 500, measured on [6-Core AMD Ryzen 5 3600].
    AVX2 SIMD: Processes 8 pixels at once using 256-bit registers, vectorizing RGB sums in a sliding window.
    Two-Pass Separable: Splits the blur into horizontal and vertical passes, slashing complexity from O(n²) to O(n) per pixel.
    Sliding Window: Dynamically updates sums by adding/subtracting edge pixels, avoiding redundant calculations.
    Benchmarked: Includes a cache limit tester to profile performance across image sizes (64x64 to 8192x8192).


#How to Run It
###Prerequisites

    C compiler (e.g., GCC) with AVX2 support.
    CPU with AVX2 (most Intel since Haswell, AMD since Zen).
    STB Image libraries (stb_image.h, stb_image_write.h)—included.

###Build
bash
make blur_

	This builds with every blur function. Additional targets can be found
	in the Makefile for testing purposes.


Usage
bash
./blur_ <input_image> <output_image> <kernel_size>

Example:
bash
./blur_ input.png output.png 10

    <input_image>: Path to your image (PNG, JPG, BMP).
    <output_image>: Output path (same formats).
    <kernel_size>: Blur radius (e.g., 10). Must be odd.

Outputs

    Blurred image at <output_image>.
    Console performance table (unoptimized vs. optimized vs. SIMD).
    result.txt: Cache benchmark results (time, memory, speedup).

Performance Highlights

    Unoptimized: Naive 4-loop blur.
    Optimized (non-SIMD): Separable + sliding window.
    SIMD (AVX2): 500x speedup over naive, kernel=60. Scales with image size until cache limits kick in.

Sample benchmark:
text
=== Blur Function Performance ===
| Version              | Time (ms)     | Speedup       |
|----------------------|---------------|---------------|
| Unoptimized          |    47589.06   |      1.00     |
| Optimized (non-SIMD) |     410.79    |     115.85    |
| Optimized (SIMD)     |      89.85    |    529.64     |

Note: Exact times depend on your hardware. Run it yourself!
Code Breakdown

    blur_image: Naive baseline—4 loops, O(kernel²) per pixel.
    blur_image_opt: Separable two-pass with sliding window, O(kernel) per pixel.
    blur_image_opt_simd: AVX2-accelerated version, 8 pixels/cycle, same complexity but vectorized.
    benchmark_cache_limit: Profiles cache effects across sizes.
    test_blur_performance: Times all three versions.

Key optimization:
c
__m256i result_r = _mm256_add_epi32(result_r, _mm256_and_si256(reg, _mm256_set1_epi32(0xFF)));

Masks and sums 8 red channels in one go—repeated for green/blue.


Built during my self-study in Parallel Computing (Spring 2025). Questions? email: brownmekeal@gmail.com