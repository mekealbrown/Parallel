#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <omp.h>

#define WARMUP_RUNS 3
#define TIMING_RUNS 10
#define MAX_THREADS 16

// Serial matrix multiplication
double matrix_multiply_serial(double** A, double** B, double** C, int n)
{
	//clock_t start, end;
	//start = clock();
	#pragma omp parallel for collapse(2)
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      C[i][j] = 0.0;
      for(int k = 0; k < n; k++){
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
	//end = clock();
	return 1.0; //((double)(end - start)) / CLOCKS_PER_SEC;
}

// MODIFIED TO BE BETTER THREAD CAPABLE
double matrix_multiply_vectorized(double** A, double** B, double** C, int n)
{
  int vec_limit = n - (n % 4);
  double local_sum = 0.0;

  // Parallelize only outer loop, reduction gives each thread its own copy of local_sum
  #pragma omp parallel for reduction(+:local_sum)
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      double local_sum = 0.0;
      
      // registers to accumulate sums
      __m256d vec_sum1 = _mm256_setzero_pd();
      __m256d vec_sum2 = _mm256_setzero_pd();
      
      // Process in chunks of 8 elements at a time using 2 SIMD registers
      for(int k = 0; k < vec_limit; k += 8){
        // Load 8 elements from A and B using two __m256d registers
        __m256d a_vec1 = _mm256_load_pd(&A[i][k]);
        __m256d b_vec1 = _mm256_load_pd(&B[k][j]);
        __m256d a_vec2 = _mm256_load_pd(&A[i][k+4]);
        __m256d b_vec2 = _mm256_load_pd(&B[k+4][j]);
        
        vec_sum1 = _mm256_fmadd_pd(a_vec1, b_vec1, vec_sum1);
        vec_sum2 = _mm256_fmadd_pd(a_vec2, b_vec2, vec_sum2);
      }
      // Horizontal sum for both vectors
      double temp1[4], temp2[4];
      _mm256_storeu_pd(temp1, vec_sum1);
      _mm256_storeu_pd(temp2, vec_sum2);
      
      // Add the results from both SIMD vectors
      local_sum = temp1[0] + temp1[1] + temp1[2] + temp1[3] + temp2[0] + temp2[1] + temp2[2] + temp2[3];

      for(int k = vec_limit; k < n; k++){
        local_sum += A[i][k] * B[k][j];
      }

      C[i][j] = local_sum;
    }
  }
  return 1.0;
}

double** allocate_matrix(int n)
{
  double** matrix;
  size_t alignment = 32; // Align to 32 bytes (for AVX2 - 256bits width)
	
  // memory for the row pointers with proper alignment
  if(posix_memalign((void**)&matrix, alignment, n * sizeof(double*)) != 0){
    printf("Memory allocation failed\n");
    return NULL;
  }

  // memory for the matrix data (the actual elements)
  if(posix_memalign((void**)&matrix[0], alignment, n * n * sizeof(double)) != 0){
    printf("Memory allocation failed for matrix data\n");
    free(matrix);
    return NULL;
  }

  for(int i = 1; i < n; i++){
    matrix[i] = matrix[0] + i * n;
  }
  return matrix;
}

void free_matrix(double** matrix)
{
  free(matrix[0]);
  free(matrix);
}

void initialize_matrix(double** matrix, int n)
{
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      matrix[i][j] = (double)rand() / RAND_MAX;
      }
  }

}

// Benchmark function to measure and compare performance
void benchmark_matrix_multiply(int size) {
  // Allocate matrices
  double** A = allocate_matrix(size);
  double** B = allocate_matrix(size);
  double** C_serial = allocate_matrix(size);
  double** C_vectorized = allocate_matrix(size);
  // Arrays to store timing results
  double serial_times[TIMING_RUNS];
  double vectorized_times[TIMING_RUNS];

  // reduce "cold start" effects
  for(int warmup = 0; warmup < WARMUP_RUNS; warmup++){
    initialize_matrix(A, size);
    initialize_matrix(B, size);
    matrix_multiply_serial(A, B, C_serial, size);
    matrix_multiply_vectorized(A, B, C_vectorized, size);
  }
  // Benchmark serial implementation
  printf("\nBenchmarking Serial Implementation (Size: %d x %d)\n", size, size);
  double serial_min = DBL_MAX, serial_max = 0, serial_total = 0;
  for(int run = 0; run < TIMING_RUNS; run++){
    initialize_matrix(A, size);
    initialize_matrix(B, size);
    double start = omp_get_wtime();
    matrix_multiply_serial(A, B, C_serial, size);
    double end = omp_get_wtime();
    
    double elapsed = end - start;
    serial_times[run] = elapsed;
    
    // Track min, max, total
    serial_min = (elapsed < serial_min) ? elapsed : serial_min;
    serial_max = (elapsed > serial_max) ? elapsed : serial_max;
    serial_total += elapsed;
  }
  // Benchmark vectorized implementation
  printf("\nBenchmarking Vectorized Implementation (Size: %d x %d)\n", size, size);
  double vec_min = DBL_MAX, vec_max = 0, vec_total = 0;
  for(int run = 0; run < TIMING_RUNS; run++){
    initialize_matrix(A, size);
    initialize_matrix(B, size);
    double start = omp_get_wtime();
    matrix_multiply_vectorized(A, B, C_vectorized, size);
    double end = omp_get_wtime();
    
    double elapsed = end - start;
    vectorized_times[run] = elapsed;
    
    // Track min, max, total
    vec_min = (elapsed < vec_min) ? elapsed : vec_min;
    vec_max = (elapsed > vec_max) ? elapsed : vec_max;
    vec_total += elapsed;
  }
  // Calculate statistics
  double serial_avg = serial_total / TIMING_RUNS;
  double vec_avg = vec_total / TIMING_RUNS;
  // Thread scaling analysis (for vectorized implementation)
  printf("\nThread Scaling Analysis (Size: %d x %d)\n", size, size);
  printf("Threads\t\tTime (s)\t\tSpeedup\n");
  double base_time = vec_avg;
  for(int num_threads = 1; num_threads <= MAX_THREADS; num_threads *= 2){
    omp_set_num_threads(num_threads);
    
    double thread_total = 0;
    for(int run = 0; run < TIMING_RUNS; run++){
      initialize_matrix(A, size);
      initialize_matrix(B, size);
      double start = omp_get_wtime();
      matrix_multiply_vectorized(A, B, C_vectorized, size);
      double end = omp_get_wtime();
      
      thread_total += end - start;
    }
    double thread_avg = thread_total / TIMING_RUNS;
    
    printf("%d\t\t%.6f\t\t%.2fx\n", 
          num_threads, 
          thread_avg, 
          base_time / thread_avg);
  }
  // Performance summary
  printf("\nPerformance Summary (Size: %d x %d)\n", size, size);
  printf("Serial Implementation:\n");
  printf("  Min Time:    %.6f s\n", serial_min);
  printf("  Max Time:    %.6f s\n", serial_max);
  printf("  Avg Time:    %.6f s\n", serial_avg);
  
  printf("Vectorized Implementation:\n");
  printf("  Min Time:    %.6f s\n", vec_min);
  printf("  Max Time:    %.6f s\n", vec_max);
  printf("  Avg Time:    %.6f s\n", vec_avg);
  
  printf("Speedup:       %.2fx\n", serial_avg / vec_avg);
  // Free matrices
  free_matrix(A);
  free_matrix(B);
  free_matrix(C_serial);
  free_matrix(C_vectorized);
}

int main(int argc, char **argv) {
  // Test with different matrix sizes
  int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
	
  for(int i = 0; i < 6; i++){
    printf("Run %d: Matrix Size %d x %d\n", i+1, sizes[i], sizes[i]);
    benchmark_matrix_multiply(sizes[i]);
    printf("\n");
  }

  return 0;
}
