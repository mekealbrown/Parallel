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
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      C[i][j] = 0.0;
      for(int k = 0; k < n; k++){
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
	return 1.0;
}

// MODIFIED TO BE BETTER THREAD CAPABLE
double matrix_multiply_vectorized(double** A, double** B, double** C, int n)
{
  int vec_limit = n - (n % 4);

  #pragma omp parallel for
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      double local_sum = 0.0;
      
      // registers to accumulate sums
      __m256d vec_sum1 = _mm256_setzero_pd();
      
      // Process in chunks of 8 elements at a time using 2 SIMD registers
      for(int k = 0; k < vec_limit; k += 4){
        // Load 8 elements from A and B using two __m256d registers
        __m256d a_vec1 = _mm256_loadu_pd(&A[i][k]);
        __m256d b_vec1 = _mm256_loadu_pd(&B[k][j]);
        
        vec_sum1 = _mm256_fmadd_pd(a_vec1, b_vec1, vec_sum1);
      }
      // Horizontal sum for both vectors
      double temp1[4];
      _mm256_storeu_pd(temp1, vec_sum1);
      
      // Add the results from both SIMD vectors
      local_sum = temp1[0] + temp1[1] + temp1[2] + temp1[3];

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
  size_t alignment = 32;
  
  // First allocation
  if(posix_memalign((void**)&matrix, alignment, n * sizeof(double*)) != 0){
    return NULL;
  }
  
  // Second allocation
  void* data;
  if(posix_memalign(&data, alignment, n * n * sizeof(double)) != 0){
    free(matrix);  // Clean up first allocation
    return NULL;
  }
  matrix[0] = (double*)data;
  
  // Set up row pointers
  for(int i = 1; i < n; i++){
    matrix[i] = matrix[0] + i * n;
  }
  return matrix;
}

void free_matrix(double** matrix)
{
  if(matrix){
    free(matrix[0]);  // Free the data array
    free(matrix);     // Free the pointer array
  }
}

void initialize_matrix(double** matrix, int n)
{
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      matrix[i][j] = (double)rand() / RAND_MAX;
      }
  }

}

void benchmark_matrix_multiply(int size)
{
  // Allocate matrices
	double** A = allocate_matrix(size);
  double** B = allocate_matrix(size);
  double** C_serial = allocate_matrix(size);
  double** C_vectorized = allocate_matrix(size);
  
  if(!A || !B || !C_serial || !C_vectorized){
    free_matrix(A);
    free_matrix(B);
    free_matrix(C_serial);
    free_matrix(C_vectorized);
    printf("Memory allocation failed\n");
    return;
  }

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

	double serial_avg = serial_total / TIMING_RUNS;

  printf("\nBenchmarking Vectorized Implementation (Size: %d x %d)\n", size, size);
	printf("Threads\t\tMin Time (s)\t\tMax Time (s)\t\tAvg Time (s)\t\tSpeedup\n");

	// Variables to track the best performance(for performance summary)
	double best_vec_min = DBL_MAX, best_vec_max = 0, best_vec_total = 0;
	int best_thread_count = 1;
	double base_time = DBL_MAX;

	for(int num_threads = 1; num_threads <= MAX_THREADS; num_threads *= 2){
	  omp_set_num_threads(num_threads);

	  double vec_min = DBL_MAX, vec_max = 0, vec_total = 0;
	  for(int run = 0; run < TIMING_RUNS; run++){
      initialize_matrix(A, size);
      initialize_matrix(B, size);

      double start = omp_get_wtime();
      matrix_multiply_vectorized(A, B, C_vectorized, size);
      double end = omp_get_wtime();
      double elapsed = end - start;

      // Track min, max, total
      vec_min = (elapsed < vec_min) ? elapsed : vec_min;
      vec_max = (elapsed > vec_max) ? elapsed : vec_max;
      vec_total += elapsed;
	  }
	
	  // Calculate average time
	  double vec_avg = vec_total / TIMING_RUNS;
	
	  // Update base time in first iteration (1 thread)
	  if(num_threads == 1){
	      base_time = vec_avg;
	  }
	
	  // Track the best performance
	  if(vec_min < best_vec_min){
	      best_vec_min = vec_min;
	      best_vec_max = vec_max;
	      best_vec_total = vec_total;
	      best_thread_count = num_threads;
	  }
	
	  // Print results for this thread count
	  printf("%d\t\t%.6f\t\t%.6f\t\t%.6f\t\t%.2fx\n",
	     num_threads,
	     vec_min,
	     vec_max,
	     vec_avg,
	     base_time / vec_avg);
	}

	// Calculate best vectorized average
	double best_vec_avg = best_vec_total / TIMING_RUNS;

	// Performance summary
	printf("\nPerformance Summary (Size: %d x %d)\n", size, size);
	printf("Serial Implementation:\n");
	printf(" Min Time: %.6f s\n", serial_min);
	printf(" Max Time: %.6f s\n", serial_max);
	printf(" Avg Time: %.6f s\n", serial_avg);
	printf("Vectorized Implementation (Best: %d threads):\n", best_thread_count);
	printf(" Min Time: %.6f s\n", best_vec_min);
	printf(" Max Time: %.6f s\n", best_vec_max);
	printf(" Avg Time: %.6f s\n", best_vec_avg);
	printf("Speedup: %.2fx\n", serial_avg / best_vec_avg);

	// Free matrices
  free_matrix(A);
  free_matrix(B);
  free_matrix(C_serial);
  free_matrix(C_vectorized);
}

int main(int argc, char **argv) {
  // Test with different matrix sizes
  int sizes[] = {150, 256, 400};
	
  for(int i = 0; i < 3; i++){
    printf("Run %d: Matrix Size %d x %d\n", i+1, sizes[i], sizes[i]);
    benchmark_matrix_multiply(sizes[i]);
    printf("\n");
  }

  return 0;
}
