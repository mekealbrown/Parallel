#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <omp.h>

#define TIMING_RUNS 100
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

  #pragma omp parallel for schedule(dynamic)
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      double local_sum = 0.0;
      
      // registers to accumulate sums
      __m256d vec_sum1 = _mm256_setzero_pd();
      
      for(int k = 0; k < vec_limit; k += 4){
        // load 8 elements from A and B using two __m256d registers
        __m256d a_vec1 = _mm256_load_pd(&A[i][k]);
        __m256d b_vec1 = _mm256_load_pd(&B[k][j]);
        
        vec_sum1 = _mm256_fmadd_pd(a_vec1, b_vec1, vec_sum1);
      }
      double temp1[4];
      _mm256_storeu_pd(temp1, vec_sum1);
      
      // add result
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

  // aligned memory for the row pointers
  if(posix_memalign((void**)&matrix, alignment, n * sizeof(double*)) != 0){
    return NULL;
  }

	// ensure rows are aligned too (fixes memory issues with load_pd intrinsic)
  for(int i = 0; i < n; i++){
    if(posix_memalign((void**)&matrix[i], alignment, n * sizeof(double)) != 0){
      for(int j = 0; j < i; j++){
        free(matrix[j]);
      }
      free(matrix);
      return NULL;
    }
  }

  // ensure each row starts at an aligned address
  for(int i = 0; i < n; i++){
    if(((uintptr_t)(matrix[i]) % 32) != 0){
      printf("Row %d is not aligned\n", i);
      return NULL;
    }
  }

  return matrix;
}


void free_matrix(double** matrix)
{
  if(matrix){
    free(matrix[0]);
    free(matrix);
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

	//======================= Warmup Cache With Data ======================================
	volatile int temp;
	for(int i = 0; i < size; i++) {
    for(int j = 0; j < size; j++) {
      temp = A[i][j]; // volatile to stop optimization
			temp = B[i][j];
    }
  }

	//======================= Benchmark Serial Implementation =============================
	double start = omp_get_wtime();
	for(int run = 0; run < TIMING_RUNS; run++){
	  initialize_matrix(A, size);
	  initialize_matrix(B, size);
	  matrix_multiply_serial(A, B, C_serial, size);
	}
	double end = omp_get_wtime();
	double serial_avg = (end - start) / TIMING_RUNS;

	//======================= Benchmark Vectorized Implementation ==========================

	double best_threads_time = DBL_MAX;
	int best_threads_num = 0;

	// test different thread counts
	for(int num_threads = 1; num_threads <= MAX_THREADS; num_threads *= 2){
	  omp_set_num_threads(num_threads);
	
	  start = omp_get_wtime();
	  for(int run = 0; run < TIMING_RUNS; run++){
	    initialize_matrix(A, size);
	    initialize_matrix(B, size);
	    matrix_multiply_vectorized(A, B, C_vectorized, size);
	  }
	  end = omp_get_wtime();
	  double vec_avg = (end - start) / TIMING_RUNS;
	
	  // find best time and num threads of best time
	  if(best_threads_time > vec_avg){
	    best_threads_time = vec_avg;
	    best_threads_num = num_threads;
	  }
	}

	//===============================================================

	// performance summary
	printf("\nSummary:\n");
	printf("Serial Implementation:\n");
	printf(" Avg Time: %.6f s\n", serial_avg);
	printf("\nBest Vectorized Implementation (%d threads):\n", (int)best_threads_num);
	printf(" Avg Time: %.6f s\n", best_threads_time);
	printf("Speedup: %.2fx\n", serial_avg / best_threads_time);

  free_matrix(A);
  free_matrix(B);
  free_matrix(C_serial);
  free_matrix(C_vectorized);
}


int main(int argc, char **argv) {
  // test with different matrix sizes
  int sizes[] = {1000, 1500, 2000};
	
  for(int i = 0; i < 3; i++){
    printf("Run %d: Matrix Size %d x %d\n", i+1, sizes[i], sizes[i]);
    benchmark_matrix_multiply(sizes[i]);
    printf("\n");
  }

  return 0;
}
