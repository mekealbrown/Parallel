#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <omp.h>

#define TIMING_RUNS 100
#define MAX_THREADS 16

double matrix_multiply_serial(double** A, double** B, double** C, int n);
void matrix_multiply_vectorized(double** A, double** B, double** C, int n);
double** allocate_matrix(int n);
void free_matrix(double** matrix, int n);
void initialize_matrix(double** matrix, int n);
void benchmark_matrix_multiply(int size);

void print_matrix(double** matrix, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%.2f ", matrix[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}


// Serial matrix multiplication
double matrix_multiply_serial(double** A, double** B, double** C, int n) {
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      C[i][j] = 0.0;
      for(int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return 1.0;
}

void matrix_multiply_vectorized(double** A, double** B, double** C, int n) {
  int vec_limit = (n / 4) * 4;  // nearest multiple of 4

  #pragma omp parallel for schedule(dynamic)
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      C[i][j] = 0.0;
      
      for(int k = 0; k < vec_limit; k += 4) {
    		__m256d a = _mm256_load_pd(&A[i][k]);
    		__m256d b = _mm256_set_pd(B[k+3][j], B[k+2][j], B[k+1][j], B[k][j]); // row major
    
    		__m256d c = _mm256_fmadd_pd(a, b, _mm256_setzero_pd());
    
    		C[i][j] += c[0] + c[1] + c[2] + c[3];
			}

      for(int k = vec_limit; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

double** allocate_matrix(int n) {
  double** matrix = NULL;
  size_t alignment = 32;
  
  size_t padded_n = ((n + 3) / 4) * 4;  // round up to nearest multiple of 4
  
  if (posix_memalign((void**)&matrix, alignment, n * sizeof(double*)) != 0) {
    fprintf(stderr, "Failed to allocate matrix pointer array\n");
    return NULL;
  }

  for (int i = 0; i < n; i++) {
    matrix[i] = NULL;
  }

  for (int i = 0; i < n; i++) {
    if (posix_memalign((void**)&matrix[i], alignment, padded_n * sizeof(double)) != 0) {
      fprintf(stderr, "Failed to allocate matrix row %d\n", i);
      free_matrix(matrix, n);
      return NULL;
    }

    // Initialize padded area to zero
    for (int j = 0; j < padded_n; j++) {
      matrix[i][j] = 0.0;
    }

    // Verify alignment
    if (((uintptr_t)matrix[i]) % alignment != 0) {
      fprintf(stderr, "Row %d is not properly aligned\n", i);
      free_matrix(matrix, n);
      return NULL;
    }
  }

  return matrix;
}

void free_matrix(double** matrix, int n) {
  if (matrix) {
    for (int i = 0; i < n; i++) {
      free(matrix[i]);
    }
    free(matrix);
  }
}

void initialize_matrix(double** matrix, int n) {
  if (!matrix) return;
  
  for (int i = 0; i < n; i++) {
    if (!matrix[i]) continue;
    for (int j = 0; j < n; j++) {
      matrix[i][j] = (double)rand() / RAND_MAX;
    }
  }
}

void benchmark_matrix_multiply(int size) {
  double** A = allocate_matrix(size);
  double** B = allocate_matrix(size);
  double** C_serial = allocate_matrix(size);
  double** C_vectorized = allocate_matrix(size);
  
  if (!A || !B || !C_serial || !C_vectorized) {
    fprintf(stderr, "Memory allocation failed for matrices\n");
    // Proper cleanup with size parameter
    free_matrix(A, size);
    free_matrix(B, size);
    free_matrix(C_serial, size);
    free_matrix(C_vectorized, size);
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
  for(int run = 0; run < TIMING_RUNS; run++) {
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
  for(int num_threads = 1; num_threads <= MAX_THREADS; num_threads *= 2) {
    omp_set_num_threads(num_threads);
  
    start = omp_get_wtime();
    for(int run = 0; run < TIMING_RUNS; run++) {
      initialize_matrix(A, size);
      initialize_matrix(B, size);
      matrix_multiply_vectorized(A, B, C_vectorized, size);
    }
    end = omp_get_wtime();
    double vec_avg = (end - start) / TIMING_RUNS;
  
    // find best time and num threads of best time
    if(best_threads_time > vec_avg) {
      best_threads_time = vec_avg;
      best_threads_num = num_threads;
    }
  }

  // performance summary
  printf("\nSummary:\n");
  printf("Serial Implementation:\n");
  printf(" Avg Time: %.6f s\n", serial_avg);
  printf("\nBest Vectorized Implementation (%d threads):\n", best_threads_num);
  printf(" Avg Time: %.6f s\n", best_threads_time);
  printf("Speedup: %.2fx\n", serial_avg / best_threads_time);

  free_matrix(A, size);
  free_matrix(B, size);
  free_matrix(C_serial, size);
  free_matrix(C_vectorized, size);
}


int main(int argc, char **argv) {
  // test with different matrix sizes
  int sizes[] = {1000, 2000, 3000};
  
  for(int i = 0; i < 3; i++) {
    printf("Run %d: Matrix Size %d x %d\n", i+1, sizes[i], sizes[i]);
    benchmark_matrix_multiply(sizes[i]);
    printf("\n");
  }

  return 0;
}