#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define TRIALS 5

// Serial matrix multiplication
double matrix_multiply_serial(double** A, double** B, double** C, int n)
{
	clock_t start, end;
	start = clock();

  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      C[i][j] = 0.0;
      for(int k = 0; k < n; k++){
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// vectorized, scalar loop unrolled, B matrix column extraction
// Optimized vectorized matrix multiplication
double matrix_multiply_vectorized(double** A, double** B, double** C, int n)
{
  clock_t start, end;
  start = clock();
    
  // Align memory to 32-byte boundary for better vectorization
  double* b_col = (double*)aligned_alloc(32, n * sizeof(double));
    
  // Pre-calculate vector iteration limit
  int vec_limit = n - (n % 4);
    
  // Main computation loop with optimizations
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      // Prefetch next row of A
      if(j + 1 < n){
        __builtin_prefetch(&A[i][0], 0, 3);
      }
            
      #pragma omp simd
      for(int k = 0; k < n; k++){
        b_col[k] = B[k][j];
      }
            
      __m256d sum0 = _mm256_setzero_pd();
      __m256d sum1 = _mm256_setzero_pd();
            
      // Process 8 elements at once using two AVX registers
      for(int k = 0; k < vec_limit; k += 8){
        __m256d a_vec0 = _mm256_loadu_pd(&A[i][k]);
        __m256d b_vec0 = _mm256_loadu_pd(&b_col[k]);
        __m256d a_vec1 = _mm256_loadu_pd(&A[i][k + 4]);
        __m256d b_vec1 = _mm256_loadu_pd(&b_col[k + 4]);
                
        sum0 = _mm256_fmadd_pd(a_vec0, b_vec0, sum0);
        sum1 = _mm256_fmadd_pd(a_vec1, b_vec1, sum1);
      }
            
      // Combine the partial sums
      __m256d sum = _mm256_add_pd(sum0, sum1);
      double temp[4];
      _mm256_storeu_pd(temp, sum);
      C[i][j] = temp[0] + temp[1] + temp[2] + temp[3];
            
      // Handle remaining elements
      for(int k = vec_limit; k < n; k++){
        C[i][j] += A[i][k] * b_col[k];
      }
    }
  }
    
  free(b_col);
  end = clock();
  return ((double)(end - start)) / CLOCKS_PER_SEC;
}

double** allocate_matrix(int n)
{
  double** matrix = (double**)malloc(n * sizeof(double*));
  for(int i = 0; i < n; i++){
    matrix[i] = (double*)malloc(n * sizeof(double));
  }
  return matrix;
}

void initialize_matrix(double** matrix, int n)
{
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      matrix[i][j] = (double)rand() / RAND_MAX;
      }
  }
}
void free_matrix(double** matrix, int n)
{
  for(int i = 0; i < n; i++){
    free(matrix[i]);
  }
  free(matrix);
}

void calculate_speedup(int size)
{
  double** A = allocate_matrix(size);
  double** B = allocate_matrix(size);
  double** C = allocate_matrix(size);

	double serial1_times[TRIALS];
	double vectorized_times[TRIALS];
    
  // Run multiple trials
  for(int t = 0; t < TRIALS; t++){
    initialize_matrix(A, size);
    initialize_matrix(B, size);
        
    serial1_times[t] = matrix_multiply_serial(A, B, C, size);
		vectorized_times[t] = matrix_multiply_vectorized(A, B, C, size);
  }
  
	double avg_1, avg_2;
	avg_1 = avg_2 = 0;
	for(int t = 0; t < TRIALS; t++){
		avg_1 += serial1_times[t];
		avg_2 += vectorized_times[t];
	}
	avg_1 /= TRIALS;
	avg_2 /= TRIALS;

	printf("--------------------------------------------------------");
	printf("\nMatrix Size: %d x %d\n", size, size);
	printf("Average General Algorithm Time: %.4f seconds\n", avg_1);
	printf("Average Fully Optimized Algorithm Time: %.4f seconds\n", avg_2);
	printf("Speedup: %.2fx\n", avg_1/avg_2);
	printf("Efficiency: %.2f%%\n", (avg_1/avg_2) * 100);
	printf("--------------------------------------------------------\n");

  free_matrix(A, size);
  free_matrix(B, size);
  free_matrix(C, size);
}

int main() {
  // Test with different matrix sizes
  int sizes[] = {256, 512, 1024};
	int sizes2[] = {100, 356, 822};

	printf("Power of two sized matrices...\n\n");
  for(int i = 0; i < 3; i++){
    printf("Run %d...\n", i+1);
    calculate_speedup(sizes[i]);
  }

	printf("Random sized matrices...\n\n");
  for(int i = 0; i < 3; i++){
    printf("Run %d...\n", i+1);
    calculate_speedup(sizes2[i]);
  }
  return 0;
}
