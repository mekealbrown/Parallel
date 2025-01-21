#include <immintrin.h>  // AVX2 intrinsics
#include <time.h>
#include <stdio.h>
#include <stdlib.h>


#define TRIALS 5

// Helper function to print a vector (for understanding)
void print_vector(__m256d vec) {
    double temp[4];
    _mm256_storeu_pd(temp, vec);
    printf("[%.2f, %.2f, %.2f, %.2f]\n", temp[0], temp[1], temp[2], temp[3]);
}

double matrix_multiply_vectorized(double** A, double** B, double** C, int n) {
    clock_t start = clock();
    
    // Allocate aligned memory for column storage
    // 32-byte alignment needed for optimal AVX2 performance
    double* b_col = (double*)aligned_alloc(32, n * sizeof(double));
    if (!b_col) {
        fprintf(stderr, "Memory allocation failed!\n");
        return -1.0;
    }
    
    // Calculate vector processing limit
    int vec_limit = n - (n % 4);  // Round down to nearest multiple of 4
    
    // Process the matrix multiplication
    for(int i = 0; i < n; i++) {           // For each row in A
        for(int j = 0; j < n; j++) {       // For each column in B
            // Extract column j from matrix B into contiguous memory
            for(int k = 0; k < n; k++) {
                b_col[k] = B[k][j];
            }
            
            // Initialize vector accumulator for 4 parallel sums
            __m256d sum_vector = _mm256_setzero_pd();  // [0.0, 0.0, 0.0, 0.0]
            
            // Vectorized multiplication and accumulation
            for(int k = 0; k < vec_limit; k += 4) {
                // Load 4 consecutive elements from row i of A
                __m256d a_vector = _mm256_loadu_pd(&A[i][k]);
								if(i == 0) print_vector(a_vector);
	 
                // Load 4 consecutive elements from our column array
                __m256d b_vector = _mm256_loadu_pd(&b_col[k]);
                
                // Multiply and add in one instruction (FMA)
                sum_vector = _mm256_fmadd_pd(a_vector, b_vector, sum_vector);
            }
            
            // Sum up the vector elements
            double temp[4];
            _mm256_storeu_pd(temp, sum_vector);
            double final_sum = temp[0] + temp[1] + temp[2] + temp[3];
            
            // Handle remaining elements (if n not divisible by 4)
            for(int k = vec_limit; k < n; k++) {
                final_sum += A[i][k] * b_col[k];
            }
            
            // Store result in output matrix
            C[i][j] = final_sum;
        }
    }
    
    free(b_col);
    return ((double)(clock() - start)) / CLOCKS_PER_SEC;
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

	//double serial1_times[TRIALS];
	double serial2_times[TRIALS];
    
  // Run multiple trials
  for(int t = 0; t < TRIALS; t++){
    initialize_matrix(A, size);
    initialize_matrix(B, size);
        
    //serial1_times[t] = matrix_multiply_serial1(A, B, C, size);
		serial2_times[t] = matrix_multiply_vectorized(A, B, C, size);
  }
  
	double avg_1, avg_2;
	avg_1 = avg_2 = 0;
	for(int t = 0; t < TRIALS; t++){
		//avg_1 += serial1_times[t];
		avg_2 += serial2_times[t];
	}
	avg_1 /= TRIALS;
	avg_2 /= TRIALS;


	printf("\nMatrix Size: %d x %d\n", size, size);
	printf("Average General Algorithm Time: %.4f seconds\n", avg_1);
	printf("Average Optimized Algorithm Time: %.4f seconds\n", avg_2);
	printf("Speedup: %.2fx\n", avg_1/avg_2);
	printf("Efficiency: %.2f%%\n", (avg_1/avg_2) * 100);


  free_matrix(A, size);
  free_matrix(B, size);
  free_matrix(C, size);
}

int main() {
  // Test with different matrix sizes
  int sizes[] = {500, 1000};
  for(int i = 0; i < 2; i++){
    printf("Run %d...\n", i+1);
    calculate_speedup(sizes[i]);
  }
  return 0;
}