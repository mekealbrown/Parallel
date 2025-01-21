#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define TRIALS 5

// Structure to hold performance metrics
typedef struct {
    double time;
    double gflops;
    double bytes_accessed;
    double operational_intensity;
} perf_metrics_t;


double** allocate_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for(int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void initialize_matrix(double** matrix, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
}

void free_matrix(double** matrix, int n) {
    for(int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Function to calculate performance metrics
perf_metrics_t calculate_metrics(int size, double time, char is_serial) {
    perf_metrics_t metrics;
    metrics.time = time;
    
    // Calculate total FLOPS (2 operations per multiply-add)
		double total_flops;
		total_flops = is_serial ? 2.0 * size * size * size : size * size * size;
    //double total_flops = 2.0 * size * size * size;
    metrics.gflops = (total_flops / time) / 1e9;
    
    // Calculate memory traffic (in bytes)
    // Reading matrix A: size^2 doubles
    // Reading matrix B: size^2 doubles
    // Writing matrix C: size^2 doubles
    metrics.bytes_accessed = 3.0 * size * size * sizeof(double);
    
    // Calculate operational intensity (FLOPS/byte)
    metrics.operational_intensity = total_flops / metrics.bytes_accessed;
    
    return metrics;
}

// Modified performance measurement functions
perf_metrics_t matrix_multiply_serial(double** A, double** B, double** C, int n) {
    clock_t start = clock();
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for(int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    double time = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    return calculate_metrics(n, time, 1);
}

perf_metrics_t matrix_multiply_vectorized(double** A, double** B, double** C, int n) {
    clock_t start = clock();
    
    double* b_col = (double*)aligned_alloc(32, n * sizeof(double));
    int vec_limit = n - (n % 4);
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if (j + 1 < n) {
                __builtin_prefetch(&A[i][0], 0, 3);
            }
            
            #pragma omp simd
            for(int k = 0; k < n; k++) {
                b_col[k] = B[k][j];
            }
            
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            
            for(int k = 0; k < vec_limit; k += 8) {
                __m256d a_vec0 = _mm256_loadu_pd(&A[i][k]);
                __m256d b_vec0 = _mm256_loadu_pd(&b_col[k]);
                __m256d a_vec1 = _mm256_loadu_pd(&A[i][k + 4]);
                __m256d b_vec1 = _mm256_loadu_pd(&b_col[k + 4]);
                
                sum0 = _mm256_fmadd_pd(a_vec0, b_vec0, sum0);
                sum1 = _mm256_fmadd_pd(a_vec1, b_vec1, sum1);
            }
            
            __m256d sum = _mm256_add_pd(sum0, sum1);
            double temp[4];
            _mm256_storeu_pd(temp, sum);
            C[i][j] = temp[0] + temp[1] + temp[2] + temp[3];
            
            for(int k = vec_limit; k < n; k++) {
                C[i][j] += A[i][k] * b_col[k];
            }
        }
    }
    
    free(b_col);
    double time = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    return calculate_metrics(n, time, 0);
}

void save_performance_data(int size) {
    double** A = allocate_matrix(size);
    double** B = allocate_matrix(size);
    double** C = allocate_matrix(size);
    
    perf_metrics_t serial_metrics[TRIALS];
    perf_metrics_t vectorized_metrics[TRIALS];
    
    // Run multiple trials
    for(int t = 0; t < TRIALS; t++) {
        initialize_matrix(A, size);
        initialize_matrix(B, size);
        
        serial_metrics[t] = matrix_multiply_serial(A, B, C, size);
        vectorized_metrics[t] = matrix_multiply_vectorized(A, B, C, size);
    }
    
    // Calculate averages
    perf_metrics_t avg_serial = {0}, avg_vectorized = {0};
    for(int t = 0; t < TRIALS; t++) {
        avg_serial.time += serial_metrics[t].time;
        avg_serial.gflops += serial_metrics[t].gflops;
        avg_serial.bytes_accessed = serial_metrics[t].bytes_accessed; // Same for all trials
        avg_serial.operational_intensity = serial_metrics[t].operational_intensity; // Same for all trials
        
        avg_vectorized.time += vectorized_metrics[t].time;
        avg_vectorized.gflops += vectorized_metrics[t].gflops;
        avg_vectorized.bytes_accessed = vectorized_metrics[t].bytes_accessed;
        avg_vectorized.operational_intensity = vectorized_metrics[t].operational_intensity;
    }
    
    avg_serial.time /= TRIALS;
    avg_serial.gflops /= TRIALS;
    avg_vectorized.time /= TRIALS;
    avg_vectorized.gflops /= TRIALS;
    
    // Save results to file
    FILE *fp = fopen("matrix_performance.csv", "a");
    if (!fp) {
        printf("Error opening file!\n");
        return;
    }
    
    // Write header if file is empty
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        fprintf(fp, "size,implementation,time,gflops,bytes_accessed,operational_intensity\n");
    }
    
    // Write data
    fprintf(fp, "%d,serial,%.6f,%.2f,%.0f,%.2f\n", 
            size, avg_serial.time, avg_serial.gflops, 
            avg_serial.bytes_accessed, avg_serial.operational_intensity);
    fprintf(fp, "%d,vectorized,%.6f,%.2f,%.0f,%.2f\n", 
            size, avg_vectorized.time, avg_vectorized.gflops, 
            avg_vectorized.bytes_accessed, avg_vectorized.operational_intensity);
    
    fclose(fp);
    
    // Print summary
    printf("\nMatrix Size: %d x %d\n", size, size);
    printf("Serial Implementation:\n");
    printf("  Time: %.4f seconds\n", avg_serial.time);
    printf("  Performance: %.2f GFLOPS\n", avg_serial.gflops);
    printf("  Operational Intensity: %.2f FLOPS/byte\n", avg_serial.operational_intensity);
    printf("\nVectorized Implementation:\n");
    printf("  Time: %.4f seconds\n", avg_vectorized.time);
    printf("  Performance: %.2f GFLOPS\n", avg_vectorized.gflops);
    printf("  Operational Intensity: %.2f FLOPS/byte\n", avg_vectorized.operational_intensity);
    printf("Speedup: %.2fx\n", avg_serial.time/avg_vectorized.time);
    
    free_matrix(A, size);
    free_matrix(B, size);
    free_matrix(C, size);
}

int main() {
    // Remove old performance data file
    remove("matrix_performance.csv");
    
    // Test with different matrix sizes
    int sizes[] = {256, 512, 1024, 100, 356, 822};
    
    printf("Collecting performance data...\n");
    for(int i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        printf("\nRun %d: Testing %dx%d matrix\n", i+1, sizes[i], sizes[i]);
        save_performance_data(sizes[i]);
    }
    
    printf("\nPerformance data saved to matrix_performance.csv\n");
    return 0;
}