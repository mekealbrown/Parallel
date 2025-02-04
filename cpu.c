#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Matrix size
#define N 5000

void matrix_multiply_cpu(double *A, double *B, double *C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main() {
    double *A, *B, *C;
    A = (double*)malloc(N * N * sizeof(double));
    B = (double*)malloc(N * N * sizeof(double));
    C = (double*)malloc(N * N * sizeof(double));

    // Initialize matrices A and B with random values
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }

    // Timing the CPU version
    double start_time = omp_get_wtime();
    matrix_multiply_cpu(A, B, C);
    double end_time = omp_get_wtime();

    printf("CPU version took: %f seconds\n", end_time - start_time);

    free(A);
    free(B);
    free(C);

    return 0;
}
