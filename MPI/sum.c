#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size, local_sum = 0, global_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process computes a local sum
    local_sum = rank + 1; // Example: process 0 -> 1, process 1 -> 2, etc.

    // Reduce all local sums to global_sum on rank 0
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total sum = %d\n", global_sum);
    }

    MPI_Finalize();
    return 0;
}