#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


void matrixMultiply(int* A, int* B, int* C, size_t m, size_t n, size_t p)
{
	for(size_t i = 0; i < m; i++){
		for(size_t j = 0; j < p; j++){
			C[i * p + j] = 0;
			for(size_t k = 0; k < n; k++){
				C[i * p + j] += A[i * n + k] * B[k * p + j];
			}
		}
	}
}

int* buildMatrix(size_t size, char random)
{
	int* matrix = (int*)malloc(size * sizeof(int));
	for(size_t i = 0; i < size; i++){
		matrix[i] = random ? (rand() % 100) : 0;
	}
	return matrix;
}

void printMatrix(int* matrix, size_t r, size_t c)
{
	for(size_t i = 0; i < r; i++){
		for(size_t j = 0; j < c; j++){
			printf("%d ", matrix[i * c + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

int main(int argc, char** argv)
{
	int size, rank;
	size_t m = 5; size_t n = 4; size_t p = 3;

	int* A; int* B; int* C;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	/*
		sendbuf → The buffer containing data to be scattered (only meaningful on the root process).

		sendcounts → An array specifying the number of elements each process receives.

		displs → An array specifying the displacement (offset) in sendbuf for each process.

		sendtype → The data type of elements in sendbuf.

		recvbuf → The buffer on each process where the received data will be stored.

		recvcount → The number of elements received by each process.

		recvtype → The data type of elements in recvbuf.

		root → The rank of the process that is scattering the data.

		comm → The MPI communicator.
	*/

	if(rank == 0){
		A = buildMatrix(m*n, 1);
		B = buildMatrix(n*p, 1);
		C = buildMatrix(m*p, 0);

		printMatrix(A, m, n);
		printf("\n");
		printMatrix(B, n, p);
		printf("\n");
		printMatrix(C, m, p);
		printf("\n");
	} else{
		B = (int*)malloc(n * p * sizeof(int));
	}

	int base_rows = m / size;
	int remain = m % size;
	int local_m = (rank < size - 1) ? base_rows : (base_rows + remain); //how many rows does this process have?

	// set up sendcounts and displs arrays for Scatterv and Gatherv
	// sendcounts is how many elements the process receives, displs is displacement(stride)
	int *sendcounts = (int*)malloc(size * sizeof(int));
	int *displs = (int*)malloc(size * sizeof(int));
	int offset = 0;
	for(int i = 0; i < size; i++){
		int rows = (i < size - 1) ? base_rows : (base_rows + remain);
		sendcounts[i] = rows * n;  // send rows * num of columns for A
		displs[i] = offset;
		offset += sendcounts[i]; // move pointer forward
	}

	int *sendcounts_c = (int*)malloc(size * sizeof(int));
	int *displs_c = (int*)malloc(size * sizeof(int));
	offset = 0;
	for(int i = 0; i < size; i++){
		int rows = (i < size - 1) ? base_rows : (base_rows + remain);
		sendcounts_c[i] = rows * p;
		displs_c[i] = offset;
		offset += sendcounts_c[i];
	}

	int *local_A = (int*)malloc(local_m * n * sizeof(int));
	int *local_C = (int*)malloc(local_m * p * sizeof(int));

	// send partitioned rows from A to each rank
	MPI_Scatterv(A, sendcounts, displs, MPI_INT, local_A, local_m * n, MPI_INT, 0, MPI_COMM_WORLD);
	// share B with every rank
	MPI_Bcast(B, n * p, MPI_INT, 0, MPI_COMM_WORLD);

	matrixMultiply(local_A, B, local_C, local_m, n, p);

	MPI_Gatherv(local_C, local_m * p, MPI_INT, C, sendcounts_c, displs_c, MPI_INT, 0, MPI_COMM_WORLD);

	if(rank == 0) {printMatrix(C, m, p);}

	if(rank == 0){
		free(A);
		free(C); 
	}
	free(B); free(sendcounts); free(sendcounts_c);
	free(displs); free(displs_c);
	MPI_Finalize();

	return 0;
}


