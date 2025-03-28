#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // get rank of this process
	MPI_Comm_size(MPI_COMM_WORLD, &size);  // get total number of processes

	if(size < 2){
		if(rank == 0){
			printf("Needs 2 or more processes.\n");
		}
		MPI_Finalize();
		return 1;
	}

	if(rank == 0){
		for(int i = 1; i < size; i++){
			int val = i;
			MPI_Send(&val, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
	} else{
		int ret_val;
		MPI_Recv(&ret_val, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("Received %d from process 0\n", ret_val);
	}



	MPI_Finalize();
}