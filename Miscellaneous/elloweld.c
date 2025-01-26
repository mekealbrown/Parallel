#include <stdio.h>
#include <omp.h>

int main()
{
	#pragma omp parallel
	{
		int nthreads, thread_id;
		nthreads = omp_get_num_threads();
		thread_id = omp_get_thread_num();
		printf("Goodbye slow serial world and Hello OpenMP\n");
		printf("I have %d thread(s) and my thread id is %d\n", nthreads, thread_id);
	}
	return 0;
}