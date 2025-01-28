#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double array_sum(double* arr, int size) {
    double total = 0.0;
    
    #pragma omp parallel for reduction(+:total)
    for(int i = 0; i < size; i++) {
        total += arr[i];
    }
    
    return total;
}

double dot_product(double* a, double* b, int size) {
    double result = 0.0;
    
    #pragma omp parallel for reduction(+:result)
    for(int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    
    return result;
}

void blur_image(double image[][8], double result[][8], int width, int height) {
	for(int i = 0; i < 8; i++){
		for(int j = 0; j < 8; j++){
			if(image[i][j] == 0) printf("  ");         // Black
        else if(image[i][j] < 85) printf(". ");    // Dark gray
        else if(image[i][j] < 170) printf("* ");   // Light gray
        else printf("# ");
		}
		printf("\n");
	}
    #pragma omp parallel for collapse(2)
    for(int y = 1; y < height-1; y++) {
        for(int x = 1; x < width-1; x++) {
						//blur by averaging neighbors
            result[y][x] = (
                image[y-1][x-1] + image[y-1][x] + image[y-1][x+1] +
                image[y][x-1]   + image[y][x]   + image[y][x+1] +
                image[y+1][x-1] + image[y+1][x] + image[y+1][x+1]
            ) / 9.0;
        }
    }
}

void find_primes(int start, int end, int* count) {
    #pragma omp parallel for reduction(+:count[0]) schedule(dynamic, 100)
    for(int n = start; n <= end; n++) {
        if(n <= 1) continue;
        
        char is_prime = 1;
        for(int i = 2; i * i <= n; i++) {
            if(n % i == 0) {
                is_prime = 0;
                break;
            }
        }
        
        if(is_prime) {
            count[0]++;
        }
    }
}

double estimate_pi(long num_points) {
    long points_inside = 0;

    // Parallelized loop with OpenMP
    #pragma omp parallel for reduction(+:points_inside)
    for(long i = 0; i < num_points; i++) {
        // Each thread should have a separate random state
        unsigned int seed = omp_get_thread_num() + i;

        // Generate random points (x, y) between 0 and 1
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;

        // Check if the point is inside the quarter circle (x^2 + y^2 <= 1)
        if(x * x + y * y <= 1.0) {
            points_inside++;
        }
    }
    // Estimate Pi
    return 4.0 * points_inside / num_points;
}

int main()
{
	double arr[10] = {1,2,3,4,5,6,7,8,9,10};
	printf("Array Sum: %f\n\n", array_sum(arr, 10));
	double arr2[10] = {2,4,6,8,10,12,14,16,18,20};
	printf("Dot Product: %f\n\n", dot_product(arr, arr2, 10));
	double heart_image[8][8] = {
    {0,   0,   0,   0,   0,   0,   0,   0},
    {0, 255, 255,   0,   0, 255, 255,   0},
    {0, 255, 255, 255, 255, 255, 255,   0},
    {0, 255, 255, 255, 255, 255, 255,   0},
    {0,   0, 255, 255, 255, 255,   0,   0},
    {0,   0,   0, 255, 255,   0,   0,   0},
    {0,   0,   0,   0,   0,   0,   0,   0},
    {0,   0,   0,   0,   0,   0,   0,   0}
	};
	double blured[8][8];
	blur_image(heart_image, blured, 8, 8);
	for(int i = 0; i < 8; i++){
		for(int j = 0; j < 8; j++){
			if(blured[i][j] == 0) printf("  ");
        else if(blured[i][j] < 85) printf(". ");
        else if(blured[i][j] < 170) printf("* ");
        else printf("# ");
		}
		printf("\n");
	}

	int count[1] = {0};
	find_primes(0, 15, count);
	printf("\nPrimes: %d\n", *count);

	long points = 1000000;
	printf("Pi: %f\n", estimate_pi(points));
}