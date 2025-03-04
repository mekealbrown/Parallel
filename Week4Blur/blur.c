#include <stdio.h>
#include <stdlib.h>
#include <smmintrin.h>
#include <sys/time.h>
#include <immintrin.h>

// STB Image setup
#define STB_IMAGE_IMPLEMENTATION
#include "Include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Include/stb_image_write.h"


typedef struct {
    unsigned char r, g, b, padding; // ensures 4-byte alignment
} Pixel_t;


Pixel_t** create_Pixel_array(unsigned char* data, int width, int height, int channels)
{
  Pixel_t** image = (Pixel_t**)malloc(height * sizeof(Pixel_t*));
  for(int i = 0; i < height; i++){
    image[i] = (Pixel_t*)malloc(width * sizeof(Pixel_t));
    for(int j = 0; j < width; j++){
      int idx = (i * width + j) * channels;
      image[i][j].r = data[idx];
      image[i][j].g = data[idx + 1];
      image[i][j].b = data[idx + 2];
    }
  }
  return image;
}

unsigned char* create_output_array(Pixel_t** image, int width, int height, int channels)
{
  unsigned char* output = (unsigned char*)malloc(width * height * channels);
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      int idx = (i * width + j) * channels;
      output[idx] = image[i][j].r;
      output[idx + 1] = image[i][j].g;
      output[idx + 2] = image[i][j].b;
    }
  }
  return output;
}

// w.o.w... 4 nested for loops
void blur_image(Pixel_t** image, Pixel_t** output, int width, int height, int kernel_size)
{
  int radius = kernel_size / 2;

  for(int y = 0; y < height; y++){
    for(int x = 0; x < width; x++){
      int sum_r = 0, sum_g = 0, sum_b = 0, count = 0;

      for(int ky = -radius; ky <= radius; ky++){
        for(int kx = -radius; kx <= radius; kx++){
          int ny = y + ky;
          int nx = x + kx;

					// if it's in bounds
          if(ny >= 0 && ny < height && nx >= 0 && nx < width){
            sum_r += image[ny][nx].r;
            sum_g += image[ny][nx].g;
            sum_b += image[ny][nx].b;
            count++;
          }
        }
      }

      output[y][x].r = sum_r / count;
      output[y][x].g = sum_g / count;
      output[y][x].b = sum_b / count;
    }
  }
}

void blur_image_opt(Pixel_t** image, Pixel_t** output, Pixel_t** temp, int width, int height, int kernel_size)
{
  int radius = kernel_size / 2;

  // horizontal pass
  for(int y = 0; y < height; y++){
    int sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;

		// initial pixel
    for(int kx = -radius; kx <= radius; kx++){
      if (kx >= 0 && kx < width) {
        sum_r += image[y][kx].r;
        sum_g += image[y][kx].g;
        sum_b += image[y][kx].b;
        count++;
      }
    }

    for(int x = 0; x < width; x++){
      temp[y][x].r = sum_r / count;
      temp[y][x].g = sum_g / count;
      temp[y][x].b = sum_b / count;

      // remove the leftmost pixel
      int left = x - radius;
      if(left >= 0){
        sum_r -= image[y][left].r;
        sum_g -= image[y][left].g;
        sum_b -= image[y][left].b;
        count--;
      }
      // add the rightmost pixel
      int right = x + radius + 1;
      if(right < width){
        sum_r += image[y][right].r;
        sum_g += image[y][right].g;
        sum_b += image[y][right].b;
        count++;
      }
    }
  }

  // vertical pass
  for(int x = 0; x < width; x++){
    int sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;

		// initial pixel
    for(int ky = -radius; ky <= radius; ky++){
      if(ky >= 0 && ky < height){
        sum_r += temp[ky][x].r;
        sum_g += temp[ky][x].g;
        sum_b += temp[ky][x].b;
        count++;
      }
    }

    for(int y = 0; y < height; y++){
      output[y][x].r = sum_r / count;
      output[y][x].g = sum_g / count;
      output[y][x].b = sum_b / count;

      // remove top pixel
      int top = y - radius;
      if(top >= 0){
        sum_r -= temp[top][x].r;
        sum_g -= temp[top][x].g;
        sum_b -= temp[top][x].b;
        count--;
      }
      // add bottom pixel
      int bottom = y + radius + 1;
      if(bottom < height){
        sum_r += temp[bottom][x].r;
        sum_g += temp[bottom][x].g;
        sum_b += temp[bottom][x].b;
        count++;
      }
    }
  }
}

/*
	1. Initialize r,g,b accumulators. Holds sums for each window r,g,b elements
	2. Compute inital sum -- pixels 0 to kernel_size-1
	3. store average for window
	4. "slide window"
	5. subtract pixel just outside the left side of the window


	Window:
	_______________
	|!|_|_|_|_|_|_|   <-- blur (0, 0)
	|_____|           <-- kernel_size = 3

	Shift window over 1 spot
	_______________
	|_|!|_|_|_|_|_|   <-- blur (1, 0)
	  |_____|         <-- kernel_size = 3


	Do this for 8 rows(horizontal pass) at a time with SIMD!
*/
void blur_image_opt_simd(Pixel_t** image, Pixel_t** output, Pixel_t** temp, int width, int height, int kernel_size)
{

	float recip = 1.0f / kernel_size;
	__m256 recip_kern_vec = _mm256_set1_ps(recip); // fill vector with reciprical -- ps is packed single-precision, good for 8 floats

  // ============ Horizontal Pass ==================
	int y;
  for(y = 0; y + 7 < height; y += 8){
    // sum accumulator registers for r,g,b
    __m256i result_r = _mm256_setzero_si256();
    __m256i result_g = _mm256_setzero_si256();
    __m256i result_b = _mm256_setzero_si256();

    // Initial window: sum pixels 0 to kernel_size-1
    for(int kx = 0; kx < kernel_size; kx++){
      // each Pixel_t is 32bit, same as int
      // cast the pointer to image[y][kx] to int pointer, then get that value
      // store 8 of these(do 8 Pixels at a time)
      __m256i reg = _mm256_setr_epi32(
        *(int*)&image[y + 0][kx],
        *(int*)&image[y + 1][kx],
        *(int*)&image[y + 2][kx],
        *(int*)&image[y + 3][kx],
        *(int*)&image[y + 4][kx],
        *(int*)&image[y + 5][kx],
        *(int*)&image[y + 6][kx],
        *(int*)&image[y + 7][kx]
      );
			// add to accumulators
      result_r = _mm256_add_epi32(result_r, _mm256_and_si256(reg, _mm256_set1_epi32(0xFF)));
      result_g = _mm256_add_epi32(result_g, _mm256_and_si256(_mm256_srli_epi32(reg, 8), _mm256_set1_epi32(0xFF)));
      result_b = _mm256_add_epi32(result_b, _mm256_and_si256(_mm256_srli_epi32(reg, 16), _mm256_set1_epi32(0xFF)));
    }

    for(int x = 0; x < width; x++){
      // average and store current window
			// convert to float, multiply by reciprical, convert back to __m256i, store
			// syntax notes:
			//  - cvtps_epi32 = convert packed single precision floats to 8 32bit ints
			//  - cvtepi32_ps = convert 8 32bit ints to 8 packed single precision floats
			//  - mul_ps      = multiply two registers holding 8 packed single precision floats
			__m256i avg_r = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(result_r), recip_kern_vec));
			__m256i avg_g = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(result_g), recip_kern_vec));
			__m256i avg_b = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(result_b), recip_kern_vec));


      for(int i = 0; i < 8; i++){
			  temp[y + i][x].r = (unsigned char)_mm256_extract_epi32(avg_r, i);
			  temp[y + i][x].g = (unsigned char)_mm256_extract_epi32(avg_g, i);
			  temp[y + i][x].b = (unsigned char)_mm256_extract_epi32(avg_b, i);
			}

      // slide window with right boundary clamping
      if(x < width - 1){
        int left_col = x;  // subtract pixel just before current window
        int right_col = x + kernel_size;  // add next pixel entering window

        // get the 8 Pixels
        __m256i reg_left = _mm256_setr_epi32(
          *(int*)&image[y + 0][left_col],
          *(int*)&image[y + 1][left_col],
          *(int*)&image[y + 2][left_col],
          *(int*)&image[y + 3][left_col],
          *(int*)&image[y + 4][left_col],
          *(int*)&image[y + 5][left_col],
          *(int*)&image[y + 6][left_col],
          *(int*)&image[y + 7][left_col]
        );
        // subtract the r,g,b elements from the accumulators
        result_r = _mm256_sub_epi32(result_r, _mm256_and_si256(reg_left, _mm256_set1_epi32(0xFF)));
        result_g = _mm256_sub_epi32(result_g, _mm256_and_si256(_mm256_srli_epi32(reg_left, 8), _mm256_set1_epi32(0xFF)));
        result_b = _mm256_sub_epi32(result_b, _mm256_and_si256(_mm256_srli_epi32(reg_left, 16), _mm256_set1_epi32(0xFF)));

        // clamp right_col to width - 1 if it exceeds the image boundary
        if(right_col >= width){
          right_col = width - 1;  // reuse the last valid column
        }

        // get the 8 in front of last window
				// add the right_col
        __m256i reg_right = _mm256_setr_epi32(
          *(int*)&image[y + 0][right_col],
          *(int*)&image[y + 1][right_col],
          *(int*)&image[y + 2][right_col],
          *(int*)&image[y + 3][right_col],
          *(int*)&image[y + 4][right_col],
          *(int*)&image[y + 5][right_col],
          *(int*)&image[y + 6][right_col],
          *(int*)&image[y + 7][right_col]
        );
        // add to accumulators, mask off parts we dont want, shift right to get next channel
        result_r = _mm256_add_epi32(result_r, _mm256_and_si256(reg_right, _mm256_set1_epi32(0xFF)));
        result_g = _mm256_add_epi32(result_g, _mm256_and_si256(_mm256_srli_epi32(reg_right, 8), _mm256_set1_epi32(0xFF)));
        result_b = _mm256_add_epi32(result_b, _mm256_and_si256(_mm256_srli_epi32(reg_right, 16), _mm256_set1_epi32(0xFF)));
      }
    }
  }

	// handle remaining Pixels if not divisible by 8
	for(y; y < height; y++){
    int sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;

		// initial pixel
    for(int kx = 0; kx < kernel_size; kx++){
      if (kx >= 0 && kx < width) {
        sum_r += image[y][kx].r;
        sum_g += image[y][kx].g;
        sum_b += image[y][kx].b;
        count++;
      }
    }

    for(int x = 0; x < width; x++){
      temp[y][x].r = sum_r / count;
      temp[y][x].g = sum_g / count;
      temp[y][x].b = sum_b / count;

      // remove the leftmost pixel
      sum_r -= image[y][x].r;
      sum_g -= image[y][x].g;
      sum_b -= image[y][x].b;
      count--;

      // add the rightmost pixel
      int right = x + kernel_size;
      if(right < width){
        sum_r += image[y][right].r;
        sum_g += image[y][right].g;
        sum_b += image[y][right].b;
        count++;
      }
    }
  }

	//=================== Vertical Pass =========================
	int x;
	for(x = 0; x + 7 < width; x += 8){
		__m256i result_r = _mm256_setzero_si256();
		__m256i result_g = _mm256_setzero_si256();
		__m256i result_b = _mm256_setzero_si256();

		for(int ky = 0; ky < kernel_size; ky++){

			__m256i reg = _mm256_setr_epi32( // 256 bit reg, set reverse, extended packed ints(32bits)
				*(int*)&temp[ky][x], *(int*)&temp[ky][x + 1],
				*(int*)&temp[ky][x + 2], *(int*)&temp[ky][x + 3],
				*(int*)&temp[ky][x + 4], *(int*)&temp[ky][x + 5],
				*(int*)&temp[ky][x + 6], *(int*)&temp[ky][x + 7]
			);
			result_r = _mm256_add_epi32(result_r, _mm256_and_si256(reg, _mm256_set1_epi32(0xFF)));
			result_g = _mm256_add_epi32(result_g, _mm256_and_si256(_mm256_srli_epi32(reg, 8), _mm256_set1_epi32(0xFF)));
			result_b = _mm256_add_epi32(result_b, _mm256_and_si256(_mm256_srli_epi32(reg, 16), _mm256_set1_epi32(0xFF)));
		}
		// we now have the sum for the first window
		// now start loop to slide window, starting with storing current average in output
		for(int y = 0; y < height; y++){
			__m256i avg_r = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(result_r), recip_kern_vec));
			__m256i avg_g = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(result_g), recip_kern_vec));
			__m256i avg_b = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(result_b), recip_kern_vec));

			for(int i = 0; i < 8; i++){
				output[y][x + i].r = _mm256_extract_epi32(avg_r, i);
				output[y][x + i].g = _mm256_extract_epi32(avg_g, i);
				output[y][x + i].b = _mm256_extract_epi32(avg_b, i);
			}

			if(y < height - 1){
				int bottom = y; int top = y + kernel_size;

				__m256i reg_bottom = _mm256_setr_epi32(
          *(int*)&temp[bottom][x + 0],
          *(int*)&temp[bottom][x + 1],
          *(int*)&temp[bottom][x + 2],
          *(int*)&temp[bottom][x + 3],
          *(int*)&temp[bottom][x + 4],
          *(int*)&temp[bottom][x + 5],
          *(int*)&temp[bottom][x + 6],
          *(int*)&temp[bottom][x + 7]
        );

				// subtract the r,g,b elements from the accumulators3 * width * height * sizeof(Pixel_t)
        result_r = _mm256_sub_epi32(result_r, _mm256_and_si256(reg_bottom, _mm256_set1_epi32(0xFF)));
        result_g = _mm256_sub_epi32(result_g, _mm256_and_si256(_mm256_srli_epi32(reg_bottom, 8), _mm256_set1_epi32(0xFF)));
        result_b = _mm256_sub_epi32(result_b, _mm256_and_si256(_mm256_srli_epi32(reg_bottom, 16), _mm256_set1_epi32(0xFF)));

        // clamp top to height - 1 if it exceeds the image boundary
        if(top >= height){
          top = height - 1;  // reuse the last valid row
        }
        // get the 8 in front of last window
				// add the right_col
        __m256i reg_top = _mm256_setr_epi32(
          *(int*)&temp[top][x + 0],
          *(int*)&temp[top][x + 1],
          *(int*)&temp[top][x + 2],
          *(int*)&temp[top][x + 3],
          *(int*)&temp[top][x + 4],
          *(int*)&temp[top][x + 5],
          *(int*)&temp[top][x + 6],
          *(int*)&temp[top][x + 7]
        );
        // add to accumulators
        result_r = _mm256_add_epi32(result_r, _mm256_and_si256(reg_top, _mm256_set1_epi32(0xFF)));
        result_g = _mm256_add_epi32(result_g, _mm256_and_si256(_mm256_srli_epi32(reg_top, 8), _mm256_set1_epi32(0xFF)));
        result_b = _mm256_add_epi32(result_b, _mm256_and_si256(_mm256_srli_epi32(reg_top, 16), _mm256_set1_epi32(0xFF)));				
			}
		}
	}

		// handle remaining Pixels if not divisible by 8
	for(x; x < width; x++){
    int sum_r = 0, sum_g = 0, sum_b = 0;
    int count = 0;

		// initial pixel
    for(int ky = 0; ky < kernel_size; ky++){
      if(ky >= 0 && ky < height){
        sum_r += temp[ky][x].r;
        sum_g += temp[ky][x].g;
        sum_b += temp[ky][x].b;
        count++;
      }
    }

    for(int y = 0; y < height; y++){
      output[y][x].r = sum_r / count;
      output[y][x].g = sum_g / count;
      output[y][x].b = sum_b / count;

      // remove top pixel
        sum_r -= temp[y][x].r;
        sum_g -= temp[y][x].g;
        sum_b -= temp[y][x].b;
        count--;

      // add bottom pixel
      int bottom = y + kernel_size;
      if(bottom < height){
        sum_r += temp[bottom][x].r;
        sum_g += temp[bottom][x].g;
        sum_b += temp[bottom][x].b;
        count++;
      }
  	}
	}
}

void benchmark_cache_limit(int kernel_size)
{
  const int max_size = 8192;
  const int runs = 5; 
  const int warmup_runs = 2;

  FILE *file = fopen("result.txt", "w"); // Open file for writing
  if(!file){
    printf("Error: Could not open result.txt for writing\n");
    return;
  }
  const char *header = "=== Cache Limit Benchmark for blur_image_opt_simd ===\n"
                       "+------------+---------------+---------------+---------------+\n"
                       "| Size       | Time (ms)     | Memory (KB)   | Speedup       |\n"
                       "+------------+---------------+---------------+---------------+\n";
  fprintf(file, "%s", header);
  double prev_time_ms = 0.0;
  int prev_size = 0;
  // Initial coarse steps up to 1024
  for(int size = 64; size <= 1024; size *= 2){
    Pixel_t** image = (Pixel_t**)malloc(size * sizeof(Pixel_t*));
    Pixel_t** output = (Pixel_t**)malloc(size * sizeof(Pixel_t*));
    Pixel_t** temp = (Pixel_t**)malloc(size * sizeof(Pixel_t*));

    for(int i = 0; i < size; i++){
      image[i] = (Pixel_t*)malloc(size * sizeof(Pixel_t));
      output[i] = (Pixel_t*)malloc(size * sizeof(Pixel_t));
      temp[i] = (Pixel_t*)malloc(size * sizeof(Pixel_t));
    }

    // dummy data
    for(int y = 0; y < size; y++){
      for(int x = 0; x < size; x++){
        image[y][x].r = (unsigned char)x;
        image[y][x].g = (unsigned char)y;
        image[y][x].b = 0;
        image[y][x].padding = 0;
      }
    }

    // Warm-up runs
    for(int i = 0; i < warmup_runs; i++){
      blur_image_opt_simd(image, output, temp, size, size, kernel_size);
    }

    // Timed runs
    struct timeval start, end;
    double time_us = 0;
    for(int i = 0; i < runs; i++){
      gettimeofday(&start, NULL);
      blur_image_opt_simd(image, output, temp, size, size, kernel_size);
      gettimeofday(&end, NULL);
      double seconds = end.tv_sec - start.tv_sec;
      double microseconds = end.tv_usec - start.tv_usec;
      time_us += seconds * 1000000 + microseconds;
    }

    double time_ms = (time_us / runs) / 1000.0;

    long long memory_bytes = 3LL * size * size * sizeof(Pixel_t);
    double memory_kb = memory_bytes / 1024.0;

    double speedup = 0.0;
    if(prev_size != 0){
      double pixel_ratio = (double)(size * size) / (prev_size * prev_size);
      speedup = time_ms / (prev_time_ms * pixel_ratio);
    } else{
      speedup = 1.0; // first entry, no comparison
    }

    char line[100];
    snprintf(line, sizeof(line), "| %4dx%-4d | %11.2f | %11.0f | %11.2f |\n", 
             size, size, time_ms, memory_kb, speedup);
    fprintf(file, "%s", line);

    prev_time_ms = time_ms;
    prev_size = size;

    for(int i = 0; i < size; i++){
      free(image[i]);
      free(output[i]);
      free(temp[i]);
    }
    free(image);
    free(output);
    free(temp);
  }

  // Finer granularity between 1024 and 2048(right where cache limit should be hit)
  for(int size = 1280; size <= 2048; size += 256){
    Pixel_t** image = (Pixel_t**)malloc(size * sizeof(Pixel_t*));
    Pixel_t** output = (Pixel_t**)malloc(size * sizeof(Pixel_t*));
    Pixel_t** temp = (Pixel_t**)malloc(size * sizeof(Pixel_t*));
		
    for(int i = 0; i < size; i++){
      image[i] = (Pixel_t*)malloc(size * sizeof(Pixel_t));
      output[i] = (Pixel_t*)malloc(size * sizeof(Pixel_t));
      temp[i] = (Pixel_t*)malloc(size * sizeof(Pixel_t));
    }

    for(int y = 0; y < size; y++){
        for(int x = 0; x < size; x++){
          image[y][x].r = (unsigned char)x;
          image[y][x].g = (unsigned char)y;
          image[y][x].b = 0;
          image[y][x].padding = 0;
        }
    }
    for(int i = 0; i < warmup_runs; i++){
      blur_image_opt_simd(image, output, temp, size, size, kernel_size);
    }

    struct timeval start, end;
    double time_us = 0;
    for(int i = 0; i < runs; i++){
      gettimeofday(&start, NULL);
      blur_image_opt_simd(image, output, temp, size, size, kernel_size);
      gettimeofday(&end, NULL);
      double seconds = end.tv_sec - start.tv_sec;
      double microseconds = end.tv_usec - start.tv_usec;
      time_us += seconds * 1000000 + microseconds;
    }

    double time_ms = (time_us / runs) / 1000.0;
    long long memory_bytes = 3LL * size * size * sizeof(Pixel_t);
    double memory_kb = memory_bytes / 1024.0;
    double pixel_ratio = (double)(size * size) / (prev_size * prev_size);
    double speedup = time_ms / (prev_time_ms * pixel_ratio);
    char line[100];
    snprintf(line, sizeof(line), "| %4dx%-4d | %11.2f | %11.0f | %11.2f |\n", 
             size, size, time_ms, memory_kb, speedup);
    fprintf(file, "%s", line);
    prev_time_ms = time_ms;
    prev_size = size;
    for(int i = 0; i < size; i++){
      free(image[i]);
      free(output[i]);
      free(temp[i]);
    }
    free(image);
    free(output);
    free(temp);
  }

  // Coarse steps beyond 2048
  for(int size = 4096; size <= max_size; size *= 2){
      Pixel_t** image = (Pixel_t**)malloc(size * sizeof(Pixel_t*));
      Pixel_t** output = (Pixel_t**)malloc(size * sizeof(Pixel_t*));
      Pixel_t** temp = (Pixel_t**)malloc(size * sizeof(Pixel_t*));

      for(int i = 0; i < size; i++){
        image[i] = (Pixel_t*)malloc(size * sizeof(Pixel_t));
        output[i] = (Pixel_t*)malloc(size * sizeof(Pixel_t));
        temp[i] = (Pixel_t*)malloc(size * sizeof(Pixel_t));
      }

      for(int y = 0; y < size; y++){
          for(int x = 0; x < size; x++){
            image[y][x].r = (unsigned char)x;
            image[y][x].g = (unsigned char)y;
            image[y][x].b = 0;
            image[y][x].padding = 0;
          }
      }
      for(int i = 0; i < warmup_runs; i++){
        blur_image_opt_simd(image, output, temp, size, size, kernel_size);
      }

      struct timeval start, end;
      double time_us = 0;
      for(int i = 0; i < runs; i++){
        gettimeofday(&start, NULL);
        blur_image_opt_simd(image, output, temp, size, size, kernel_size);
        gettimeofday(&end, NULL);
        double seconds = end.tv_sec - start.tv_sec;
        double microseconds = end.tv_usec - start.tv_usec;
        time_us += seconds * 1000000 + microseconds;
      }

      double time_ms = (time_us / runs) / 1000.0;
      long long memory_bytes = 3LL * size * size * sizeof(Pixel_t);
      double memory_kb = memory_bytes / 1024.0;
      double pixel_ratio = (double)(size * size) / (prev_size * prev_size);
      double speedup = time_ms / (prev_time_ms * pixel_ratio);
      char line[100];
      snprintf(line, sizeof(line), "| %4dx%-4d | %11.2f | %11.0f | %11.2f |\n", 
               size, size, time_ms, memory_kb, speedup);
      fprintf(file, "%s", line);
      prev_time_ms = time_ms;
      prev_size = size;
      for (int i = 0; i < size; i++) {
          free(image[i]);
          free(output[i]);
          free(temp[i]);
      }
      free(image);
      free(output);
      free(temp);
  }

  const char *footer = "+------------+---------------+---------------+---------------+\n";
  fprintf(file, "%s", footer);
  fclose(file);
}

void test_blur_performance
(Pixel_t** input_image, Pixel_t** output_image, Pixel_t** temp, int width, int height, int kernel_size)
{
  struct timeval start, end;
  double seconds, microseconds, time_unopt = 0, time_opt = 0, time_simd = 0;
  const int warmup_runs = 2;
  const int test_runs = 5;  // number of timed runs

  // get that data in the cache
  for(int i = 0; i < warmup_runs; i++){
    blur_image(input_image, output_image, width, height, kernel_size);
    blur_image_opt(input_image, output_image, temp, width, height, kernel_size);
    blur_image_opt_simd(input_image, output_image, temp, width, height, kernel_size);
  }

	// ifdef blocks to get specific function results if desired
	#ifdef UNOP
  	// timed runs for naive version
  	for(int i = 0; i < test_runs; i++){
  	  gettimeofday(&start, NULL);
  	  blur_image(input_image, output_image, width, height, kernel_size);
  	  gettimeofday(&end, NULL);
  	  seconds = end.tv_sec - start.tv_sec;
  	  microseconds = end.tv_usec - start.tv_usec;
  	  time_unopt += seconds * 1000000 + microseconds;
  	}
  	time_unopt /= test_runs;
	#endif

	#ifdef OP
  	// timed runs for optimized version (non-SIMD)
  	for(int i = 0; i < test_runs; i++){
  	  gettimeofday(&start, NULL);
  	  blur_image_opt(input_image, output_image, temp, width, height, kernel_size);
  	  gettimeofday(&end, NULL);
  	  seconds = end.tv_sec - start.tv_sec;
  	  microseconds = end.tv_usec - start.tv_usec;
  	  time_opt += seconds * 1000000 + microseconds;
  	}
  	time_opt /= test_runs;
	#endif

	#ifdef OP_SIMD
  	// Timed runs for Optimized (SIMD)
  	for(int i = 0; i < test_runs; i++){
  	  gettimeofday(&start, NULL);
  	  blur_image_opt_simd(input_image, output_image, temp, width, height, kernel_size);
  	  gettimeofday(&end, NULL);
  	  seconds = end.tv_sec - start.tv_sec;
  	  microseconds = end.tv_usec - start.tv_usec;
  	  time_simd += seconds * 1000000 + microseconds;
  	}
  	time_simd /= test_runs;
	#endif

  // convert to milliseconds
  double time_unopt_ms = time_unopt / 1000.0;
  double time_opt_ms = time_opt / 1000.0;
  double time_simd_ms = time_simd / 1000.0;
	
  printf("\n=== Blur Function Performance ===\n");
  printf("+----------------------+---------------+------------+\n");
  printf("| Version              | Time (ms)     | Speedup       |\n");
  printf("+----------------------+---------------+------------+\n");
  printf("| Unoptimized          | %11.2f | %11.2f |\n", time_unopt_ms, 1.0);
  printf("| Optimized (non-SIMD) | %11.2f | %11.2f |\n", time_opt_ms, time_unopt / time_opt);
  printf("| Optimized (SIMD)     | %11.2f | %11.2f |\n", time_simd_ms, time_unopt / time_simd);
  printf("+----------------------+---------------+------------+\n");
}

int main(int argc, char** argv)
{
  if(argc != 4){
    printf("Usage: %s <input_image> <output_image> <kernel_size>\n", argv[0]);
    return 1;
  }

	int kernel_size = atoi(argv[3]);
	
  // Load image
  int width, height, channels;
  unsigned char* data = stbi_load(argv[1], &width, &height, &channels, 0);
  if(!data){
    printf("Error loading image %s\n", argv[1]);
    return 1;
  }

  printf("Loaded image: %dx%d Pixel_ts, %d channels\n", width, height, channels);

  Pixel_t** input_image = create_Pixel_array(data, width, height, channels);
  stbi_image_free(data); // Free original

  // Create output image
  Pixel_t** output_image = (Pixel_t**)malloc(height * sizeof(Pixel_t*));
  for(int i = 0; i < height; i++){
    output_image[i] = (Pixel_t*)malloc(width * sizeof(Pixel_t));
  }

	// temporary storage to hold horizontal blur
	Pixel_t** temp =  (Pixel_t**)malloc(height * sizeof(Pixel_t*));
	for(int i = 0; i < height; i++){
    temp[i] = (Pixel_t*)malloc(width * sizeof(Pixel_t));
	}


	test_blur_performance(input_image, output_image, temp, width, height, kernel_size);

	benchmark_cache_limit(kernel_size);

  // Convert back to stb format
  unsigned char* output_data = create_output_array(output_image, width, height, channels);

  // Save image
  int success = 0;
  const char* ext = strrchr(argv[2], '.');
  if(!ext){
    printf("Output filename needs an extension\n");
  } else if(strcmp(ext, ".png") == 0){
    success = stbi_write_png(argv[2], width, height, channels, output_data, width * channels);
  } else if(strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0){
    success = stbi_write_jpg(argv[2], width, height, channels, output_data, 90); // 90 = quality
  } else if(strcmp(ext, ".bmp") == 0){
    success = stbi_write_bmp(argv[2], width, height, channels, output_data);
  } else{
    printf("Unsupported output format. Use .png, .jpg, or .bmp\n");
  }

  if(!success){
    printf("Error writing output image\n");
  }

  // Cleanup
  free(output_data);
  for(int i = 0; i < height; i++){
    free(input_image[i]);
    free(output_image[i]);
		free(temp[i]);
  }
  free(input_image);
  free(output_image);
	free(temp);

  return !success;
}
