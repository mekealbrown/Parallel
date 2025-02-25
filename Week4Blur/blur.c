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
	2. Compute inital sum - pixels 0 to kernel_size-1
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

  // ============ Horizontal Pass ==================
  for(int y = 0; y + 7 < height; y += 8){
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

    int sums_r[8], sums_g[8], sums_b[8];
    for(int x = 0; x < width; x++){
      // average and store current window
			// TODO: reciprical multiplication to speed up division??
      _mm256_storeu_si256((__m256i*)sums_r, result_r);
      _mm256_storeu_si256((__m256i*)sums_g, result_g);
      _mm256_storeu_si256((__m256i*)sums_b, result_b);
      for(int i = 0; i < 8; i++){
        output[y + i][x].r = (unsigned char)(sums_r[i] / kernel_size);
        output[y + i][x].g = (unsigned char)(sums_g[i] / kernel_size);
        output[y + i][x].b = (unsigned char)(sums_b[i] / kernel_size);
      }

      // slide window with right boundary clamping
      if(x < width - 1){
        int left_col = x;  // subtract pixel just before current window
        int right_col = x + kernel_size;  // add next pixel entering window

        // subtract left_col if valid
        if(left_col >= 0){
          // get our 8 Pixels
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
        }

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
        // add to accumulators
        result_r = _mm256_add_epi32(result_r, _mm256_and_si256(reg_right, _mm256_set1_epi32(0xFF)));
        result_g = _mm256_add_epi32(result_g, _mm256_and_si256(_mm256_srli_epi32(reg_right, 8), _mm256_set1_epi32(0xFF)));
        result_b = _mm256_add_epi32(result_b, _mm256_and_si256(_mm256_srli_epi32(reg_right, 16), _mm256_set1_epi32(0xFF)));
      }
    }
  }
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

	//============================= TESTING =================================
	// unoptimized
	struct timeval start, end;
	gettimeofday(&start, NULL);
  blur_image(input_image, output_image, width, height, kernel_size);
	gettimeofday(&end, NULL);

	double seconds = end.tv_sec - start.tv_sec;
  double microseconds = end.tv_usec - start.tv_usec;
  double total_microseconds_un = seconds * 1000000 + microseconds;
	printf("Unoptimized Blur: %2f\n", total_microseconds_un);

	// optimized
	gettimeofday(&start, NULL);
	blur_image_opt_simd(input_image, output_image, temp, width, height, kernel_size);
	//blur_image_opt(input_image, output_image, temp, width, height, 20);
	gettimeofday(&end, NULL);

	seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  double total_microseconds_op = seconds * 1000000 + microseconds;
	printf("Optimized Blur:   %2f\n", total_microseconds_op);

	printf("Speedup: %2fx\n", total_microseconds_un / total_microseconds_op);
	//============================ END TESTING ===============================

	// DOES IT COMPILE AND RUN?
	


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
