#include <stdio.h>
#include <stdlib.h>

// STB Image setup
#define STB_IMAGE_IMPLEMENTATION
#include "Include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Include/stb_image_write.h"

typedef struct {
  unsigned char r, g, b;
} Pixel_t;


Pixel_t** create_Pixel_array(unsigned char* data, int width, int height, int channels) {
  Pixel_t** image = (Pixel_t**)malloc(height * sizeof(Pixel_t*));
  for (int i = 0; i < height; i++) {
    image[i] = (Pixel_t*)malloc(width * sizeof(Pixel_t));
    for (int j = 0; j < width; j++) {
      int idx = (i * width + j) * channels;
      image[i][j].r = data[idx];
      image[i][j].g = data[idx + 1];
      image[i][j].b = data[idx + 2];
    }
  }
  return image;
}

unsigned char* create_output_array(Pixel_t** image, int width, int height, int channels) {
  unsigned char* output = (unsigned char*)malloc(width * height * channels);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int idx = (i * width + j) * channels;
      output[idx] = image[i][j].r;
      output[idx + 1] = image[i][j].g;
      output[idx + 2] = image[i][j].b;
    }
  }
  return output;
}

void blurImage(Pixel_t** image, Pixel_t** output, int width, int height, int kernel_size) {
  int radius = kernel_size / 2;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int sum_r = 0, sum_g = 0, sum_b = 0, count = 0;

      for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
          int ny = y + ky;
          int nx = x + kx;

					// if it's in bounds
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
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

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <input_image> <output_image>\n", argv[0]);
    return 1;
  }

  // Load image
  int width, height, channels;
  unsigned char* data = stbi_load(argv[1], &width, &height, &channels, 0);
  if (!data) {
    printf("Error loading image %s\n", argv[1]);
    return 1;
  }

  printf("Loaded image: %dx%d Pixel_ts, %d channels\n", width, height, channels);

  Pixel_t** input_image = create_Pixel_array(data, width, height, channels);
  stbi_image_free(data); // Free original

  // Create output image
  Pixel_t** output_image = (Pixel_t**)malloc(height * sizeof(Pixel_t*));
  for (int i = 0; i < height; i++) {
    output_image[i] = (Pixel_t*)malloc(width * sizeof(Pixel_t));
  }

  blurImage(input_image, output_image, width, height, 10);

  // Convert back to stb format
  unsigned char* output_data = create_output_array(output_image, width, height, channels);

  // Save image
  int success = 0;
  const char* ext = strrchr(argv[2], '.');
  if (!ext) {
    printf("Output filename needs an extension\n");
  } else if (strcmp(ext, ".png") == 0) {
    success = stbi_write_png(argv[2], width, height, channels, output_data, width * channels);
  } else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0) {
    success = stbi_write_jpg(argv[2], width, height, channels, output_data, 90); // 90 = quality
  } else if (strcmp(ext, ".bmp") == 0) {
    success = stbi_write_bmp(argv[2], width, height, channels, output_data);
  } else {
    printf("Unsupported output format. Use .png, .jpg, or .bmp\n");
  }

  if (!success) {
    printf("Error writing output image\n");
  }

  // Cleanup
  free(output_data);
  for (int i = 0; i < height; i++) {
    free(input_image[i]);
    free(output_image[i]);
  }
  free(input_image);
  free(output_image);

  return !success;
}
