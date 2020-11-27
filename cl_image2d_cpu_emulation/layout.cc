#include <vector>
#include <string>
#include "matrix.h"

int main() {

  std::vector<size_t> input_shape(4);
  input_shape[0] = 1;
  input_shape[1] = 64;
  input_shape[2] = 64;
  input_shape[3] = 64;//42;
  size_t N = input_shape[0];
  size_t C = input_shape[1];
  size_t H = input_shape[2];
  size_t W = input_shape[3];
  size_t buffer_size = N * C * H * W;
  float *p = init_matrix(input_shape, -1, "NCHW Layout");
  //print_matrix(input_shape, p, "NCHW Layout");

  std::vector<size_t> image_shape = tensor_shape_to_image_shape(input_shape);
  size_t image_size = image_shape[0] * image_shape[1] * 4;
  size_t image_width = image_shape[0];  // (C+3)/4*W
  size_t w_block = image_width / W;  // (C+3)/4
  float *image = init_matrix(image_shape, 0, "image");
  //print_matrix(image_shape, image, "image");

#if 1
  size_t i0 = 0;
  for (size_t n = 0; n < N; n++) {
    for (size_t c = 0; c < w_block * 4; c++) { // c < (C+3)
      size_t i1 = i0 + (c / 4) * W;  // i1 = i0 + (c/4)*W
      for (size_t h = 0; h < H; h++) {
        size_t i2 = (i1 << 2) + c % 4;  // i2 = i1*4 + c%4
        for (size_t w = 0; w < W; w++) {
          if (c < C) {
            // size_t x = (n * image_width * H + h * image_width + (c / 4) * W + w) * 4 +
            // (c % 4);
            image[i2] = *p;//Float2Half(*p);
            i2 += 4;
            p++;
          } else {
            image[i2] = 0.f;//Float2Half(0.f);
            i2 += 4;
          }
        }
        i1 += image_width;
      }
    }
    i0 += image_width * H;  // i0+=(C+3)/4*W*H. consider batch size: n
  }
#else
  size_t i0 = 0;
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C + 3; ++c) {
      size_t i1 = i0 + (c / 4) * w;
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          size_t nchw_idx = n * 
          image[i0] = nchw[nchw_idx];
        }
      }
    }
    i0 += image_width * image_height;
  }
#endif

  //print_matrix(image_shape, image, "filled image");
  for (size_t h = 0; h < 10; ++h) {
    for (size_t w = 0; w < 10; ++w) {
      print_image_pixel(image, image_shape[0], image_shape[1], w, h);
    }
  }
  return 0;
}
