#include <iostream>
#include <vector>

float* init_matrix(std::vector<size_t> shape, float value = -1, std::string name = "") {
  std::cout << "========= init " << name << " =========" << std::endl;
  if (shape.size() == 2) {
    std::cout << "image_width:" << shape[0] << std::endl; 
    std::cout << "image_height:" << shape[1] << std::endl;
  } else if (shape.size() == 4) {
    std::cout << "N:" << shape[0] << std::endl;
    std::cout << "C:" << shape[1] << std::endl;
    std::cout << "H:" << shape[2] << std::endl;
    std::cout << "W:" << shape[3] << std::endl;
  }

  size_t elem_size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    elem_size *= shape[i];
  }
  elem_size = shape.size() == 2 ? elem_size * 4 : elem_size;
  float* p = (float*)calloc(sizeof(float), elem_size);
  for (size_t i = 0; i < elem_size; ++i) {
    p[i] = value < 0 ? i + 1 : value;
  }
  std::cout << std::endl;
  return p;
}


void print_matrix(std::vector<size_t> shape, float* p, std::string name="") {
  bool is_buffer = shape.size() == 2 ? false : true;

  size_t image_width = is_buffer ? -1 : shape[0];
  size_t image_height = is_buffer ? -1 : shape[1];
  size_t elem_size = image_width * image_height * 4;  // 4 for depth RGBA

  size_t N = is_buffer ? shape[0] : -1;
  size_t C = is_buffer ? shape[1] : -1;
  size_t H = is_buffer ? shape[2] : -1;
  size_t W = is_buffer ? shape[3] : -1;
  elem_size = is_buffer ? N * C * H * W : -1;

  std::cout << "======== " << name << " =========" << std::endl;
  if (is_buffer) {
    for (size_t n = 0; n < N; ++n) {
      for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            size_t idx = n*C*H*W + c*H*W + h*W + w;
            std::cout << p[idx] << "\t";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;  // next channel
      }
    }
  } else {  // image
    for (size_t img_h = 0; img_h < image_height; ++img_h) {
      for (size_t img_w = 0; img_w < image_width; ++img_w) {
        for (size_t pidx = 0; pidx < 4; ++pidx) {
          size_t idx = img_h * image_width * 4 + img_w * 4 + pidx;
          std::cout << p[idx] << "\t";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
}


std::vector<size_t> tensor_shape_to_image_shape(std::vector<size_t> shape) {
  size_t N = shape[0];
  size_t C = shape[1];
  size_t H = shape[2];
  size_t W = shape[3];

  std::vector<size_t> image_dim(2);
  image_dim[0] = (C + 3) / 4 * W;  // image_width
  image_dim[1] = N * H;  // image_height

  return image_dim;
}

int main() {

  std::vector<size_t> input_shape(4);
  input_shape[0] = 1;
  input_shape[1] = 5;
  input_shape[2] = 3;
  input_shape[3] = 1;
  size_t N = input_shape[0];
  size_t C = input_shape[1];
  size_t H = input_shape[2];
  size_t W = input_shape[3];
  size_t buffer_size = N * C * H * W;
  float *p = init_matrix(input_shape, -1, "NCHW Layout");
  print_matrix(input_shape, p, "NCHW Layout");

  std::vector<size_t> image_shape = tensor_shape_to_image_shape(input_shape);
  size_t image_size = image_shape[0] * image_shape[1] * 4;
  size_t image_width = image_shape[0];  // (C+3)/4*W
  size_t w_block = image_width / W;  // (C+3)/4
  float *image = init_matrix(image_shape, 0, "image");
  print_matrix(image_shape, image, "image");

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

  print_matrix(image_shape, image, "filled image");
  return 0;
}
