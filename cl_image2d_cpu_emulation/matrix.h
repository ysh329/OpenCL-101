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

void print_image_pixel(float* image, size_t image_width, size_t image_height, size_t x, size_t y, std::string name="") {
  std::cout << "============= print_image_pixel " << name << " ================" << std::endl;
  std::cout << name << "(" << x << "," << y << "):[";
  for (size_t pidx = 0; pidx < 4; ++pidx) {
    size_t idx = y * image_width * 4 + x * 4 + pidx;
    std::cout << image[idx];
    if (pidx <= 2) std::cout << " ";
  }
  std::cout << "] \n";//std::endl;
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
