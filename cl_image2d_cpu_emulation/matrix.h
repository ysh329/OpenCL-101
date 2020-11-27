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

float* init_image(float* tensor, std::vector<size_t> shape, std::vector<size_t> image_shape) {
  if (!tensor) {
    std::cout << "nullptr";
    exit(1);
  }

  
}

void free_image(float* image) {
  if (image) {
    free(image);
    image = NULL;
  }
}

void print_matrix(std::vector<size_t> shape, float* p, std::string name="") {
  std::cout << "========= print " << name << " ===========" << std::endl;
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

void print_image_pixel(float* image, size_t image_width, size_t image_height, size_t image_width_idx, size_t image_height_idx, std::string image_name="") {
  
  if (image_width_idx >= image_width || image_height_idx >= image_height) {
    std::cout << "image index [" << image_width_idx << "," << image_height_idx << "] error:" << std::endl;
    return;
  }

  size_t start_idx = 4 * image_width * image_height_idx + 4 * image_width_idx;
  if (image_name == "") {
    std::cout << "image";
  } else {
    std::cout << image_name;
  }
  std::cout << "[" << image_width_idx << "," << image_height_idx << "]:";
  for (size_t i = 0; i < 4; ++i) {
    std::cout << image[start_idx + i] ;
    if (i < 3) std::cout << ",";
  }
  std::cout << " buffer_idx:" << start_idx << "," << start_idx + 1 << "," << start_idx + 2 << "," << start_idx + 3;
  std::cout << std::endl;
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
