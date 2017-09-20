#include <stdio.h>
#include <stdlib.h>
#include "matop.h"

int main(void) {

	float *a, *b, *c, *d, *c_T;
	int width = 4;
	int height = 2;
	int data_size = width * height * sizeof(float);


	a = (float *) malloc (data_size);
	rand_mat(a, height*width, 4);
	printf("a:\n");
	print_mat(a, width, height);

	b = (float *) malloc (data_size);
	init_mat(b, height*width, 3);
	printf("b:\n");
	print_mat(b, width, height);

  for (int i=0; i<width*height; i++)
      printf("%.2f %.2f\n", a[i], b[i]);
	equal_vec(a, b, width*height);
  equal_mat(a, b, width, height);

	//printf("check a, c");
	//equal_vec(a, c, width*height);
  c = (float *) malloc (data_size);
  dotprod_mat(a, b, c, width*height);
  print_mat(c, width, height);

  printf("c_T:\n");
  c_T = (float *) malloc (data_size);
  transpose_mat(c, width, height, c_T);
  print_mat(c_T, height, width);

  float *z;
  float f[] = {1,2,3,
             4,5,6};
  float e[] = {1, 4,
             2, 5,
             3, 6};
  z = (float *) malloc (2*2*sizeof(float));
  mult_mat(f, e, z, 2, 2, 3);
  print_mat(z, 2, 2);
	return 1;
}
