#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <time.h>
#include <sys/time.h>

/* opencl */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define     MEM_SIZE            (128)
#define     MAX_SOURCE_SIZE     (0x1000000)

#define     MATRIX_WIDTH        (2)
#define     MATRIX_HEIGHT       (3)
#define     VECTOR_LEN          (3)


void init_mat(float *mat, int len, int setElemOne) {
    for (int idx = 0; idx < len; idx++)
        if (setElemOne)
            mat[idx] = 1;
        else
            mat[idx] = 0;
}

void rand_mat(float *mat, int len, int range) {
    if (range < 1) {
        printf("range value can't be less than 1.\n");
        exit(-1);
    }
    srand( (unsigned) time(0) );
    for (int idx = 0; idx < len; idx++)
        mat[idx] = rand() % range;
}

void print_mat(float *mat, int width, int height) {
    for (int c = 0; c < width; c++) {
        for (int r = 0; r < height; r++)
            printf("%.2f ", mat[c*height+r]);
        printf("\n");
    }
    printf("\n");
}

void add_mat(float *a, float *b, float *res, int width, int height) {
    for (int c = 0; c < width; c++)
        for (int r = 0; r < height; r++)
            res[c*height + r] = a[c*height + r] + b[c*height + r];
}

void add_vec(float *a, float *b, float *res, int len) {
    for (int idx = 0; idx < len; idx++) 
        res[idx] = a[idx] + b[idx];
}

void transpose_mat(float *a, int width, int height, float *res) {
    for (int c = 0; c < width; c++)
        for (int r = 0; r < height; r++) {
            res[r*width + c] = a[c*height + r];
			//printf("res[%d, %d]:= a[%d, %d] = %.2f \n", r, c, c, r, a[c*height+r]);
			//printf("res[%d] := a[%d] = %.2f \n\n", r*width + c, c*height + r, a[c*height+r]);
		}
}

int equal_mat(float *a, float *b, int width, int height) {
	for (int c = 0; c < width; c++)
		for (int r = 0; r < height; r++) {
			if (a[c*height+r] - b[c*height+r] > 10e-8) {
				//printf("matrix a is NOT equal to matrix b\n");
				return 0;
			}
		}
	//printf("matrix a is equal to matrix b\n");
	return 1;
}

int equal_vec(float *a, float *b, int len) {
	for (int idx = 0; idx < len; idx++)
		if (a[idx] - b[idx] > 10e-8) {
			//printf("matrix a is NOT equal to matrix b\n");
			return 0;
		}
	//printf("matrix a is equal to matrix b\n");
	return 1;
}


int main(void) {

    float *a, *b, *c;
    a = (float *) malloc (2 * 3 * sizeof(float));
    b = (float *) malloc (2 * 3 * sizeof(float));
    c = (float *) malloc (2 * 3 * sizeof(float));

    printf("a:\n");
    rand_mat(a, 2, 2);
    print_mat(a, 2, 3);

    printf("b:\n");
    rand_mat(b, 2, 3);
    print_mat(b, 2, 3);

    printf("c:\n");
    init_mat(c, 2*3, 0);
    print_mat(c, 2, 3);

    printf("c := a + b, using add_mat \n");
    add_mat(a, b, c, 2, 3);
    print_mat(c, 2, 3);

    printf("c := a + b, using add_vec\n");
    init_mat(c, 2*3, 0);
    add_vec(a, b, c, 2*3);
    print_mat(c, 2, 3);

    printf("c^T:\n");
    float *c_t;
	c_t = (float *) malloc (2 * 3 * sizeof(float));
    transpose_mat(c, 2, 3, c_t);
	printf("finished transpose\n");
    print_mat(c_t, 3, 2);

	equal_mat(c, c, 2, 3);
	equal_mat(a, b, 2, 3);

	equal_vec(c, c, 2*3);
	equal_vec(a, b, 2*3);

    

    return 1;
}

