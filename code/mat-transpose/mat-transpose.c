#include <stdio.h>

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

float* transpose_mat(float *a, int width, int height) {
    float *res;
    res = (float *) malloc (width * height * sizeof(float));
    init_mat(res, width * height, 0);
    for (int c = 0; c < width; c++)
        for (int r = 0; r < height; r++)
            res[r*width + c] = a[c*height + r];
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
    c_t = transpose_mat(c, 2, 3);
    print_mat(c_t, 3, 2);

    

    return 1;
}

