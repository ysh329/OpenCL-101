// If use this .h file,
// YOU should define ELEM_TYPE
// MACRO in your CPP file.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

double timer(void)
{
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue, &dummy);
  double etime = (double)Tvalue.tv_sec + 1.0e-6*((double)Tvalue.tv_usec);

  return etime;
}

ELEM_TYPE min(ELEM_TYPE a, ELEM_TYPE b) {
    if (a > b) {
        return b;
    }
    else {
        return a;
    }
}

void init_mat(ELEM_TYPE *mat, int len, ELEM_TYPE setVal) {
    for (int idx = 0; idx < len; idx++)
        mat[idx] = (ELEM_TYPE)setVal;
}

void rand_mat(ELEM_TYPE *mat, int len, int range) {
    if (range < 1) {
        printf("range value can't be less than 1.\n");
        exit(-1);
    }
    srand( (unsigned) time(0) );
    for (int idx = 0; idx < len; idx++)
        mat[idx] = (ELEM_TYPE) (rand() % range);
}

// row-major
void print_mat(ELEM_TYPE *mat, int width, int height) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++)
                printf("%.2f ", (float)mat[r*width+c]);
        printf("\n");
    }
    printf("\n");
}

void print_vec(ELEM_TYPE *vec, int len) {
    for (int idx = 0; idx < len; idx++)
        printf("%.2f \n", (float)vec[idx]);
}

// row-major
void add_mat(ELEM_TYPE *a, ELEM_TYPE *b, ELEM_TYPE *res, int width, int height) {
    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++) 
            res[r*width + c] = a[r*width + c] + b[r*width + c];
}

ELEM_TYPE max(ELEM_TYPE a, ELEM_TYPE b) {
    if (a > b)
        return a;
    else
        return b;
}

void add_vec(ELEM_TYPE *a, ELEM_TYPE *b, ELEM_TYPE *res, int len) {
    for (int idx = 0; idx < len; idx++) 
        res[idx] = a[idx] + b[idx];
}

// row-major
// CPU4 FLOAT CPU-MATRIX-TRANS-FOR-B 2048x2048 0.344637 s
void transpose_mat_naive(ELEM_TYPE *a, int height, int width, ELEM_TYPE *res) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            res[col * height + row] = a[row * width + col];
        }
    }
}

// row-major
// wrong impl
void transpose_mat_inplace(ELEM_TYPE *a, int height, int width, ELEM_TYPE *res) {
    ELEM_TYPE tmp;
    int count = 0;
    printf("width: %d; height: %d\n", width, height);
    if (width > height) {
        printf("up\n");
        for (int row = 0; row < height; row++) {
		    for (int col = row; col < width; col++) {
                count += 1;
                tmp = a[row * width + col];
                res[row * width + col]  = a[col * height + row];
                res[col * height + row] = tmp;
                //printf("res[%d, %d]:= a[%d, %d] = %.2f \n", r, c, c, r, a[c*height+r]);
                //printf("res[%d] := a[%d] = %.2f \n\n", r*width + c, c*height + r, a[c*height+r]);
            }
        }
    }
    else {
        printf("down\n");
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < row; col++) {
                count += 1;
                tmp = a[row * width + col];
				res[row * width + col]  = a[col * height + row];
				res[col * height + row] = tmp;
            }
        }
    }
    printf("count:%d\n", count);
}

// row-major
int equal_mat(ELEM_TYPE *a, ELEM_TYPE *b, int width, int height) {
    int correct_num = 0;
    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++) 
            if (a[r*width + c] == b[r*width + c])
                correct_num += 1;

    float correct_rate = (float) correct_num / ( width * height );
    printf(">>> [TEST] correct rate: %.4f\n", correct_rate);
    if (1.0 - correct_rate < 10e-6)
        printf(">>> [TEST] ~ Bingo ~ matrix a == matrix b\n\n");
    else
        printf(">>> [TEST] matrix a is equal to matrix b\n\n");
    return 1;
}

int equal_vec(ELEM_TYPE *a, ELEM_TYPE *b, int len) {
    int correct_num = 0;
    for (int idx = 0; idx < len; idx++) 
        if (a[idx] - b[idx] < (ELEM_TYPE)1.0)
            correct_num += 1;

    float correct_rate = (float) correct_num / (float) len;
    printf(">>> [TEST] correct rate: %.4f\n", correct_rate);
    if (1.0 - correct_rate < 10e-6)
        printf(">>> [TEST] ~ Bingo ~ matrix a == matrix b\n\n");
    else
        printf(">>> [TEST] matrix a is NOT equal to matrix b\n\n");
    return 1;
}

void dotprod_mat(ELEM_TYPE *a, ELEM_TYPE *b, ELEM_TYPE *res, int len) {
    for (int idx = 0; idx < len; idx++)
        res[idx] = a[idx] * b[idx];
}

void dotprod_mat_alpha(ELEM_TYPE *a, ELEM_TYPE *res, int len, ELEM_TYPE alpha) {
    for (int idx = 0; idx < len; idx++)
        res[idx] = a[idx] * alpha;
}

// row-major
void mult_mat_alpha(ELEM_TYPE *a, ELEM_TYPE *b, ELEM_TYPE *res, int M, int N, int K, ELEM_TYPE alpha) {
    int i, j, p;
    init_mat(res, N*M, 0); 

    for (i = 0; i < M; i++) {
        for (p = 0; p < K; p++) {
            for (j = 0; j < N; j++) {
                res[i * N + j] += a[i * K + p] * b[p * N + j] * alpha;
                // check
                //printf("res[%d * %d + %d] %.2f += a[%d, %d] %.2f * b[%d, %d] %.2f \n", i, N, j, res[i*N+j], i, p, a[i*K+p], p, j, b[p*N+j]);
            }
        }
    }
}

void mult_mat(ELEM_TYPE *a, ELEM_TYPE *b, ELEM_TYPE *res, int M, int N, int K) {
    int i, j, p;
    init_mat(res, N*M, 0); 

    for (i = 0; i < M; i++) {
        for (p = 0; p < K; p++) {
            for (j = 0; j < N; j++) {
                res[i * N + j] += a[i * K + p] * b[p * N + j];
                // check
                //printf("res[%d * %d + %d] %.2f += a[%d, %d] %.2f * b[%d, %d] %.2f \n", i, N, j, res[i*N+j], i, p, a[i*K+p], p, j, b[p*N+j]);
            }
        }
    }
}


void copy_mat(ELEM_TYPE *a, ELEM_TYPE *b, int len) {
    for (int i = 0; i < len; i++) {
        b[i] = a[i];
    }
}
