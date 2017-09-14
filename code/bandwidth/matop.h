#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init_mat(float *mat, int len, float setVal) {
    for (int idx = 0; idx < len; idx++)
        mat[idx] = (ELEM_TYPE)setVal;
}

void rand_mat(float *mat, int len, int range) {
    if (range < 1) {
        printf("range value can't be less than 1.\n");
        exit(-1);
    }
    srand( (unsigned) time(0) );
    for (int idx = 0; idx < len; idx++)
        mat[idx] = (ELEM_TYPE) (rand() % range);
}

// row-major
void print_mat(float *mat, int width, int height) {
#ifdef NOT_PRINT_FLAG
    return;
#endif
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++)
            printf("%.2f ", (ELEM_TYPE)mat[r*width+c]);
        printf("\n");
    }
    printf("\n");
}

void print_vec(float *vec, int len) {
    for (int idx = 0; idx < len; idx++)
        printf("%.2f \n", (ELEM_TYPE)vec[idx]);
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
void transpose_mat(ELEM_TYPE *a, int width, int height, ELEM_TYPE *res) {
    for (int r = 0; r < height; r++)
		    for (int c = 0; c < width; c++) {
            res[c*height + r] = a[r*width + c]; 
            //printf("res[%d, %d]:= a[%d, %d] = %.2f \n", r, c, c, r, a[c*height+r]);
            //printf("res[%d] := a[%d] = %.2f \n\n", r*width + c, c*height + r, a[c*height+r]);
        }
}

// row-major
int equal_mat(ELEM_TYPE *a, ELEM_TYPE *b, int width, int height) {
    int correct_num = 0;
    for (int r = 0; r < height; r++)
        for (int c = 0; c < width; c++) 
            if (a[r*width + c] == b[r*width + c])
                correct_num += 1;

    float correct_rate = (float) correct_num / ( width * height );
    printf(">>> correct rate: %.4f\n", correct_rate);
    if (1.0 - correct_rate < 10e-6)
        printf(">>> ~ Bingo ~ matrix a == matrix b\n\n");
    else
        printf(">>> matrix a is equal to matrix b\n\n");
    return 1;
}

int equal_vec(ELEM_TYPE *a, ELEM_TYPE *b, int len) {
    int correct_num = 0;
    for (int idx = 0; idx < len; idx++) 
        if (a[idx] == b[idx])
            correct_num += 1;

    float correct_rate = (float) correct_num / (float) len;
    printf(">>> correct rate: %.4f\n", correct_rate);
    if (1.0 - correct_rate < 10e-8)
        printf(">>> ~ Bingo ~ matrix a == matrix b\n\n");
    else
        printf(">>> matrix a is NOT equal to matrix b\n\n");
    return 1;
}

void dotprod_mat(ELEM_TYPE *a, ELEM_TYPE *b, ELEM_TYPE *res, int len) {
    for (int idx; idx < len; idx++)
        res[idx] = a[idx] * b[idx];
}

// row-major
void mult_mat(ELEM_TYPE *a, ELEM_TYPE *b, ELEM_TYPE *res, int M, int N, int K) {
    int i, j, p;
    init_mat(res, M*K, 0); 
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) 
            for (p = 0; p < K; p++) {
                res[i * M + j] += a[i * K + p] * b[p * N + j]; 
                //printf("res[%d, %d] %.2f += a[%d, %d] %.2f * b[%d, %d] %.2f \n", i, j, res[i*M+j], i, p, a[i*K+p], p, j, b[p*N+j]);
            }
}

void copy_mat(ELEM_TYPE *a, ELEM_TYPE *b, int len) {
    for (int i = 0; i < len; i++) {
        b[i] = a[i];
    }
}
