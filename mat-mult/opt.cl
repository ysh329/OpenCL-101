#pragma OPENCL EXTENSION cl_khr_fp16 : enable


__kernel void mat_mult_naive(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    CL_ELEM_TYPE res = 0;

    for (int p = 0; p < K; p++) {
        res += a[row * M + p] * b[p * N + col];
    }
    c[row * N + col] = res;
}

#define A(i,j) a[ (j) * lda + (i) ]
#define B(i,j) b[ (j) * ldb + (i) ]
#define C(i,j) c[ (j) * ldc + (i) ]


__kernel void mat_mult_2(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1);

    AddDot(K, &a[0 * M + row], M, &b[(col+0) * N + 0], &c[(col+0)*N + row]);
    AddDot(K, &a[0 * M + row], M, &b[(col+1) * N + 0], &c[(col+1)*N + row]);
    AddDot(K, &a[0 * M + row], M, &b[(col+2) * N + 0], &c[(col+2)*N + row]);
    AddDot(K, &a[0 * M + row], M, &b[(col+3) * N + 0], &c[(col+3)*N + row]);

}

__kernel void AddDot(const int K, __global const CL_INPUT_TYPE *x, const int incx, __global const CL_INPUT_TYPE *y, __global CL_INPUT_TYPE *gamma) {
   
    for (int p = 0; p < K; p++) {
        *gamma += x[ p * incx ] * y[ p ];
    }
}
