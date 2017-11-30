#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void mat_trans_naive(const int heightB,
                              const int widthB,
                              __global const CL_INPUT_TYPE *b,
                              __global CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    bT[col*heightB + row] = b[row*widthB + col];
}

__kernel void mat_mult_naive_trans(const int M,
                                   const int N,
                                   const int K,
                                   __global const CL_INPUT_TYPE *a,
                                   __global const CL_INPUT_TYPE *bT,
                                   __global const CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    CL_ELEM_TYPE res = 0;

    for (int p = 0; p < K; p++) {
        res += a[row * K + p] * bT[row * K + p];
    }
    c[row * N + col] = res;
}
