#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// float 1024x1024 0.017492 s 0.059945 GFLOPS
__kernel void mat_trans_naive(const int heightB,
                              const int widthB,
                              __global const CL_INPUT_TYPE *b,
                              __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    bT[col*heightB + row] = b[row*widthB + col];
}

// float 1024x1024x1024 2.950411 s 0.727859 GFLOPS
__kernel void mat_mult_trans_naive(const int M,
                                   const int N,
                                   const int K,
                                   __global const CL_INPUT_TYPE *a,
                                   __global const CL_INPUT_TYPE *bT,
                                   __global       CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    CL_ELEM_TYPE res = 0;

    for (int p = 0; p < K; p++) {
        res += a[row * K + p] * bT[col * K + p];
    }
    c[row * N + col] = res;
}
