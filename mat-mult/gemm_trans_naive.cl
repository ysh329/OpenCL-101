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

/**************************** col only **************************/

// [1 1024 1] float 1024x1024 0.007381 s 0.142072 GFLOPS
__kernel void mat_trans_too_naive_c0(const int heightB,
                                     const int widthB,
                                     __global const CL_INPUT_TYPE *b,
                                     __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0);
    for (int row = 0; row < heightB; row++) {
        bT[col * heightB + row] = b[row * widthB + col];
    }
}

// [1024 1 1] float 1024x1024 0.006511 s 0.161051 GFLOPS 
__kernel void mat_trans_too_naive_c1(const int heightB,
                                     const int widthB,
                                     __global const CL_INPUT_TYPE *b,
                                     __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(1);
    for (int row = 0; row < heightB; row++) {
        bT[col * heightB + row] = b[row * widthB + col];
    }
}

// [1 1 1024] float 1024x1024 0.006749 s 0.155369 GFLOPS
__kernel void mat_trans_too_naive_c2(const int heightB,
                                     const int widthB,
                                     __global const CL_INPUT_TYPE *b,
                                     __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(2);
    for (int row = 0; row < heightB; row++) {
        bT[col * heightB + row] = b[row * widthB + col];
    }
}

/**************************** row only **************************/

// [1 1024 1] float 1024x1024 0.005090 s 0.206026 GFLOPS
__kernel void mat_trans_too_naive_r0(const int heightB,
                                     const int widthB,
                                     __global const CL_INPUT_TYPE *b,
                                     __global       CL_INPUT_TYPE *bT) {
    const int row = get_global_id(0);
    for (int col = 0; col < widthB; col++) {
        bT[col * heightB + row] = b[row * widthB + col];
    }
}

// [1024 1 1] float 1024x1024 0.004797 s 0.218569 GFLOPS
__kernel void mat_trans_too_naive_r1(const int heightB,
                                     const int widthB,
                                     __global const CL_INPUT_TYPE *b,
                                     __global       CL_INPUT_TYPE *bT) {
    const int row = get_global_id(1);
    for (int col = 0; col < widthB; col++) {
        bT[col * heightB + row] = b[row * widthB + col];
    }
}


// [1 1 1024] float 1024x1024 0.004858 s 0.215864 GFLOPS
__kernel void mat_trans_too_naive_r2(const int heightB,
                                     const int widthB,
                                     __global const CL_INPUT_TYPE *b,
                                     __global       CL_INPUT_TYPE *bT) {
    const int row = get_global_id(2);
    for (int col = 0; col < widthB; col++) {
        bT[col * heightB + row] = b[row * widthB + col];
    }
}

/**************************** mat mult **************************/

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
