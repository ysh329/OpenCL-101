#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define  SUM_VEC4(vec4)  vec4.s0+vec4.s1+vec4.s2+vec4.s3

// float 1024x1024 0.018215 s 0.057568 GFLOPS
__kernel void mat_trans_vec4(const int heightB,
                             const int widthB,
                             __global const CL_INPUT_TYPE *b,
                             __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0)*4;
    const int row = get_global_id(1);

    CL_ELEM_TYPE bb = vload4(0, b+row*widthB+col);
    bT[(col)  *heightB + row] = bb.s0;
    bT[(col+1)*heightB + row] = bb.s1;
    bT[(col+2)*heightB + row] = bb.s2;
    bT[(col+3)*heightB + row] = bb.s3;

}

// float 1024x1024x1024 0.741765 s 2.895098 GFLOPS 
__kernel void mat_mult_trans_vec4(const int M,
                                  const int N,
                                  const int K,
                                  __global const CL_INPUT_TYPE *a,
                                  __global const CL_INPUT_TYPE *bT,
                                  __global       CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    CL_ELEM_TYPE res = 0;

    for (int p = 0; p < K; p+=4) {
        CL_ELEM_TYPE aa = vload4(0, a+row*K+p);
        CL_ELEM_TYPE bb = vload4(0, bT+col*K+p);
        res += aa * bb;
    }
    //vstore(res, 0, (CL_ELEM_TYPE *)(c+row*N+col));
    c[row * N + col] = SUM_VEC4(res);
}
