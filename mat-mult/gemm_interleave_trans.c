#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define  SUM_VEC4(vec4)  vec4.s0+vec4.s1+vec4.s2+vec4.s3


__kernel void mat_trans_vec4(const int k,
                             const int n,
                             __global const CL_INPUT_TYPE *b,
                             __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0) * VEC_LEN;
    const int row = get_global_id(1);

    CL_ELEM_TYPE bb = vload4(0, b+row*n+col);
    vstore4(bb, 0, bT+(col/VEC_LEN) * (VEC_LEN*k) + row*VEC_LEN);
}



__kernel void mat_interleave_vec4(const int m,
                                  const int k,
                                  __global const CL_INPUT_TYPE *a,
                                  __global       CL_INPUT_TYPE *aI) {
    const int col = get_global_id(0) * VEC_LEN;
    const int row = get_global_id(1) * VEC_LEN;

    CL_ELEM_TYPE aa0 = vload4(0, a + row     * k + col),
                 aa1 = vload4(0, a + (row+1) * k + col),
                 aa2 = vload4(0, a + (row+2) * k + col),
                 aa3 = vload4(0, a + (row+3) * k + col);

    CL_ELEM_TYPE 
    res = (CL_ELEM_TYPE)(aa0.s0, aa1.s0, aa2.s0, aa3.s0);
    vstore4(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN));

    res = (CL_ELEM_TYPE)(aa0.s1, aa1.s1, aa2.s1, aa3.s1);
    vstore4(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN+VEC_LEN));

    res = (CL_ELEM_TYPE)(aa0.s2, aa1.s2, aa2.s2, aa3.s2);
    vstore4(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN+VEC_LEN*2));

    res = (CL_ELEM_TYPE)(aa0.s3, aa1.s3, aa2.s3, aa3.s3);
    vstore4(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN+VEC_LEN*3));

}



// float4
// [256,256,1] [1,1,1] p+=8  100times 1024x1024x1024 0.220724 s  9.729284 GFLOPS
// [256,256,1] [4,4,1] p+=8  100times 1024x1024x1024 0.081261 s 26.427127 GFLOPS
// [256,256,1] [4,4,1] p+=12 100times 1024x1024x1020 0.075207 s 28.442905 GFLOPS

// half4
// [256,256,1] [1,1,1] p+=8  100times 1024x1024x1024 0.103472 s 20.754221 GFLOPS 
// [256,256,1] [4,4,1] p+=8  100times 1024x1024x1024 0.061210 s 35.084041 GFLOPS
// [256,256,1] [4,4,1] p+=12 100times 1024x1024x1024 0.058183 s 36.765208 GFLOPS

__kernel void gemm_interleaved_transposed_vec4(const int aI_height, // height of aI
                                               const int bT_height, // height of bT
                                               const int aI_width,  // width of aI or bT
                                               __global const CL_INPUT_TYPE *aI,
                                               __global const CL_INPUT_TYPE *bT,
                                               __global       CL_INPUT_TYPE *c) {
#ifndef USE_LOCAL_WOKR_SIZE
    const int col = get_global_id(0); // col of bT: [0, n/4) <==> [0, bT_height)
    const int row = get_global_id(1); // row of aI: [0, m/4) <==> [0, aI_height)
#else
    const int row = get_group_id(1) * VEC_LEN + get_local_id(1);
    const int col = get_group_id(0) * VEC_LEN + get_local_id(0);
#endif

    CL_ELEM_TYPE c00 = 0.0,
                 c10 = 0.0,
                 c20 = 0.0,
                 c30 = 0.0;

    for (int p = 0; p < aI_width; p += 12) {
        CL_ELEM_TYPE
        aa = vload4(0, aI + row * aI_width + p),
        bb = vload4(0, bT + col * aI_width + p);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;

        aa = vload4(0, aI + row * aI_width + p+VEC_LEN);
        bb = vload4(0, bT + col * aI_width + p+VEC_LEN);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;
//*
        aa = vload4(0, aI + row * aI_width + p+VEC_LEN*2);
        bb = vload4(0, bT + col * aI_width + p+VEC_LEN*2);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;
//*/
    }

    vstore4(c00, 0, c+(row*VEC_LEN  ) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore4(c10, 0, c+(row*VEC_LEN+1) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore4(c20, 0, c+(row*VEC_LEN+2) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore4(c30, 0, c+(row*VEC_LEN+3) * (bT_height*VEC_LEN) + (col*VEC_LEN));
}


/*
__kernel void gemm_interleaved_transposed_vec8(const int aI_height, // height of aI
                                               const int bT_height, // height of bT
                                               const int aI_width,  // width of aI or bT
                                               __global const CL_INPUT_TYPE *aI,
                                               __global const CL_INPUT_TYPE *bT,
                                               __global       CL_INPUT_TYPE *c) {
#ifndef USE_LOCAL_WOKR_SIZE
    const int col = get_global_id(0); // col of bT: [0, n/4) <==> [0, bT_height)
    const int row = get_global_id(1); // row of aI: [0, m/4) <==> [0, aI_height)
#else
    const int col = get_group_id(1) * VEC_LEN + get_local_id(1);
    const int row = get_group_id(0) * VEC_LEN + get_local_id(0);
#endif

    CL_ELEM_TYPE c00 = 0.0,
                 c10 = 0.0,
                 c20 = 0.0,
                 c30 = 0.0;

    for (int p = 0; p < aI_width; p += 8) {
        CL_ELEM_TYPE
        aa = vload8(0, aI + row * aI_width + p),
        bb = vload8(0, bT + col * aI_width + p);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;

        aa = vload8(0, aI + row * aI_width + p+VEC_LEN);
        bb = vload8(0, bT + col * aI_width + p+VEC_LEN);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;

        aa = vload8(0, aI + row * aI_width + p+VEC_LEN*2);
        bb = vload8(0, bT + col * aI_width + p+VEC_LEN*2);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;
    }

    vstore8(c00, 0, c+(row*VEC_LEN  ) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore8(c10, 0, c+(row*VEC_LEN+1) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore8(c20, 0, c+(row*VEC_LEN+2) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore8(c30, 0, c+(row*VEC_LEN+3) * (bT_height*VEC_LEN) + (col*VEC_LEN));
}
*/
