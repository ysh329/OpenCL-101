#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define  SUM_VEC4(vec4)  vec4.s0+vec4.s1+vec4.s2+vec4.s3


__kernel void mat_trans_vec4(const int k,
                             const int n,
                             __global const CL_INPUT_TYPE *b,
                             __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0) * VEC_LEN;
    const int row = get_global_id(1);

    CL_ELEM_TYPE bb = vload8(0, b+row*n+col);
    vstore8(bb, 0, bT+(col/VEC_LEN) * (VEC_LEN*k) + row*VEC_LEN);
}



__kernel void mat_interleave_vec4(const int m,
                                  const int k,
                                  __global const CL_INPUT_TYPE *a,
                                  __global       CL_INPUT_TYPE *aI) {
    const int col = get_global_id(0) * VEC_LEN;
    const int row = get_global_id(1) * VEC_LEN;

    CL_ELEM_TYPE aa0 = vload8(0, a + row     * k + col),
                 aa1 = vload8(0, a + (row+1) * k + col),
                 aa2 = vload8(0, a + (row+2) * k + col),
                 aa3 = vload8(0, a + (row+3) * k + col),
                 aa4 = vload8(0, a + (row+4) * k + col),
                 aa5 = vload8(0, a + (row+5) * k + col),
                 aa6 = vload8(0, a + (row+6) * k + col),
                 aa7 = vload8(0, a + (row+7) * k + col);

    CL_ELEM_TYPE 
    res = (CL_ELEM_TYPE)(aa0.s0, aa1.s0, aa2.s0, aa3.s0, aa4.s0, aa5.s0, aa6.s0, aa7.s0);
    vstore8(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN));

    res = (CL_ELEM_TYPE)(aa0.s1, aa1.s1, aa2.s1, aa3.s1, aa4.s1, aa5.s1, aa6.s1, aa7.s1);
    vstore8(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN+VEC_LEN));

    res = (CL_ELEM_TYPE)(aa0.s2, aa1.s2, aa2.s2, aa3.s2, aa4.s2, aa5.s2, aa6.s2, aa7.s2);
    vstore8(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN+VEC_LEN*2));

    res = (CL_ELEM_TYPE)(aa0.s3, aa1.s3, aa2.s3, aa3.s3, aa4.s3, aa5.s3, aa6.s3, aa7.s3);
    vstore8(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN+VEC_LEN*3));


    res = (CL_ELEM_TYPE)(aa0.s4, aa1.s4, aa2.s4, aa3.s4, aa4.s4, aa5.s4, aa6.s4, aa7.s4);
    vstore8(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN+VEC_LEN*4));

    res = (CL_ELEM_TYPE)(aa0.s5, aa1.s5, aa2.s5, aa3.s5, aa4.s5, aa5.s5, aa6.s5, aa7.s5);
    vstore8(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN+VEC_LEN*5));

    res = (CL_ELEM_TYPE)(aa0.s6, aa1.s6, aa2.s6, aa3.s6, aa4.s6, aa5.s6, aa6.s6, aa7.s6);
    vstore8(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN+VEC_LEN*6));

    res = (CL_ELEM_TYPE)(aa0.s7, aa1.s7, aa2.s7, aa3.s7, aa4.s7, aa5.s7, aa6.s7, aa7.s7);
    vstore8(res, 0, aI+(row/VEC_LEN)*(VEC_LEN*k) + (col*VEC_LEN+VEC_LEN*7));

}



// float8
// [128,128,1] [1,1,1] p+=8  100times 1024x1024x1024 0.932780 s 2.302241 GFLOPS
// [128,128,1] [8,8,1] p+=8  100times 1024x1024x1024 0.243791 s 8.808704 GFLOPS


// half8
// [128,128,1] [1,1,1] p+=8  100times 1024x1024x1024 0.218132 s 9.844870 GFLOPS 

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
    const int col = get_group_id(1) * VEC_LEN + get_local_id(1);
    const int row = get_group_id(0) * VEC_LEN + get_local_id(0);
#endif

    CL_ELEM_TYPE c00 = 0.0,
                 c10 = 0.0,
                 c20 = 0.0,
                 c30 = 0.0,
                 c40 = 0.0,
                 c50 = 0.0,
                 c60 = 0.0,
                 c70 = 0.0;

    for (int p = 0; p < aI_width; p += 24) {
        CL_ELEM_TYPE
        aa = vload8(0, aI + row * aI_width + p),
        bb = vload8(0, bT + col * aI_width + p);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;
        c40 += (CL_ELEM_TYPE)aa.s4 * bb;
        c50 += (CL_ELEM_TYPE)aa.s5 * bb; 
        c60 += (CL_ELEM_TYPE)aa.s6 * bb;
        c70 += (CL_ELEM_TYPE)aa.s7 * bb;
//*
        aa = vload8(0, aI + row * aI_width + p+VEC_LEN);
        bb = vload8(0, bT + col * aI_width + p+VEC_LEN);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;
        c40 += (CL_ELEM_TYPE)aa.s4 * bb;
        c50 += (CL_ELEM_TYPE)aa.s5 * bb; 
        c60 += (CL_ELEM_TYPE)aa.s6 * bb;
        c70 += (CL_ELEM_TYPE)aa.s7 * bb;
//*/
//*
        aa = vload8(0, aI + row * aI_width + p+VEC_LEN*2);
        bb = vload8(0, bT + col * aI_width + p+VEC_LEN*2);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;
//*/

    }

    vstore8(c00, 0, c+(row*VEC_LEN  ) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore8(c10, 0, c+(row*VEC_LEN+1) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore8(c20, 0, c+(row*VEC_LEN+2) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore8(c30, 0, c+(row*VEC_LEN+3) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore8(c40, 0, c+(row*VEC_LEN+4) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore8(c50, 0, c+(row*VEC_LEN+5) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore8(c60, 0, c+(row*VEC_LEN+6) * (bT_height*VEC_LEN) + (col*VEC_LEN));
    vstore8(c70, 0, c+(row*VEC_LEN+7) * (bT_height*VEC_LEN) + (col*VEC_LEN));

}
