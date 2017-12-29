#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define  SUM_VEC4(vec4)  vec4.s0+vec4.s1+vec4.s2+vec4.s3


__kernel void mat_trans_vec4(const int k,
                             const int n,
                             __global const CL_INPUT_TYPE *b,
                             __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0) * 4;
    const int row = get_global_id(1);

    CL_ELEM_TYPE bb = vload4(0, b+row*n+col);
    vstore4(bb, 0, bT+(col/4) * (4*k) + row*4);
}



__kernel void mat_interleave_vec4(const int m,
                                  const int k,
                                  __global const CL_INPUT_TYPE *a,
                                  __global       CL_INPUT_TYPE *aI) {
    const int col = get_global_id(0) * 4;
    const int row = get_global_id(1) * 4;

    CL_ELEM_TYPE aa0 = vload4(0, a + row     * k + col),
                 aa1 = vload4(0, a + (row+1) * k + col),
                 aa2 = vload4(0, a + (row+2) * k + col),
                 aa3 = vload4(0, a + (row+3) * k + col);

    CL_ELEM_TYPE 
    res = (CL_ELEM_TYPE)(aa0.s0, aa1.s0, aa2.s0, aa3.s0);
    vstore4(res, 0, aI+(row/4)*(4*k) + (col*4));

    res = (CL_ELEM_TYPE)(aa0.s1, aa1.s1, aa2.s1, aa3.s1);
    vstore4(res, 0, aI+(row/4)*(4*k) + (col*4+4));

    res = (CL_ELEM_TYPE)(aa0.s2, aa1.s2, aa2.s2, aa3.s2);
    vstore4(res, 0, aI+(row/4)*(4*k) + (col*4+8));

    res = (CL_ELEM_TYPE)(aa0.s3, aa1.s3, aa2.s3, aa3.s3);
    vstore4(res, 0, aI+(row/4)*(4*k) + (col*4+12));

}

// float4
// [256,256,1] [1,1,1] 10times  1024x1024x1024 0.220724 s  9.729284 GFLOPS
// [256,256,1] [4,4,1] 100times 1024x1024x1024 0.085041 s 25.252324 GFLOPS
// half4
// [256,256,1] [1,1,1] 1024x1024x1024 0.107563 s 19.964855 GFLOPS

__kernel void gemm_interleaved_transposed_vec4(const int aI_height, // height of aI
                                               const int bT_height, // height of bT
                                               const int aI_width,  // width of aI or bT
                                               __global const CL_INPUT_TYPE *aI,
                                               __global const CL_INPUT_TYPE *bT,
                                               __global       CL_INPUT_TYPE *c) {
   // const int col = get_global_id(0); // col of bT: [0, n/4) <==> [0, bT_height)
   // const int row = get_global_id(1); // row of aI: [0, m/4) <==> [0, aI_height)
    const int col = get_group_id(1) * 4 + get_local_id(1);
    const int row = get_group_id(0) * 4 + get_local_id(0);

    CL_ELEM_TYPE c00 = 0.0f,
                 c10 = 0.0f,
                 c20 = 0.0f,
                 c30 = 0.0f;

    for (int p = 0; p < aI_width; p += 12) {
        CL_ELEM_TYPE
        aa = vload4(0, aI + row * aI_width + p),
        bb = vload4(0, bT + col * aI_width + p);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;

        
        aa = vload4(0, aI + row * aI_width + p+4);
        bb = vload4(0, bT + col * aI_width + p+4);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;


        aa = vload4(0, aI + row * aI_width + p+8);
        bb = vload4(0, bT + col * aI_width + p+8);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb; 
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;       
    }

    vstore4(c00, 0, c+(row*4  ) * (bT_height*4) + (col*4));
    vstore4(c10, 0, c+(row*4+1) * (bT_height*4) + (col*4));
    vstore4(c20, 0, c+(row*4+2) * (bT_height*4) + (col*4));
    vstore4(c30, 0, c+(row*4+3) * (bT_height*4) + (col*4));

}
