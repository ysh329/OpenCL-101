#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define  SUM_VEC4(vec4)  vec4.s0+vec4.s1+vec4.s2+vec4.s3

/*********************** TRAN *********************/
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

// float 1024x1024 
/*
__kernel void mat_trans_vec4_v33(const int heightB,
                                const int widthB,
                                __global const CL_INPUT_TYPE *b,
                                __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0)*8;
    const int row = get_global_id(1);

    CL_ELEM_TYPE bb = vload8(0, b+row*widthB+col);
    bT[(col)  *heightB + row] = bb.s0;
    bT[(col+1)*heightB + row] = bb.s1;
    bT[(col+2)*heightB + row] = bb.s2;
    bT[(col+3)*heightB + row] = bb.s3;
}
*/

// float 1024x1024 0.020684 s 0.050696 GFLOPS 
__kernel void mat_trans_vec4_v3(const int heightB,
                                const int widthB,
                                __global const CL_INPUT_TYPE *b,
                                __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0)*4;
    const int row = get_global_id(1)*4;

    CL_ELEM_TYPE bb0 = vload4(0, b+row*widthB+col);
    bT[(col)  *heightB + row  ] = bb0.s0;
    bT[(col+1)*heightB + row  ] = bb0.s1;
    bT[(col+2)*heightB + row  ] = bb0.s2;
    bT[(col+3)*heightB + row  ] = bb0.s3;

    CL_ELEM_TYPE bb1 = vload4(0, b+(row+1)*widthB+col);
    bT[(col)  *heightB + row+1] = bb1.s0;
    bT[(col+1)*heightB + row+1] = bb1.s1;
    bT[(col+2)*heightB + row+1] = bb1.s2;
    bT[(col+3)*heightB + row+1] = bb1.s3;

    CL_ELEM_TYPE bb2 = vload4(0, b+(row+2)*widthB+col);
    bT[(col)  *heightB + row+2] = bb2.s0;
    bT[(col+1)*heightB + row+2] = bb2.s1;
    bT[(col+2)*heightB + row+2] = bb2.s2;
    bT[(col+3)*heightB + row+2] = bb2.s3;

    CL_ELEM_TYPE bb3 = vload4(0, b+(row+3)*widthB+col);
    bT[(col)  *heightB + row+3] = bb3.s0;
    bT[(col+1)*heightB + row+3] = bb3.s1;
    bT[(col+2)*heightB + row+3] = bb3.s2;
    bT[(col+3)*heightB + row+3] = bb3.s3;

}


// float 1024x1024 0.022585 s 0.046427 GFLOPS
__kernel void mat_trans_vec4_v2(const int heightB,
                                const int widthB,
                                __global const CL_INPUT_TYPE *b,
                                __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0)*8;
    const int row = get_global_id(1);

    CL_ELEM_TYPE bb0 = vload4(0, b+row*widthB+col);
    bT[(col)  *heightB + row] = bb0.s0;
    bT[(col+1)*heightB + row] = bb0.s1;
    bT[(col+2)*heightB + row] = bb0.s2;
    bT[(col+3)*heightB + row] = bb0.s3;

    CL_ELEM_TYPE bb1 = vload4(0, b+row*widthB+col+4);
    bT[(col+4)*heightB + row] = bb1.s0;
    bT[(col+5)*heightB + row] = bb1.s1;
    bT[(col+6)*heightB + row] = bb1.s2;
    bT[(col+7)*heightB + row] = bb1.s3;
}

// [128 64 1] float 1024x1024 bugbugbugbug -11
//*
__kernel void mat_trans_vec4_v22(const int heightB,
                                const int widthB,
                                __global const CL_INPUT_TYPE *b,
                                __global       CL_INPUT_TYPE *bT) {

    const int col = get_global_id(0)*8;
    const int row = get_global_id(1)*2;

    CL_ELEM_TYPE bb0 = vload4(0, b+row*widthB+col);
    bT[(col)  *heightB + row] = bb0.s0;
    bT[(col+1)*heightB + row] = bb0.s1;
    bT[(col+2)*heightB + row] = bb0.s2;
    bT[(col+3)*heightB + row] = bb0.s3;

    CL_ELEM_TYPE bb1 = vload4(0, b+row*widthB+col+4);
    bT[(col+4)*heightB + row] = bb1.s0;
    bT[(col+5)*heightB + row] = bb1.s1;
    bT[(col+6)*heightB + row] = bb1.s2;
    bT[(col+7)*heightB + row] = bb1.s3;

    CL_ELEM_TYPE bb2 = vload4(0, b+(row+1)*widthB+col);
    bT[(col)  *heightB + row+1] = bb2.s0;
    bT[(col+1)*heightB + row+1] = bb2.s1;
    bT[(col+2)*heightB + row+1] = bb2.s2;
    bT[(col+3)*heightB + row+1] = bb2.s3;

    CL_ELEM_TYPE bb3 = vload4(0, b+(row+1)*widthB+col+4);
    bT[(col+4)*heightB + row+1] = bb3.s0;
    bT[(col+5)*heightB + row+1] = bb3.s1;
    bT[(col+6)*heightB + row+1] = bb3.s2;
    bT[(col+7)*heightB + row+1] = bb3.s3;

}
//*/

/*********************** MULT *********************/
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


// float 1024x1024x1024 0.682137 s 3.148169 GFLOPS
__kernel void mat_mult_trans_vec4_v2(const int M,
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


// [256, 256 1] float 1024x1024x1024 0.502325 s 4.275089 GFLOPS
__kernel void mat_mult_trans_vec4x4(const int M,
                                    const int N,
                                    const int K,
                                    __global const CL_INPUT_TYPE *a,
                                    __global const CL_INPUT_TYPE *bT,
                                    __global       CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) * 4;
    const int row = get_global_id(1) * 4;
    CL_ELEM_TYPE res00 = 0, res01 = 0, res02 = 0, res03 = 0,
                 res10 = 0, res11 = 0, res12 = 0, res13 = 0,
                 res20 = 0, res21 = 0, res22 = 0, res23 = 0,
                 res30 = 0, res31 = 0, res32 = 0, res33 = 0;

    for (int p = 0; p < K; p+=4) 
    {
        CL_ELEM_TYPE aa0 = vload4(0, a + row     * K + p);
        CL_ELEM_TYPE aa1 = vload4(0, a + (row+1) * K + p);
        CL_ELEM_TYPE aa2 = vload4(0, a + (row+2) * K + p);
        CL_ELEM_TYPE aa3 = vload4(0, a + (row+3) * K + p);

        CL_ELEM_TYPE bb0 = vload4(0, bT + col     * K + p);
        CL_ELEM_TYPE bb1 = vload4(0, bT + (col+1) * K + p);
        CL_ELEM_TYPE bb2 = vload4(0, bT + (col+2) * K + p);
        CL_ELEM_TYPE bb3 = vload4(0, bT + (col+3) * K + p);

        res00 += aa0 * bb0; res01 += aa0 * bb1; res02 += aa0 * bb2; res03 += aa0 * bb3;
        res10 += aa1 * bb0; res11 += aa1 * bb1; res12 += aa1 * bb2; res13 += aa1 * bb3;
        res20 += aa2 * bb0; res21 += aa2 * bb1; res22 += aa2 * bb2; res23 += aa2 * bb3;
        res30 += aa3 * bb0; res31 += aa3 * bb1; res32 += aa3 * bb2; res33 += aa3 * bb3;
    }

    c[row * N + col    ] = SUM_VEC4(res00); 
    c[row * N + (col+1)] = SUM_VEC4(res01); 
    c[row * N + (col+2)] = SUM_VEC4(res02); 
    c[row * N + (col+3)] = SUM_VEC4(res03); 

    c[(row + 1) * N + col    ] = SUM_VEC4(res10); 
    c[(row + 1) * N + (col+1)] = SUM_VEC4(res11); 
    c[(row + 1) * N + (col+2)] = SUM_VEC4(res12); 
    c[(row + 1) * N + (col+3)] = SUM_VEC4(res13); 

    c[(row + 2) * N + col    ] = SUM_VEC4(res20); 
    c[(row + 2) * N + (col+1)] = SUM_VEC4(res21); 
    c[(row + 2) * N + (col+2)] = SUM_VEC4(res22); 
    c[(row + 2) * N + (col+3)] = SUM_VEC4(res23); 

    c[(row + 3) * N + col    ] = SUM_VEC4(res30); 
    c[(row + 3) * N + (col+1)] = SUM_VEC4(res31); 
    c[(row + 3) * N + (col+2)] = SUM_VEC4(res32); 
    c[(row + 3) * N + (col+3)] = SUM_VEC4(res33); 

}
//*/
