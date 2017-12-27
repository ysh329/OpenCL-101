#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define  SUM_VEC8(vec8)  vec8.s0+vec8.s1+vec8.s2+vec8.s3+vec8.s4+vec8.s5+vec8.s6+vec8.s7


// float 1024x1024 0.021925 s 0.047825 GFLOPS  
/*
__kernel void mat_trans_vec8_v1(const int heightB,
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
    bT[(col+4)*heightB + row] = bb.s4;
    bT[(col+5)*heightB + row] = bb.s5;
    bT[(col+6)*heightB + row] = bb.s6;
    bT[(col+7)*heightB + row] = bb.s7;
}
*/

// [128 1024 1] float8 1024x1024 0.003156 s 0.332235 GFLOPS
// [256 2048 1] float8 2048x2048 0.012941 s 0.324115 GFLOPS
__kernel void mat_trans_vec8_v3(const int heightB,
                                const int widthB,
                                __global const CL_INPUT_TYPE *b,
                                __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0);
    const int row = get_global_id(1)*8;

    CL_ELEM_TYPE bb = (CL_ELEM_TYPE)(b[row*widthB+col],
                                     b[(row+1)*widthB+col],
                                     b[(row+2)*widthB+col],
                                     b[(row+3)*widthB+col],
                                     b[(row+4)*widthB+col],
                                     b[(row+5)*widthB+col],
                                     b[(row+6)*widthB+col],
                                     b[(row+7)*widthB+col]);

    vstore8(bb, 0, bT+col*heightB+row);
    //vstore_half8(bb, 0, bT+col*heightB+row); half8 with bug -11
}

// float 1024x1024 0.019560 s 0.053608 GFLOPS
/* 
__kernel void mat_trans_vec8_v2(const int heightB,
                                const int widthB,
                                __global const CL_INPUT_TYPE *b,
                                __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0)*8;
    const int row = get_global_id(1)*8;

    CL_ELEM_TYPE bb0 = vload8(0, b+row*widthB+col);
    bT[(col)  *heightB + row  ] = bb0.s0;
    bT[(col+1)*heightB + row  ] = bb0.s1;
    bT[(col+2)*heightB + row  ] = bb0.s2;
    bT[(col+3)*heightB + row  ] = bb0.s3;
    bT[(col+4)*heightB + row  ] = bb0.s4;
    bT[(col+5)*heightB + row  ] = bb0.s5;
    bT[(col+6)*heightB + row  ] = bb0.s6;
    bT[(col+7)*heightB + row  ] = bb0.s7;

    CL_ELEM_TYPE bb1 = vload8(0, b+(row+1)*widthB+col);
    bT[(col)  *heightB + row+1] = bb1.s0;
    bT[(col+1)*heightB + row+1] = bb1.s1;
    bT[(col+2)*heightB + row+1] = bb1.s2;
    bT[(col+3)*heightB + row+1] = bb1.s3;
    bT[(col+4)*heightB + row+1] = bb1.s4;
    bT[(col+5)*heightB + row+1] = bb1.s5;

    bT[(col+7)*heightB + row+1] = bb1.s7;

    CL_ELEM_TYPE bb2 = vload8(0, b+(row+2)*widthB+col);
    bT[(col)  *heightB + row+2] = bb2.s0;
    bT[(col+1)*heightB + row+2] = bb2.s1;
    bT[(col+2)*heightB + row+2] = bb2.s2;
    bT[(col+3)*heightB + row+2] = bb2.s3;
    bT[(col+4)*heightB + row+2] = bb2.s4;
    bT[(col+5)*heightB + row+2] = bb2.s5;
    bT[(col+6)*heightB + row+2] = bb2.s6;
    bT[(col+7)*heightB + row+2] = bb2.s7;

    CL_ELEM_TYPE bb3 = vload8(0, b+(row+3)*widthB+col);
    bT[(col)  *heightB + row+3] = bb3.s0;
    bT[(col+1)*heightB + row+3] = bb3.s1;
    bT[(col+2)*heightB + row+3] = bb3.s2;
    bT[(col+3)*heightB + row+3] = bb3.s3;
    bT[(col+4)*heightB + row+3] = bb3.s4;
    bT[(col+5)*heightB + row+3] = bb3.s5;
    bT[(col+6)*heightB + row+3] = bb3.s6;
    bT[(col+7)*heightB + row+3] = bb3.s7;

    CL_ELEM_TYPE bb4 = vload8(0, b+(row+4)*widthB+col);
    bT[(col)  *heightB + row+4] = bb4.s0;
    bT[(col+1)*heightB + row+4] = bb4.s1;
    bT[(col+2)*heightB + row+4] = bb4.s2;
    bT[(col+3)*heightB + row+4] = bb4.s3;
    bT[(col+4)*heightB + row+4] = bb4.s4;
    bT[(col+5)*heightB + row+4] = bb4.s5;
    bT[(col+6)*heightB + row+4] = bb4.s6;
    bT[(col+7)*heightB + row+4] = bb4.s7;

    CL_ELEM_TYPE bb5 = vload8(0, b+(row+5)*widthB+col);
    bT[(col)  *heightB + row+5] = bb5.s0;
    bT[(col+1)*heightB + row+5] = bb5.s1;
    bT[(col+2)*heightB + row+5] = bb5.s2;
    bT[(col+3)*heightB + row+5] = bb5.s3;
    bT[(col+4)*heightB + row+5] = bb5.s4;
    bT[(col+5)*heightB + row+5] = bb5.s5;
    bT[(col+6)*heightB + row+5] = bb5.s6;
    bT[(col+7)*heightB + row+5] = bb5.s7;

    CL_ELEM_TYPE bb6 = vload8(0, b+(row+6)*widthB+col);
    bT[(col)  *heightB + row+6] = bb6.s0;
    bT[(col+1)*heightB + row+6] = bb6.s1;
    bT[(col+2)*heightB + row+6] = bb6.s2;
    bT[(col+3)*heightB + row+6] = bb6.s3;
    bT[(col+4)*heightB + row+6] = bb6.s4;
    bT[(col+5)*heightB + row+6] = bb6.s5;
    bT[(col+6)*heightB + row+6] = bb6.s6;
    bT[(col+7)*heightB + row+6] = bb6.s7;

    CL_ELEM_TYPE bb7 = vload8(0, b+(row+7)*widthB+col);
    bT[(col)  *heightB + row+7] = bb7.s0;
    bT[(col+1)*heightB + row+7] = bb7.s1;
    bT[(col+2)*heightB + row+7] = bb7.s2;
    bT[(col+3)*heightB + row+7] = bb7.s3;
    bT[(col+4)*heightB + row+7] = bb7.s4;
    bT[(col+5)*heightB + row+7] = bb7.s5;
    bT[(col+6)*heightB + row+7] = bb7.s6;
    bT[(col+7)*heightB + row+7] = bb7.s7;

}
*/
