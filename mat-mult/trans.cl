#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define  SUM_VEC4(vec4)  vec4.s0+vec4.s1+vec4.s2+vec4.s3

// float 1024x1024 0.018215 s 0.057568 GFLOPS
/*
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
*/

__kernel void mat_trans_vec16(const int heightB,
                             const int widthB,
                             __global const CL_INPUT_TYPE *b,
                             __global       CL_INPUT_TYPE *bT) {
    const int col = get_global_id(0)*16;
    const int row = get_global_id(1);

    CL_ELEM_TYPE bb = vload16(0, b+row*widthB+col);
    bT[(col   )*heightB + row] = bb.s0;
    bT[(col+ 1)*heightB + row] = bb.s1;
    bT[(col+ 2)*heightB + row] = bb.s2;
    bT[(col+ 3)*heightB + row] = bb.s3;
    bT[(col+ 4)*heightB + row] = bb.s4;
    bT[(col+ 5)*heightB + row] = bb.s5;
    bT[(col+ 6)*heightB + row] = bb.s6;
    bT[(col+ 7)*heightB + row] = bb.s7;
    bT[(col+ 8)*heightB + row] = bb.s8;
    bT[(col+ 9)*heightB + row] = bb.s9;
    bT[(col+10)*heightB + row] = bb.s10;
    bT[(col+11)*heightB + row] = bb.s11;
    bT[(col+12)*heightB + row] = bb.s12;
    bT[(col+13)*heightB + row] = bb.s13;
    bT[(col+14)*heightB + row] = bb.s14;
    bT[(col+15)*heightB + row] = bb.s15;

}

