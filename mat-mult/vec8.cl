#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void mat_mult_vec8(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    CL_ELEM_TYPE res = 0;
    CL_ELEM_TYPE aa, bb;

    for (int p = 0; p < K; p += 8) {

        // 2 row elems: a[row * M + p], a[row * M + p + 1]
        aa = *((__global CL_ELEM_TYPE *)(a + row * M + p));

        // 2 col elems: b[p * N + col], b[(p+1) * N + col]
        bb = (CL_ELEM_TYPE)
                    (
                       *(__global CL_INPUT_TYPE *)(b + p * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+1) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+2) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+3) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+4) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+5) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+6) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+7) * N + col)
                    );
        res += aa * bb;
    }
    c[row * N + col] = res.s0 + res.s1 + res.s2 + res.s3 + res.s4 + res.s5 + res.s6 + res.s7;
}


// float8 1024x1024x1024 0.860143 s 2.496659 GFLOPS
// half8 1024x1024x1024 0.162382 s 13.224895 GFLOPS

__kernel void mat_mult_vec8x8_continue(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 3;
    const int row = get_global_id(1) << 3;

    CL_ELEM_TYPE aa1, aa2, aa3, aa4, aa5, aa6, aa7, aa8,
                 bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8,
                 cc1 = 0, cc2 = 0, cc3 = 0, cc4 = 0, cc5 = 0, cc6 = 0, cc7 = 0, cc8 = 0;

    for (int p = 0; p < K; p += 8) {
        aa1 = *(
                   (__global CL_ELEM_TYPE *)(a + row * K + p)
               );
        aa2 = *(
                   (__global CL_ELEM_TYPE *)(a + (row+1) * K + p)
               );
        aa3 = *(
                   (__global CL_ELEM_TYPE *)(a + (row+2) * K + p)
               );  
        aa4 = *(
                   (__global CL_ELEM_TYPE *)(a + (row+3) * K + p)
               );
        aa5 = *(
                   (__global CL_ELEM_TYPE *)(a + (row+4) * K + p)
               );
        aa6 = *(
                   (__global CL_ELEM_TYPE *)(a + (row+5) * K + p)
               );
        aa7 = *(
                   (__global CL_ELEM_TYPE *)(a + (row+6) * K + p)
               );  
        aa8 = *(
                   (__global CL_ELEM_TYPE *)(a + (row+7) * K + p)
               );


        bb1 = *(
                   (__global CL_ELEM_TYPE *)(b + p * N + col)
               );  
        bb2 = *(
                   (__global CL_ELEM_TYPE *)(b + (p+1) * N + col)
               );  
        bb3 = *(
                   (__global CL_ELEM_TYPE *)(b + (p+2) * N + col)
               );
        bb4 = *(
                   (__global CL_ELEM_TYPE *)(b + (p+3) * N + col)
               );  
        bb5 = *(
                   (__global CL_ELEM_TYPE *)(b + (p+4) * N + col)
               );  
        bb6 = *(
                   (__global CL_ELEM_TYPE *)(b + (p+5) * N + col)
               );
        bb7 = *(
                   (__global CL_ELEM_TYPE *)(b + (p+6) * N + col)
               );  
        bb8 = *(
                   (__global CL_ELEM_TYPE *)(b + (p+7) * N + col)
               );  

        cc1.s0 += aa1.s0 * bb1.s0 +
                  aa1.s1 * bb2.s0 +
                  aa1.s2 * bb3.s0 +
                  aa1.s3 * bb4.s0 +
                  aa1.s4 * bb5.s0 +
                  aa1.s5 * bb6.s0 + 
                  aa1.s6 * bb7.s0 +
                  aa1.s7 * bb8.s0;
        cc1.s1 += aa1.s0 * bb1.s1 +
                  aa1.s1 * bb2.s1 +
                  aa1.s2 * bb3.s1 +
                  aa1.s3 * bb4.s1 +
                  aa1.s4 * bb5.s1 +
                  aa1.s5 * bb6.s1 +
                  aa1.s6 * bb7.s1 +
                  aa1.s7 * bb8.s1;
        cc1.s2 += aa1.s0 * bb1.s2 +
                  aa1.s1 * bb2.s2 +
                  aa1.s2 * bb3.s2 +
                  aa1.s3 * bb4.s2 +
                  aa1.s4 * bb5.s2 +
                  aa1.s5 * bb6.s2 +
                  aa1.s6 * bb7.s2 +
                  aa1.s7 * bb8.s2;
        cc1.s3 += aa1.s0 * bb1.s3 +
                  aa1.s1 * bb2.s3 +
                  aa1.s2 * bb3.s3 +
                  aa1.s3 * bb4.s3 +
                  aa1.s4 * bb5.s3 +
                  aa1.s5 * bb6.s3 +
                  aa1.s6 * bb7.s3 +
                  aa1.s7 * bb8.s3;
        cc1.s4 += aa1.s0 * bb1.s4 +
                  aa1.s1 * bb2.s4 +
                  aa1.s2 * bb3.s4 +
                  aa1.s3 * bb4.s4 +
                  aa1.s4 * bb5.s4 +
                  aa1.s5 * bb6.s4 + 
                  aa1.s6 * bb7.s4 +
                  aa1.s7 * bb8.s4;
        cc1.s5 += aa1.s0 * bb1.s5 +
                  aa1.s1 * bb2.s5 +
                  aa1.s2 * bb3.s5 +
                  aa1.s3 * bb4.s5 +
                  aa1.s4 * bb5.s5 +
                  aa1.s5 * bb6.s5 +
                  aa1.s6 * bb7.s5 +
                  aa1.s7 * bb8.s5;
        cc1.s6 += aa1.s0 * bb1.s6 +
                  aa1.s1 * bb2.s6 +
                  aa1.s2 * bb3.s6 +
                  aa1.s3 * bb4.s6 +
                  aa1.s4 * bb5.s6 +
                  aa1.s5 * bb6.s6 +
                  aa1.s6 * bb7.s6 +
                  aa1.s7 * bb8.s6;
        cc1.s7 += aa1.s0 * bb1.s7 +
                  aa1.s1 * bb2.s7 +
                  aa1.s2 * bb3.s7 +
                  aa1.s3 * bb4.s7 +
                  aa1.s4 * bb5.s7 +
                  aa1.s5 * bb6.s7 +
                  aa1.s6 * bb7.s7 +
                  aa1.s7 * bb8.s7;


        cc2.s0 += aa2.s0 * bb1.s0 +
                  aa2.s1 * bb2.s0 +
                  aa2.s2 * bb3.s0 +
                  aa2.s3 * bb4.s0 +
                  aa2.s4 * bb5.s0 +
                  aa2.s5 * bb6.s0 + 
                  aa2.s6 * bb7.s0 +
                  aa2.s7 * bb8.s0;
        cc2.s1 += aa2.s0 * bb1.s1 +
                  aa2.s1 * bb2.s1 +
                  aa2.s2 * bb3.s1 +
                  aa2.s3 * bb4.s1 +
                  aa2.s4 * bb5.s1 +
                  aa2.s5 * bb6.s1 +
                  aa2.s6 * bb7.s1 +
                  aa2.s7 * bb8.s1;
        cc2.s2 += aa2.s0 * bb1.s2 +
                  aa2.s1 * bb2.s2 +
                  aa2.s2 * bb3.s2 +
                  aa2.s3 * bb4.s2 +
                  aa2.s4 * bb5.s2 +
                  aa2.s5 * bb6.s2 +
                  aa2.s6 * bb7.s2 +
                  aa2.s7 * bb8.s2;
        cc2.s3 += aa2.s0 * bb1.s3 +
                  aa2.s1 * bb2.s3 +
                  aa2.s2 * bb3.s3 +
                  aa2.s3 * bb4.s3 +
                  aa2.s4 * bb5.s3 +
                  aa2.s5 * bb6.s3 +
                  aa2.s6 * bb7.s3 +
                  aa2.s7 * bb8.s3;
        cc2.s4 += aa2.s0 * bb1.s4 +
                  aa2.s1 * bb2.s4 +
                  aa2.s2 * bb3.s4 +
                  aa2.s3 * bb4.s4 +
                  aa2.s4 * bb5.s4 +
                  aa2.s5 * bb6.s4 + 
                  aa2.s6 * bb7.s4 +
                  aa2.s7 * bb8.s4;
        cc2.s5 += aa2.s0 * bb1.s5 +
                  aa2.s1 * bb2.s5 +
                  aa2.s2 * bb3.s5 +
                  aa2.s3 * bb4.s5 +
                  aa2.s4 * bb5.s5 +
                  aa2.s5 * bb6.s5 +
                  aa2.s6 * bb7.s5 +
                  aa2.s7 * bb8.s5;
        cc2.s6 += aa2.s0 * bb1.s6 +
                  aa2.s1 * bb2.s6 +
                  aa2.s2 * bb3.s6 +
                  aa2.s3 * bb4.s6 +
                  aa2.s4 * bb5.s6 +
                  aa2.s5 * bb6.s6 +
                  aa2.s6 * bb7.s6 +
                  aa2.s7 * bb8.s6;
        cc2.s7 += aa2.s0 * bb1.s7 +
                  aa2.s1 * bb2.s7 +
                  aa2.s2 * bb3.s7 +
                  aa2.s3 * bb4.s7 +
                  aa2.s4 * bb5.s7 +
                  aa2.s5 * bb6.s7 +
                  aa2.s6 * bb7.s7 +
                  aa2.s7 * bb8.s7;

        cc3.s0 += aa3.s0 * bb1.s0 +
                  aa3.s1 * bb2.s0 +
                  aa3.s2 * bb3.s0 +
                  aa3.s3 * bb4.s0 +
                  aa3.s4 * bb5.s0 +
                  aa3.s5 * bb6.s0 + 
                  aa3.s6 * bb7.s0 +
                  aa3.s7 * bb8.s0;
        cc3.s1 += aa3.s0 * bb1.s1 +
                  aa3.s1 * bb2.s1 +
                  aa3.s2 * bb3.s1 +
                  aa3.s3 * bb4.s1 +
                  aa3.s4 * bb5.s1 +
                  aa3.s5 * bb6.s1 +
                  aa3.s6 * bb7.s1 +
                  aa3.s7 * bb8.s1;
        cc3.s2 += aa3.s0 * bb1.s2 +
                  aa3.s1 * bb2.s2 +
                  aa3.s2 * bb3.s2 +
                  aa3.s3 * bb4.s2 +
                  aa3.s4 * bb5.s2 +
                  aa3.s5 * bb6.s2 +
                  aa3.s6 * bb7.s2 +
                  aa3.s7 * bb8.s2;
        cc3.s3 += aa3.s0 * bb1.s3 +
                  aa3.s1 * bb2.s3 +
                  aa3.s2 * bb3.s3 +
                  aa3.s3 * bb4.s3 +
                  aa3.s4 * bb5.s3 +
                  aa3.s5 * bb6.s3 +
                  aa3.s6 * bb7.s3 +
                  aa3.s7 * bb8.s3;
        cc3.s4 += aa3.s0 * bb1.s4 +
                  aa3.s1 * bb2.s4 +
                  aa3.s2 * bb3.s4 +
                  aa3.s3 * bb4.s4 +
                  aa3.s4 * bb5.s4 +
                  aa3.s5 * bb6.s4 + 
                  aa3.s6 * bb7.s4 +
                  aa3.s7 * bb8.s4;
        cc3.s5 += aa3.s0 * bb1.s5 +
                  aa3.s1 * bb2.s5 +
                  aa3.s2 * bb3.s5 +
                  aa3.s3 * bb4.s5 +
                  aa3.s4 * bb5.s5 +
                  aa3.s5 * bb6.s5 +
                  aa3.s6 * bb7.s5 +
                  aa3.s7 * bb8.s5;
        cc3.s6 += aa3.s0 * bb1.s6 +
                  aa3.s1 * bb2.s6 +
                  aa3.s2 * bb3.s6 +
                  aa3.s3 * bb4.s6 +
                  aa3.s4 * bb5.s6 +
                  aa3.s5 * bb6.s6 +
                  aa3.s6 * bb7.s6 +
                  aa3.s7 * bb8.s6;
        cc3.s7 += aa3.s0 * bb1.s7 +
                  aa3.s1 * bb2.s7 +
                  aa3.s2 * bb3.s7 +
                  aa3.s3 * bb4.s7 +
                  aa3.s4 * bb5.s7 +
                  aa3.s5 * bb6.s7 +
                  aa3.s6 * bb7.s7 +
                  aa3.s7 * bb8.s7;

        cc4.s0 += aa4.s0 * bb1.s0 +
                  aa4.s1 * bb2.s0 +
                  aa4.s2 * bb3.s0 +
                  aa4.s3 * bb4.s0 +
                  aa4.s4 * bb5.s0 +
                  aa4.s5 * bb6.s0 + 
                  aa4.s6 * bb7.s0 +
                  aa4.s7 * bb8.s0;
        cc4.s1 += aa4.s0 * bb1.s1 +
                  aa4.s1 * bb2.s1 +
                  aa4.s2 * bb3.s1 +
                  aa4.s3 * bb4.s1 +
                  aa4.s4 * bb5.s1 +
                  aa4.s5 * bb6.s1 +
                  aa4.s6 * bb7.s1 +
                  aa4.s7 * bb8.s1;
        cc4.s2 += aa4.s0 * bb1.s2 +
                  aa4.s1 * bb2.s2 +
                  aa4.s2 * bb3.s2 +
                  aa4.s3 * bb4.s2 +
                  aa4.s4 * bb5.s2 +
                  aa4.s5 * bb6.s2 +
                  aa4.s6 * bb7.s2 +
                  aa4.s7 * bb8.s2;
        cc4.s3 += aa4.s0 * bb1.s3 +
                  aa4.s1 * bb2.s3 +
                  aa4.s2 * bb3.s3 +
                  aa4.s3 * bb4.s3 +
                  aa4.s4 * bb5.s3 +
                  aa4.s5 * bb6.s3 +
                  aa4.s6 * bb7.s3 +
                  aa4.s7 * bb8.s3;
        cc4.s4 += aa4.s0 * bb1.s4 +
                  aa4.s1 * bb2.s4 +
                  aa4.s2 * bb3.s4 +
                  aa4.s3 * bb4.s4 +
                  aa4.s4 * bb5.s4 +
                  aa4.s5 * bb6.s4 + 
                  aa4.s6 * bb7.s4 +
                  aa4.s7 * bb8.s4;
        cc4.s5 += aa4.s0 * bb1.s5 +
                  aa4.s1 * bb2.s5 +
                  aa4.s2 * bb3.s5 +
                  aa4.s3 * bb4.s5 +
                  aa4.s4 * bb5.s5 +
                  aa4.s5 * bb6.s5 +
                  aa4.s6 * bb7.s5 +
                  aa4.s7 * bb8.s5;
        cc4.s6 += aa4.s0 * bb1.s6 +
                  aa4.s1 * bb2.s6 +
                  aa4.s2 * bb3.s6 +
                  aa4.s3 * bb4.s6 +
                  aa4.s4 * bb5.s6 +
                  aa4.s5 * bb6.s6 +
                  aa4.s6 * bb7.s6 +
                  aa4.s7 * bb8.s6;
        cc4.s7 += aa4.s0 * bb1.s7 +
                  aa4.s1 * bb2.s7 +
                  aa4.s2 * bb3.s7 +
                  aa4.s3 * bb4.s7 +
                  aa4.s4 * bb5.s7 +
                  aa4.s5 * bb6.s7 +
                  aa4.s6 * bb7.s7 +
                  aa4.s7 * bb8.s7;


        cc5.s0 += aa5.s0 * bb1.s0 +
                  aa5.s1 * bb2.s0 +
                  aa5.s2 * bb3.s0 +
                  aa5.s3 * bb4.s0 +
                  aa5.s4 * bb5.s0 +
                  aa5.s5 * bb6.s0 + 
                  aa5.s6 * bb7.s0 +
                  aa5.s7 * bb8.s0;
        cc5.s1 += aa5.s0 * bb1.s1 +
                  aa5.s1 * bb2.s1 +
                  aa5.s2 * bb3.s1 +
                  aa5.s3 * bb4.s1 +
                  aa5.s4 * bb5.s1 +
                  aa5.s5 * bb6.s1 +
                  aa5.s6 * bb7.s1 +
                  aa5.s7 * bb8.s1;
        cc5.s2 += aa5.s0 * bb1.s2 +
                  aa5.s1 * bb2.s2 +
                  aa5.s2 * bb3.s2 +
                  aa5.s3 * bb4.s2 +
                  aa5.s4 * bb5.s2 +
                  aa5.s5 * bb6.s2 +
                  aa5.s6 * bb7.s2 +
                  aa5.s7 * bb8.s2;
        cc5.s3 += aa5.s0 * bb1.s3 +
                  aa5.s1 * bb2.s3 +
                  aa5.s2 * bb3.s3 +
                  aa5.s3 * bb4.s3 +
                  aa5.s4 * bb5.s3 +
                  aa5.s5 * bb6.s3 +
                  aa5.s6 * bb7.s3 +
                  aa5.s7 * bb8.s3;
        cc5.s4 += aa5.s0 * bb1.s4 +
                  aa5.s1 * bb2.s4 +
                  aa5.s2 * bb3.s4 +
                  aa5.s3 * bb4.s4 +
                  aa5.s4 * bb5.s4 +
                  aa5.s5 * bb6.s4 + 
                  aa5.s6 * bb7.s4 +
                  aa5.s7 * bb8.s4;
        cc5.s5 += aa5.s0 * bb1.s5 +
                  aa5.s1 * bb2.s5 +
                  aa5.s2 * bb3.s5 +
                  aa5.s3 * bb4.s5 +
                  aa5.s4 * bb5.s5 +
                  aa5.s5 * bb6.s5 +
                  aa5.s6 * bb7.s5 +
                  aa5.s7 * bb8.s5;
        cc5.s6 += aa5.s0 * bb1.s6 +
                  aa5.s1 * bb2.s6 +
                  aa5.s2 * bb3.s6 +
                  aa5.s3 * bb4.s6 +
                  aa5.s4 * bb5.s6 +
                  aa5.s5 * bb6.s6 +
                  aa5.s6 * bb7.s6 +
                  aa5.s7 * bb8.s6;
        cc5.s7 += aa5.s0 * bb1.s7 +
                  aa5.s1 * bb2.s7 +
                  aa5.s2 * bb3.s7 +
                  aa5.s3 * bb4.s7 +
                  aa5.s4 * bb5.s7 +
                  aa5.s5 * bb6.s7 +
                  aa5.s6 * bb7.s7 +
                  aa5.s7 * bb8.s7;

        cc6.s0 += aa6.s0 * bb1.s0 +
                  aa6.s1 * bb2.s0 +
                  aa6.s2 * bb3.s0 +
                  aa6.s3 * bb4.s0 +
                  aa6.s4 * bb5.s0 +
                  aa6.s5 * bb6.s0 + 
                  aa6.s6 * bb7.s0 +
                  aa6.s7 * bb8.s0;
        cc6.s1 += aa6.s0 * bb1.s1 +
                  aa6.s1 * bb2.s1 +
                  aa6.s2 * bb3.s1 +
                  aa6.s3 * bb4.s1 +
                  aa6.s4 * bb5.s1 +
                  aa6.s5 * bb6.s1 +
                  aa6.s6 * bb7.s1 +
                  aa6.s7 * bb8.s1;
        cc6.s2 += aa6.s0 * bb1.s2 +
                  aa6.s1 * bb2.s2 +
                  aa6.s2 * bb3.s2 +
                  aa6.s3 * bb4.s2 +
                  aa6.s4 * bb5.s2 +
                  aa6.s5 * bb6.s2 +
                  aa6.s6 * bb7.s2 +
                  aa6.s7 * bb8.s2;
        cc6.s3 += aa6.s0 * bb1.s3 +
                  aa6.s1 * bb2.s3 +
                  aa6.s2 * bb3.s3 +
                  aa6.s3 * bb4.s3 +
                  aa6.s4 * bb5.s3 +
                  aa6.s5 * bb6.s3 +
                  aa6.s6 * bb7.s3 +
                  aa6.s7 * bb8.s3;
        cc6.s4 += aa6.s0 * bb1.s4 +
                  aa6.s1 * bb2.s4 +
                  aa6.s2 * bb3.s4 +
                  aa6.s3 * bb4.s4 +
                  aa6.s4 * bb5.s4 +
                  aa6.s5 * bb6.s4 + 
                  aa6.s6 * bb7.s4 +
                  aa6.s7 * bb8.s4;
        cc6.s5 += aa6.s0 * bb1.s5 +
                  aa6.s1 * bb2.s5 +
                  aa6.s2 * bb3.s5 +
                  aa6.s3 * bb4.s5 +
                  aa6.s4 * bb5.s5 +
                  aa6.s5 * bb6.s5 +
                  aa6.s6 * bb7.s5 +
                  aa6.s7 * bb8.s5;
        cc6.s6 += aa6.s0 * bb1.s6 +
                  aa6.s1 * bb2.s6 +
                  aa6.s2 * bb3.s6 +
                  aa6.s3 * bb4.s6 +
                  aa6.s4 * bb5.s6 +
                  aa6.s5 * bb6.s6 +
                  aa6.s6 * bb7.s6 +
                  aa6.s7 * bb8.s6;
        cc6.s7 += aa6.s0 * bb1.s7 +
                  aa6.s1 * bb2.s7 +
                  aa6.s2 * bb3.s7 +
                  aa6.s3 * bb4.s7 +
                  aa6.s4 * bb5.s7 +
                  aa6.s5 * bb6.s7 +
                  aa6.s6 * bb7.s7 +
                  aa6.s7 * bb8.s7;


        cc7.s0 += aa7.s0 * bb1.s0 +
                  aa7.s1 * bb2.s0 +
                  aa7.s2 * bb3.s0 +
                  aa7.s3 * bb4.s0 +
                  aa7.s4 * bb5.s0 +
                  aa7.s5 * bb6.s0 + 
                  aa7.s6 * bb7.s0 +
                  aa7.s7 * bb8.s0;
        cc7.s1 += aa7.s0 * bb1.s1 +
                  aa7.s1 * bb2.s1 +
                  aa7.s2 * bb3.s1 +
                  aa7.s3 * bb4.s1 +
                  aa7.s4 * bb5.s1 +
                  aa7.s5 * bb6.s1 +
                  aa7.s6 * bb7.s1 +
                  aa7.s7 * bb8.s1;
        cc7.s2 += aa7.s0 * bb1.s2 +
                  aa7.s1 * bb2.s2 +
                  aa7.s2 * bb3.s2 +
                  aa7.s3 * bb4.s2 +
                  aa7.s4 * bb5.s2 +
                  aa7.s5 * bb6.s2 +
                  aa7.s6 * bb7.s2 +
                  aa7.s7 * bb8.s2;
        cc7.s3 += aa7.s0 * bb1.s3 +
                  aa7.s1 * bb2.s3 +
                  aa7.s2 * bb3.s3 +
                  aa7.s3 * bb4.s3 +
                  aa7.s4 * bb5.s3 +
                  aa7.s5 * bb6.s3 +
                  aa7.s6 * bb7.s3 +
                  aa7.s7 * bb8.s3;
        cc7.s4 += aa7.s0 * bb1.s4 +
                  aa7.s1 * bb2.s4 +
                  aa7.s2 * bb3.s4 +
                  aa7.s3 * bb4.s4 +
                  aa7.s4 * bb5.s4 +
                  aa7.s5 * bb6.s4 + 
                  aa7.s6 * bb7.s4 +
                  aa7.s7 * bb8.s4;
        cc7.s5 += aa7.s0 * bb1.s5 +
                  aa7.s1 * bb2.s5 +
                  aa7.s2 * bb3.s5 +
                  aa7.s3 * bb4.s5 +
                  aa7.s4 * bb5.s5 +
                  aa7.s5 * bb6.s5 +
                  aa7.s6 * bb7.s5 +
                  aa7.s7 * bb8.s5;
        cc7.s6 += aa7.s0 * bb1.s6 +
                  aa7.s1 * bb2.s6 +
                  aa7.s2 * bb3.s6 +
                  aa7.s3 * bb4.s6 +
                  aa7.s4 * bb5.s6 +
                  aa7.s5 * bb6.s6 +
                  aa7.s6 * bb7.s6 +
                  aa7.s7 * bb8.s6;
        cc7.s7 += aa7.s0 * bb1.s7 +
                  aa7.s1 * bb2.s7 +
                  aa7.s2 * bb3.s7 +
                  aa7.s3 * bb4.s7 +
                  aa7.s4 * bb5.s7 +
                  aa7.s5 * bb6.s7 +
                  aa7.s6 * bb7.s7 +
                  aa7.s7 * bb8.s7;


        cc8.s0 += aa8.s0 * bb1.s0 +
                  aa8.s1 * bb2.s0 +
                  aa8.s2 * bb3.s0 +
                  aa8.s3 * bb4.s0 +
                  aa8.s4 * bb5.s0 +
                  aa8.s5 * bb6.s0 + 
                  aa8.s6 * bb7.s0 +
                  aa8.s7 * bb8.s0;
        cc8.s1 += aa8.s0 * bb1.s1 +
                  aa8.s1 * bb2.s1 +
                  aa8.s2 * bb3.s1 +
                  aa8.s3 * bb4.s1 +
                  aa8.s4 * bb5.s1 +
                  aa8.s5 * bb6.s1 +
                  aa8.s6 * bb7.s1 +
                  aa8.s7 * bb8.s1;
        cc8.s2 += aa8.s0 * bb1.s2 +
                  aa8.s1 * bb2.s2 +
                  aa8.s2 * bb3.s2 +
                  aa8.s3 * bb4.s2 +
                  aa8.s4 * bb5.s2 +
                  aa8.s5 * bb6.s2 +
                  aa8.s6 * bb7.s2 +
                  aa8.s7 * bb8.s2;
        cc8.s3 += aa8.s0 * bb1.s3 +
                  aa8.s1 * bb2.s3 +
                  aa8.s2 * bb3.s3 +
                  aa8.s3 * bb4.s3 +
                  aa8.s4 * bb5.s3 +
                  aa8.s5 * bb6.s3 +
                  aa8.s6 * bb7.s3 +
                  aa8.s7 * bb8.s3;
        cc8.s4 += aa8.s0 * bb1.s4 +
                  aa8.s1 * bb2.s4 +
                  aa8.s2 * bb3.s4 +
                  aa8.s3 * bb4.s4 +
                  aa8.s4 * bb5.s4 +
                  aa8.s5 * bb6.s4 + 
                  aa8.s6 * bb7.s4 +
                  aa8.s7 * bb8.s4;
        cc8.s5 += aa8.s0 * bb1.s5 +
                  aa8.s1 * bb2.s5 +
                  aa8.s2 * bb3.s5 +
                  aa8.s3 * bb4.s5 +
                  aa8.s4 * bb5.s5 +
                  aa8.s5 * bb6.s5 +
                  aa8.s6 * bb7.s5 +
                  aa8.s7 * bb8.s5;
        cc8.s6 += aa8.s0 * bb1.s6 +
                  aa8.s1 * bb2.s6 +
                  aa8.s2 * bb3.s6 +
                  aa8.s3 * bb4.s6 +
                  aa8.s4 * bb5.s6 +
                  aa8.s5 * bb6.s6 +
                  aa8.s6 * bb7.s6 +
                  aa8.s7 * bb8.s6;
        cc8.s7 += aa8.s0 * bb1.s7 +
                  aa8.s1 * bb2.s7 +
                  aa8.s2 * bb3.s7 +
                  aa8.s3 * bb4.s7 +
                  aa8.s4 * bb5.s7 +
                  aa8.s5 * bb6.s7 +
                  aa8.s6 * bb7.s7 +
                  aa8.s7 * bb8.s7;
    }

    *(__global CL_ELEM_TYPE *)(c + row * N + col) = cc1;
    *(__global CL_ELEM_TYPE *)(c + (row+1) * N + col) = cc2;
    *(__global CL_ELEM_TYPE *)(c + (row+2) * N + col) = cc3;
    *(__global CL_ELEM_TYPE *)(c + (row+3) * N + col) = cc4;
    *(__global CL_ELEM_TYPE *)(c + (row+4) * N + col) = cc5;
    *(__global CL_ELEM_TYPE *)(c + (row+5) * N + col) = cc6;
    *(__global CL_ELEM_TYPE *)(c + (row+6) * N + col) = cc7;
    *(__global CL_ELEM_TYPE *)(c + (row+7) * N + col) = cc8;

}
