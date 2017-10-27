#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// float 1024x1024x1024 1.224809 s 1.753322 GFLOPS
__kernel void mat_mult_vec4(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    CL_ELEM_TYPE res = 0;
    CL_ELEM_TYPE aa, bb;

    for (int p = 0; p < K; p += 4) {

        // 2 row elems: a[row * M + p], a[row * M + p + 1]
        aa = *((__global CL_ELEM_TYPE *)(a + row * M + p));

        // 2 col elems: b[p * N + col], b[(p+1) * N + col]
        bb = (CL_ELEM_TYPE)
                    (
                       *(__global CL_INPUT_TYPE *)(b + p * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+1) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+2) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+3) * N + col)
                    );
        res += aa * bb;
    }
    c[row * N + col] = res.s0 + res.s1 + res.s2 + res.s3;
}

// float4: 1024x1024x1024 0.259872 s 8.263633 GFLOPS
// half4:  1024x1024x1024 0.145462 s 14.763193 GFLOPS
__kernel void mat_mult_vec4x4_continue(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 2;
    const int row = get_global_id(1) << 2;

    CL_ELEM_TYPE aa1, aa2, aa3, aa4,
                 bb1, bb2, bb3, bb4,
                 cc1 = 0, cc2 = 0, cc3 = 0, cc4 = 0;
    for (int p = 0; p < K; p += 4) {
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

        cc1.s0 += aa1.s0 * bb1.s0 +
                  aa1.s1 * bb2.s0 + 
                  aa1.s2 * bb3.s0 +
                  aa1.s3 * bb4.s0;
        cc1.s1 += aa1.s0 * bb1.s1 +
                  aa1.s1 * bb2.s1 +
                  aa1.s2 * bb3.s1 +
                  aa1.s3 * bb4.s1;
        cc1.s2 += aa1.s0 * bb1.s2 +
                  aa1.s1 * bb2.s2 +
                  aa1.s2 * bb3.s2 +
                  aa1.s3 * bb4.s2;
        cc1.s3 += aa1.s0 * bb1.s3 +
                  aa1.s1 * bb2.s3 +
                  aa1.s2 * bb3.s3 +
                  aa1.s3 * bb4.s3;

        cc2.s0 += aa2.s0 * bb1.s0 +
                  aa2.s1 * bb2.s0 + 
                  aa2.s2 * bb3.s0 +
                  aa2.s3 * bb4.s0;
        cc2.s1 += aa2.s0 * bb1.s1 +
                  aa2.s1 * bb2.s1 +
                  aa2.s2 * bb3.s1 +
                  aa2.s3 * bb4.s1;
        cc2.s2 += aa2.s0 * bb1.s2 +
                  aa2.s1 * bb2.s2 +
                  aa2.s2 * bb3.s2 +
                  aa2.s3 * bb4.s2;
        cc2.s3 += aa2.s0 * bb1.s3 +
                  aa2.s1 * bb2.s3 +
                  aa2.s2 * bb3.s3 +
                  aa2.s3 * bb4.s3;

        cc3.s0 += aa3.s0 * bb1.s0 +
                  aa3.s1 * bb2.s0 + 
                  aa3.s2 * bb3.s0 +
                  aa3.s3 * bb4.s0;
        cc3.s1 += aa3.s0 * bb1.s1 +
                  aa3.s1 * bb2.s1 +
                  aa3.s2 * bb3.s1 +
                  aa3.s3 * bb4.s1;
        cc3.s2 += aa3.s0 * bb1.s2 +
                  aa3.s1 * bb2.s2 +
                  aa3.s2 * bb3.s2 +
                  aa3.s3 * bb4.s2;
        cc3.s3 += aa3.s0 * bb1.s3 +
                  aa3.s1 * bb2.s3 +
                  aa3.s2 * bb3.s3 +
                  aa3.s3 * bb4.s3;

        cc4.s0 += aa4.s0 * bb1.s0 +
                  aa4.s1 * bb2.s0 + 
                  aa4.s2 * bb3.s0 +
                  aa4.s3 * bb4.s0;
        cc4.s1 += aa4.s0 * bb1.s1 +
                  aa4.s1 * bb2.s1 +
                  aa4.s2 * bb3.s1 +
                  aa4.s3 * bb4.s1;
        cc4.s2 += aa4.s0 * bb1.s2 +
                  aa4.s1 * bb2.s2 +
                  aa4.s2 * bb3.s2 +
                  aa4.s3 * bb4.s2;
        cc4.s3 += aa4.s0 * bb1.s3 +
                  aa4.s1 * bb2.s3 +
                  aa4.s2 * bb3.s3 +
                  aa4.s3 * bb4.s3;

    }

    c[row * N + col] = cc1.s0;      c[row * N + (col+1)] = cc1.s1;      c[row * N + (col+2)] = cc1.s2;      c[row * N + (col+3)] = cc1.s3; 
    c[(row+1) * N + col] = cc2.s0;  c[(row+1) * N + (col+1)] = cc2.s1;  c[(row+1) * N + (col+2)] = cc2.s2;  c[(row+1) * N + (col+3)] = cc2.s3; 
    c[(row+2) * N + col] = cc3.s0;  c[(row+2) * N + (col+1)] = cc3.s1;  c[(row+2) * N + (col+2)] = cc3.s2;  c[(row+2) * N + (col+3)] = cc3.s3; 
    c[(row+3) * N + col] = cc4.s0;  c[(row+3) * N + (col+1)] = cc4.s1;  c[(row+3) * N + (col+2)] = cc4.s2;  c[(row+3) * N + (col+3)] = cc4.s3;  

    //*(__global CL_ELEM_TYPE *)(c + row * N + col) = cc1;
    //*(__global CL_ELEM_TYPE *)(c + (row+1) * N + col) = cc2;
    //*(__global CL_ELEM_TYPE *)(c + (row+2) * N + col) = cc3;
    //*(__global CL_ELEM_TYPE *)(c + (row+3) * N + col) = cc4;
}
