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

/**************************** matrix add ******************************/

__kernel void mat_add_vec2(const int M,
                           const int N, 
                           __global float *c, 
                           float beta) {
    const int col = get_global_id(0)*2;
    const int row = get_global_id(1);

    float2 cc = vload2(0, (c+row*N+col));
    // beta = 1.3
    float2 res = (float2)beta * cc;
    vstore2(res, 0, c+row*N+col);
}

__kernel void mat_add(const int M,
                      const int N, 
                      __global CL_INPUT_TYPE *c, 
                      CL_INPUT_TYPE beta) {
    const int row = get_global_id(0);
    const int col = get_global_id(1)*4;

    CL_ELEM_TYPE cc = vload4(0, (c+row*N+col));
    // beta = 1.3
    CL_ELEM_TYPE res = (CL_ELEM_TYPE)beta * cc;
    vstore4(res, 0, c+row*N+col);
}

/**************************** matrix multiplication ******************************/

// 1024x1024x1024  [1024,256,1][1,1,1]  
// float4    0.654687 s 3.281771 GFLOPS
// half4     0.587530 s 3.656891 GFLOPS 
// 2048x2048x2048 [2048,512,1][1,1,1] 
// float4    5.435951 s 3.161188 GFLOPS 
// half4     5.049684 s 3.402998 GFLOPS 
__kernel void mat_mult_ma(const int M,
                          const int N,
                          const int K,
                          __global const CL_INPUT_TYPE *a,
                          __global const CL_INPUT_TYPE *b,
                          __global       CL_INPUT_TYPE *c) {
    const int row = get_global_id(0);
    const int col = get_global_id(1) * 4;

    CL_ELEM_TYPE sum = 0.0;

    for (int p = 0; p < K; p++) {
        CL_ELEM_TYPE aa = (CL_ELEM_TYPE)a[row*K+p];;//(__global CL_ELEM_TYPE *)(a + row * K + p);
        CL_ELEM_TYPE bb = vload4(0, b+p*N+col);

        sum += aa * bb;
    }
    vstore4(sum, 0, c+row*N+col);
}


// 1024x1024x1024  [1024,256,1][16,16,1] better than [4,4,1] and slightly better than [8,8,1]
// float4 0.200998 s 10.689332 GFLOPS  
// half4  0.192145 s 11.181833 GFLOPS   
// 2048x2048x2048  [2048,512,1][16,16,1] better than [4,4,1] and [8,8,1]
// float4 1.748242 s  9.829337 GFLOPS
// half4  1.609643 s 10.675699 GFLOPS 
__kernel void mat_mult_ma_vec1x4(const int M,
                                 const int N,
                                 const int K,
                                 __global const CL_INPUT_TYPE *a,
                                 __global const CL_INPUT_TYPE *b,
                                 __global       CL_INPUT_TYPE *c) {

    int const row = get_group_id(0) * WORK_GROUP_COL + get_local_id(0);  // 1024
    int const col = get_group_id(1) * WORK_GROUP_ROW + get_local_id(1);  // 256

    CL_ELEM_TYPE res = 0; 
    for (int p = 0; p < K; p++)
    {
        CL_ELEM_TYPE a_vec = (CL_ELEM_TYPE)a[row * K + p];
        CL_ELEM_TYPE b_vec = vload4(col, (__global CL_INPUT_TYPE *)(b + p * N));
        //CL_ELEM_TYPE b_vec = vload4(0, (__global CL_INPUT_TYPE *)(b + p * N + col));
        res += a_vec * b_vec;
    }
    vstore4(res, col, (__global CL_INPUT_TYPE *)(c + row * N));
    //vstore4(res, 0, (__global CL_INPUT_TYPE *)(c + row * N + col));
}

// 1024x1024x1024 [1024,256,1]
__kernel void mat_mult_global_ma_vec1x4(const int M,
                                        const int N,
                                        const int K,
                                        __global const CL_INPUT_TYPE *a,
                                        __global const CL_INPUT_TYPE *b,
                                        __global       CL_INPUT_TYPE *c) {

    int const row = get_global_id(0);      // 1024
    int const col = get_global_id(1) * 4;  // 256

    CL_ELEM_TYPE res = 0; 
    for (int p = 0; p < K; p++)
    {
        CL_ELEM_TYPE a_vec = (CL_ELEM_TYPE)a[row * K + p];
        CL_ELEM_TYPE b_vec = vload4(col, (__global CL_INPUT_TYPE *)(b + p * N));
        res += a_vec * b_vec;
    }
    vstore4(res, col, (__global CL_INPUT_TYPE *)(c + row * N));
}


////////////////////// without alpha & beta ////////////////////
// float4: 1024x1024x1024 0.259872 s  8.263633 GFLOPS
// half4:  1024x1024x1024 0.145462 s 14.763193 GFLOPS
// ACL:
// FP32:   1024x1024x1024 0.084823 s
// FP16:   1024x1024x1024 0.039247 s
// OpenBLAS OMP 1 A72
// FP32:   1024x1024x1024 0.193891 s 11.075726 GFLOPS
// FP16:
__kernel void mat_mult_vec4x4_continue(const int M,
                                       const int N, 
                                       const int K, 
                                       __global const CL_INPUT_TYPE *a, 
                                       __global const CL_INPUT_TYPE *b, 
                                       __global       CL_INPUT_TYPE *c) {
    const int row = get_global_id(0) << 2;
    const int col = get_global_id(1) << 2;

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

    //c[row * N + col] = cc1.s0;      c[row * N + (col+1)] = cc1.s1;      c[row * N + (col+2)] = cc1.s2;      c[row * N + (col+3)] = cc1.s3; 
    //c[(row+1) * N + col] = cc2.s0;  c[(row+1) * N + (col+1)] = cc2.s1;  c[(row+1) * N + (col+2)] = cc2.s2;  c[(row+1) * N + (col+3)] = cc2.s3; 
    //c[(row+2) * N + col] = cc3.s0;  c[(row+2) * N + (col+1)] = cc3.s1;  c[(row+2) * N + (col+2)] = cc3.s2;  c[(row+2) * N + (col+3)] = cc3.s3; 
    //c[(row+3) * N + col] = cc4.s0;  c[(row+3) * N + (col+1)] = cc4.s1;  c[(row+3) * N + (col+2)] = cc4.s2;  c[(row+3) * N + (col+3)] = cc4.s3;  

    // ALPHA c = alpha * c;
    *(__global CL_ELEM_TYPE *)(c + row * N + col)     = cc1;
    *(__global CL_ELEM_TYPE *)(c + (row+1) * N + col) = cc2;
    *(__global CL_ELEM_TYPE *)(c + (row+2) * N + col) = cc3;
    *(__global CL_ELEM_TYPE *)(c + (row+3) * N + col) = cc4;
}

////////////////////// with alpha & beta    ////////////////////
// float4:
// half4:
// ACL
// FP32
// FP16
__kernel void mat_mult_vec4x4_continue_alpha(const int M,
                                             const int N, 
                                             const int K, 
                                             __global const CL_INPUT_TYPE *a, 
                                             __global const CL_INPUT_TYPE *b, 
                                             __global       CL_INPUT_TYPE *c,
                                             CL_INPUT_TYPE alpha) {
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

    //c[row * N + col] = cc1.s0;      c[row * N + (col+1)] = cc1.s1;      c[row * N + (col+2)] = cc1.s2;      c[row * N + (col+3)] = cc1.s3; 
    //c[(row+1) * N + col] = cc2.s0;  c[(row+1) * N + (col+1)] = cc2.s1;  c[(row+1) * N + (col+2)] = cc2.s2;  c[(row+1) * N + (col+3)] = cc2.s3; 
    //c[(row+2) * N + col] = cc3.s0;  c[(row+2) * N + (col+1)] = cc3.s1;  c[(row+2) * N + (col+2)] = cc3.s2;  c[(row+2) * N + (col+3)] = cc3.s3; 
    //c[(row+3) * N + col] = cc4.s0;  c[(row+3) * N + (col+1)] = cc4.s1;  c[(row+3) * N + (col+2)] = cc4.s2;  c[(row+3) * N + (col+3)] = cc4.s3;  

    // ALPHA c = alpha * c;
    *(__global CL_ELEM_TYPE *)(c + row * N + col)     = (CL_ELEM_TYPE)(alpha) * cc1;
    *(__global CL_ELEM_TYPE *)(c + (row+1) * N + col) = (CL_ELEM_TYPE)(alpha) * cc2;
    *(__global CL_ELEM_TYPE *)(c + (row+2) * N + col) = (CL_ELEM_TYPE)(alpha) * cc3;
    *(__global CL_ELEM_TYPE *)(c + (row+3) * N + col) = (CL_ELEM_TYPE)(alpha) * cc4;
}


