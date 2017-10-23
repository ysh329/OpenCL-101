#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void mat_mult_vec2(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    CL_ELEM_TYPE res = 0;
    CL_ELEM_TYPE aa, bb;

    for (int p = 0; p < K; p += 2) {

        // 2 row elems: a[row * M + p], a[row * M + p + 1]
        aa = *((__global CL_ELEM_TYPE *)(a + row * M + p));

        // 2 col elems: b[p * N + col], b[(p+1) * N + col]
        bb = (CL_ELEM_TYPE)
                    (
                       *(__global CL_INPUT_TYPE *)(b + p * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+1) * N + col)
                    );
        res += aa * bb;
    }
    c[row * N + col] = res.s0 + res.s1;
}


__kernel void mat_mult_vec2x1(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1) << 1;

    CL_ELEM_TYPE aa1, aa2, bb, res1 = 0, res2 = 0;

    for (int p = 0; p < K; p+=2) {
        aa1 = *((__global CL_ELEM_TYPE *)(a + row * M + p));
        aa2 = *((__global CL_ELEM_TYPE *)(a + (row+1) * M + p));

        bb = (CL_ELEM_TYPE)
                    (
                       *(__global CL_INPUT_TYPE *)(b + p * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+1) * N + col)
                    );
        
        res1 += aa1 * bb;
        res2 += aa2 * bb;
    }
    c[row * N + col] = res1.s0 + res1.s1;
    c[(row+1) * N + col] = res2.s0 + res2.s1;
}


__kernel void mat_mult_vec1x2(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 1;
    const int row = get_global_id(1);

    CL_ELEM_TYPE aa,
                 bb1, bb2,
                 cc1 = 0, cc2 = 0;

    for (int p = 0; p < K; p+=2) {
        aa = *((__global CL_ELEM_TYPE *)(a + row * M + p));

        bb1 = (CL_ELEM_TYPE)
                    (
                       *(__global CL_INPUT_TYPE *)(b + p * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+1) * N + col)
                    );
        bb2 = (CL_ELEM_TYPE)
                    (
                       *(__global CL_INPUT_TYPE *)(b + p * N + (col+1) ),
                       *(__global CL_INPUT_TYPE *)(b + (p+1) * N + (col+1) )
                    );       
        cc1 += aa * bb1;
        cc2 += aa * bb2;
    }
    c[row * N + col] = cc1.s0 + cc1.s1;
    c[row * N + (col+1)] = cc2.s0 + cc2.s1;
}


// TODO: Strange, I think it's right!
__kernel void mat_mult_vec1x2_continue(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(0);

    CL_ELEM_TYPE aa, bb1, bb2, cc = 0;

    for (int p = 0; p < K; p += 2) {
        aa = *(
                  (__global CL_ELEM_TYPE *)(a + row * M + p) 
              );

        bb1 = *( 
                  (__global CL_ELEM_TYPE *)(b + p * N + col) 
              );
        bb2 = *( 
                  (__global CL_ELEM_TYPE *)(b + (p+1) * N + col) 
              );
        cc.s0 += aa.s0 * bb1.s0 + aa.s1 * bb2.s0;
        cc.s1 += aa.s0 + bb1.s1 + aa.s1 * bb2.s1;
    }
    c[row * N + col] = cc.s0;
    c[row * N + (col+1)] = cc.s1;
}


// perf up 11%! 1024x1024x1024 float
// float: naive: 0.59s this: 0.53s
__kernel void mat_mult_vec2x2(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0) << 1;
    const int row = get_global_id(1) << 1;

    CL_ELEM_TYPE aa1, aa2, bb1, bb2, 
                 c_00 = 0, c_01 = 0,
                 c_10 = 0, c_11 = 0;

    for (int p = 0; p < K; p+=2) {
        aa1 = *((__global CL_ELEM_TYPE *)(a + row * M + p));
        aa2 = *((__global CL_ELEM_TYPE *)(a + (row+1) * M + p));

        bb1 = (CL_ELEM_TYPE)
                    (
                       *(__global CL_INPUT_TYPE *)(b + p * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+1) * N + col)
                    );
        bb2 = (CL_ELEM_TYPE)
                    (
                       *(__global CL_INPUT_TYPE *)(b + p * N + (col+1)),
                       *(__global CL_INPUT_TYPE *)(b + (p+1) * N + (col+1))
                    );
        
        c_00 += aa1 * bb1, c_01 += aa1 * bb2;
        c_10 += aa2 * bb1, c_11 += aa2 * bb2;
    }

    c[row * N + col] = c_00.s0 + c_00.s1,      c[row * N + (col+1)] = c_01.s0 + c_01.s1;
    c[(row+1) * N + col] = c_10.s0 + c_10.s1,  c[(row+1) * N + (col+1)] = c_11.s0 + c_11.s1;
}



// TODO: Strange! I think it's right!
__kernel void mat_mult_vec2x2_continue(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    CL_ELEM_TYPE aa1, aa2,
                 bb1, bb2,
                 cc1 = 0, 
                 cc2 = 0;

    for (int p = 0; p < K; p+=2) {
        aa1 = *(
                   (__global CL_ELEM_TYPE *)(a + row * M + p)
               );
        aa2 = *(
                   (__global CL_ELEM_TYPE *)(a + (row+1) * M + p)
               );

        bb1 = *(
                   (__global CL_ELEM_TYPE *)(b + p * N + col)
               );
        bb2 = *(
                   (__global CL_ELEM_TYPE *)(b + (p+1) * N + col)
               );
        cc1 = (CL_ELEM_TYPE)
                  (aa1.s0*bb1.s0 + aa1.s1*bb2.s0,    aa1.s0*bb1.s1 + aa1.s1*bb2.s1);
        cc2 = (CL_ELEM_TYPE)
                  (aa2.s0*bb1.s0 + aa2.s1*bb2.s0,    aa2.s0*bb1.s1 + aa2.s1*bb2.s1);
    }
    //*(__global CL_ELEM_TYPE *)(c + row * N + col) = cc1.s0;         *(__global CL_ELEM_TYPE *)(c + row * N + (col+1)) = cc1.s1;
    //*(__global CL_ELEM_TYPE *)(c + (row+1) * N + col) = cc2.s0;     *(__global CL_ELEM_TYPE *)(c + (row+1) * N + (col+1)) = cc2.s0;

    //*(__global CL_ELEM_TYPE *)(c + (row+1) * N + col) = cc2;

    c[row * N + col] = cc1.s0;      c[row * N + (col+1)] = cc1.s1;
    c[(row+1)*N + col] = cc2.s0;    c[(row+1)*N + (col+1)] = cc2.s1;

}

