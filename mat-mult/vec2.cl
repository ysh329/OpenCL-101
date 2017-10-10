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

__kernel void mat_mult_vec2x2(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1) << 1;

    CL_ELEM_TYPE aa1, aa2, bb, 
                 c_00 = 0, c_01 = 0;
                 c_10 = 0, c_11 = 0;

    for (int p = 0; p < K; p+=2) {
        aa1 = *((__global CL_ELEM_TYPE *)(a + row * M + p));
        aa2 = *((__global CL_ELEM_TYPE *)(a + (row+1) * M + p));

        bb = (CL_ELEM_TYPE)
                    (
                       *(__global CL_INPUT_TYPE *)(b + p * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+1) * N + col)
                    );
        
        c_00 += aa1 * bb;
        c_10 += aa2 * bb;
    }
    c[row * N + col] = c_00.s0 + c_00.s1;
    c[(row+1) * N + col] = c_10.s0 + c_10.s1;
}
