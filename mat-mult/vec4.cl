#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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

