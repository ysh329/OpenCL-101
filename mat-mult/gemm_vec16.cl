#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void mat_mult_vec16(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    CL_ELEM_TYPE res = 0;
    CL_ELEM_TYPE aa, bb;

    for (int p = 0; p < K; p += 16) {

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
                       *(__global CL_INPUT_TYPE *)(b + (p+7) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+8) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+9) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+10) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+11) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+12) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+13) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+14) * N + col),
                       *(__global CL_INPUT_TYPE *)(b + (p+15) * N + col)
                    );
        res += aa * bb;
    }
    c[row * N + col] = res.s0 + res.s1 + res.s2 + res.s3 + res.s4 + res.s5 + res.s6 + res.s7 + res.s8 + res.s9 +
                       res.sa + res.sb + res.sc + res.sd + res.se + res.sf;
}

