#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void mat_mult_naive(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    CL_ELEM_TYPE res = 0;

    for (int p = 0; p < K; p++) {
        res += a[row * M + p] * b[p * N + col];
    }
    c[row * N + col] = res;
}

__kernel void mat_mult_vec2(const int M, const int N, const int K, __global const CL_INPUT_TYPE *a, __global const CL_INPUT_TYPE *b, __global CL_INPUT_TYPE *c) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    CL_ELEM_TYPE res = 0;

}






__kernel void global_bandwidth_vec1(const int heightA, const int widthA, __global const CL_INPUT_TYPE *a, __global CL_INPUT_TYPE *b) {
    const int idx = get_global_id(0);
    b[idx] = a[idx];
}

// input-variable-type must be non-vector format: such as int, float, double. ( TypeN is Wrong, error code: -14)
__kernel void global_bandwidth_vec2(const int heightA, const int widthA, __global const CL_INPUT_TYPE *a, __global CL_INPUT_TYPE *b) {
    const int idx = get_global_id(0);
    const int step = idx << 1;

    CL_ELEM_TYPE value = *((__global CL_ELEM_TYPE *)(a + step));
    *((__global CL_ELEM_TYPE *)(b + step)) = value;
}

__kernel void global_bandwidth_vec4(const int heightA, const int widthA, __global const CL_INPUT_TYPE *a, __global CL_INPUT_TYPE *b) {
    const int idx = get_global_id(0);
    const int step = idx << 2;

    CL_ELEM_TYPE value = *((__global CL_ELEM_TYPE *)(a + step));
    *((__global CL_ELEM_TYPE *)(b + step)) = value;
}

__kernel void global_bandwidth_vec8(const int heightA, const int widthA, __global const CL_INPUT_TYPE *a, __global CL_INPUT_TYPE *b) {
    const int idx = get_global_id(0);
    const int step = idx << 3;

    CL_ELEM_TYPE value = *((__global CL_ELEM_TYPE *)(a + step));
    *((__global CL_ELEM_TYPE *)(b + step)) = value;
}

__kernel void global_bandwidth_vec16(const int heightA, const int widthA, __global const CL_INPUT_TYPE *a, __global CL_INPUT_TYPE *b) {
    const int idx = get_global_id(0);
    const int step = idx << 4;

    CL_ELEM_TYPE value = *((__global CL_ELEM_TYPE *)(a + step));
    *((__global CL_ELEM_TYPE *)(b + step)) = value;
}
