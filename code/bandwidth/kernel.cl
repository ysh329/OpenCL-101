__kernel void global_bandwidth_vec1(const int heightA, const int widthA, __global const CL_ELEM_TYPE *a, __global CL_ELEM_TYPE *b) {
    const int idx = get_global_id(0);
    b[idx] = a[idx];
}

__kernel void global_bandwidth_vec2(const int heightA, const int widthA, __global const CL_ELEM_TYPE *a, __global CL_ELEM_TYPE *b) {
    const int idx = get_global_id(0);
    const int step = idx << 1;

    CL_ELEM_TYPE value = *((__global CL_ELEM_TYPE *)(a + step));
    *((__global CL_ELEM_TYPE *)(b + step)) = value;
}

__kernel void global_bandwidth_vec4(const int heightA, const int widthA, __global const CL_ELEM_TYPE *a, __global CL_ELEM_TYPE *b) {
    const int idx = get_global_id(0);
    const int step = idx << 2;

    CL_ELEM_TYPE value = *((__global CL_ELEM_TYPE *)(a + step));
    *((__global CL_ELEM_TYPE *)(b + step)) = value;
}

__kernel void global_bandwidth_vec8(const int heightA, const int widthA, __global const CL_ELEM_TYPE *a, __global CL_ELEM_TYPE *b) {
    const int idx = get_global_id(0);
    const int step = idx << 3;

    CL_ELEM_TYPE value = *((__global CL_ELEM_TYPE *)(a + step));
    *((__global CL_ELEM_TYPE *)(b + step)) = value;
}

__kernel void global_bandwidth_vec16(const int heightA, const int widthA, __global const CL_ELEM_TYPE *a, __global CL_ELEM_TYPE *b) {
    const int idx = get_global_id(0);
    const int step = idx << 4;

    CL_ELEM_TYPE value = *((__global CL_ELEM_TYPE *)(a + step));
    *((__global CL_ELEM_TYPE *)(b + step)) = value;
}
