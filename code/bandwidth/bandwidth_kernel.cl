#define ELEM_TYPE float

__kernel void global_bandwidth_int_v1(const int heightA, const int widthA, __global const int *a, __global int *b) {
    const int idx = get_global_id(0);
    b[idx] = a[idx];
}

__kernel void global_bandwidth_float_v1(const int heightA, const int widthA, __global const float *a, __global float *b) {
    const int idx = get_global_id(0);
    b[idx] = a[idx];
}

__kernel void global_bandwidth_float_v2(const int heightA, const int widthA, __global const float *a, __global float *b) {
    const int idx = get_global_id(0);
    const int step = idx << 1;

    float2 value = *((__global float2 *)(a + step));
    *((__global float2 *)(b + step)) = value;
}

__kernel void global_bandwidth_float_v4(const int heightA, const int widthA, __global const float *a, __global float *b) {
    const int idx = get_global_id(0);
    const int step = idx << 2;
    
    float4 value = *((__global float4 *)(a + step)); 
    //((__global float4 *)(b + step))[0] = value;
    *((__global float4 *)(b + step)) = value;
}

