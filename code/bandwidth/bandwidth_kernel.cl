__kernel void global_bandwidth_float_v1(const int heightA, const int widthA, __global const float *a, __global float *b) {
    const int idx = get_global_id(0);

    b[idx] = a[idx];
}

/*
__kernel void global_bandwidth_float_v2(const int heightA, const int widthA, __global const float2 *a, __global float2 *b) {
    const int idx = get_global_id(0);
    const int step = idx << 2;

   float2 value = *((__global float2 *)(a + step));
   *((__global float2 *)(b + step)) = value;
}

__kernel void global_bandwidth_float_v4(const int heightA, const int widthA, __global const float4 *a, __global float4 *b) {
    const int idx = get_global_id(0);
    const int step = idx << 4;

    float4 value = *((__global float4 *)(a + step)) 
    *((__global float4 *)(b + step)) = value;
}*/
