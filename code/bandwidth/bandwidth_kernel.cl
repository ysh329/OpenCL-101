#define ELEM_TYPE float
__kernel void global_bandwidth_float_v1(const int heightA, const int widthA, __global const ELEM_TYPE *a, __global ELEM_TYPE *b) {
    const int idx = get_global_id(0);

    b[idx] = a[idx];
}

__kernel void global_bandwidth_float_v1_2(const int heightA, const int widthA, __global const ELEM_TYPE *a, __global ELEM_TYPE *b) {
    const int colA = get_global_id(0);
    const int rowA = get_global_id(1);

    if ((colA < widthA) && (rowA < heightA) ) {
        b[rowA * widthA + colA] = a[rowA * widthA + colA];
    }

}

__kernel void global_bandwidth_float_v2(const int heightA, const int widthA, __global const ELEM_TYPE *a, __global ELEM_TYPE *b) {
    int alpha = 4;
    const int colA = get_global_id(0);
    const int rowA = get_global_id(1);
    //step = colA << 2;  
   // float4 value = *((__global float4 *)(src + step)) 
    // *((__global float4 *)(dst + step) = value;

}

__kernel void demo(const int heightA, const int widthA, __global const float *a, __global float *b) {

    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);

    const int local_id = get_local_id(0);

    const int global_id = get_global_id(0);
    const int global_id2 = get_global_id(1);

    b[global_id2 * widthA + global_id] = a[global_id2 * widthA + global_id]*2;

    if (global_id == 0) {
        b[global_id2 * widthA + global_id] = a[global_id2 * widthA + global_id]*2*31;
        printf("global_id: %4d, local_id: %4d, group_id: %4d, local_size: %4d\n", global_id, local_id, group_id, local_size);
    }
}
