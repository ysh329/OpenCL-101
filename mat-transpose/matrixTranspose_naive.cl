__kernel void matrixTranspose(const int heightA,
                              const int widthA,
                              __global const float *a,
                              __global float *a_T) {

    const int colA = get_global_id(0);

    for (int rowA = 0; rowA < heightA; rowA++) {
        a_T[colA * heightA + rowA] = a[rowA * widthA + colA];
    }
    
}
