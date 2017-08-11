__kernel void matrixMultiplication(const int M,
				   const int N,
				   const int K,
				   __global const float *a,
				   __global const float *b,
				   __global float *c) {
	const int rowC = get_global_id(0);
	const int colC = get_global_id(0);

    if (colC < N) {
    	float acc = 0.0;
        for (int k = 0; k < K; k++) {
            acc += a[k * M + rowC] * b[colC * K + k];
        }
        c[colC * M + rowC] = acc;
    }
}
