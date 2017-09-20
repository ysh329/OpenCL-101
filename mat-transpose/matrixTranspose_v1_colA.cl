__kernel void matrixTranspose(const int heightA,
							  const int widthA,
							  __global const float *a,
							  __global float *a_T) {
	const int colA = get_global_id(0);

	if (colA < widthA) {
		for (int rowA = 0; rowA < heightA; rowA++) {
			a_T[rowA * widthA + colA] = a[colA * heightA + rowA];
		}
	}
}
