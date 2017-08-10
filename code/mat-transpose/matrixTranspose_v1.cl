__kernel void matrixTranspose(const int heightA,
							  const int widthA,
							  __global const float *a,
							  __global float *a_T) {
	const int rowA = get_global_id(0);

	if (rowA < heightA) {
		for (int colA = 0; colA < widthA; colA++) {
			a_T[rowA * widthA + colA] = a[colA * heightA + rowA];
		}
	}
}
