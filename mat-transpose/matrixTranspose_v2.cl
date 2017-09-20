__kernel void matrixTranspose(const int heightA,
							  const int widthA,
							  __global const float *a,
							  __global float *a_T) {
	const int rowA = get_global_id(0);
	const int colA = get_global_id(1);

	if (rowA < heightA && colA < widthA) {
		a_T[rowA * widthA + colA] = a[colA * heightA + rowA];
	}
}
