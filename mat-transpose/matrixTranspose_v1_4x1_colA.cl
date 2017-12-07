__kernel void matrixTranspose(const int heightA,
							  const int widthA,
							  __global const float *a,
							  __global float *a_T) {
	const int colA = get_global_id(0) * 4;
	for (int rowA = 0; rowA < widthA; rowA++) {
		a_T[ (rowA + 0) * widthA + colA] = a[colA * heightA + (rowA + 0) ];
		a_T[ (rowA + 1) * widthA + colA] = a[colA * heightA + (rowA + 1) ];
		a_T[ (rowA + 2) * widthA + colA] = a[colA * heightA + (rowA + 2) ];
		a_T[ (rowA + 3) * widthA + colA] = a[colA * heightA + (rowA + 3) ];
	}
}
