__kernel void matrixTranspose(const int heightA,
							  const int widthA,
							  __global const float *a,
							  __global float *a_T) {
	int alpha = 4;
	const int rowA = get_global_id(0) * alpha;
	printf("%d ", rowA);
	if (rowA * alpha < heightA) {
		for (int colA = 0; colA < widthA; colA++) {
			a_T[ (rowA + 0) * widthA + colA] = a[colA * heightA + (rowA + 0) ];
			a_T[ (rowA + 1) * widthA + colA] = a[colA * heightA + (rowA + 1) ];
			a_T[ (rowA + 2) * widthA + colA] = a[colA * heightA + (rowA + 2) ];
			a_T[ (rowA + 3) * widthA + colA] = a[colA * heightA + (rowA + 3) ];
		}
	}
	else
		printf("global id error: %d\n", rowA);
}
