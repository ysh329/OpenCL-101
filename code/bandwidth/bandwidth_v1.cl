__kernel void global_bandwith_v1_float(const int heightA,
							               const int widthA,
							               __global const ELEM_TYPE *a,
                             __global ELEM_TYPE *b) {
	const int colA = get_global_id(0);

	if (colA < widthA) {
		for (int rowA = 0; rowA < heightA; rowA++) {
			b[rowA * widthA + colA] = a[rowA * widthA + colA];
		}
	}
}
