__kernel void matrixTranspose(const int heightA,
							  const int widthA,
							  __global const float *a,
							  __global float *a_T) {
	int alpha = 8;
	const int colA = get_global_id(0) * alpha;
	const int rowA = get_global_id(1) * alpha;

	if (rowA < heightA && colA < widthA) {
		a_T[ (rowA + 0) * widthA + (colA + 0) ] = a[ (colA + 0) * heightA + (rowA + 0) ];
		a_T[ (rowA + 0) * widthA + (colA + 1) ] = a[ (colA + 1) * heightA + (rowA + 0) ];
		a_T[ (rowA + 0) * widthA + (colA + 2) ] = a[ (colA + 2) * heightA + (rowA + 0) ];
		a_T[ (rowA + 0) * widthA + (colA + 3) ] = a[ (colA + 3) * heightA + (rowA + 0) ];
		a_T[ (rowA + 0) * widthA + (colA + 4) ] = a[ (colA + 4) * heightA + (rowA + 0) ];
		a_T[ (rowA + 0) * widthA + (colA + 5) ] = a[ (colA + 5) * heightA + (rowA + 0) ];
		a_T[ (rowA + 0) * widthA + (colA + 6) ] = a[ (colA + 6) * heightA + (rowA + 0) ];
		a_T[ (rowA + 0) * widthA + (colA + 7) ] = a[ (colA + 7) * heightA + (rowA + 0) ];

		a_T[ (rowA + 1) * widthA + (colA + 0) ] = a[ (colA + 0) * heightA + (rowA + 1) ];
		a_T[ (rowA + 1) * widthA + (colA + 1) ] = a[ (colA + 1) * heightA + (rowA + 1) ];
		a_T[ (rowA + 1) * widthA + (colA + 2) ] = a[ (colA + 2) * heightA + (rowA + 1) ];
		a_T[ (rowA + 1) * widthA + (colA + 3) ] = a[ (colA + 3) * heightA + (rowA + 1) ];
		a_T[ (rowA + 1) * widthA + (colA + 4) ] = a[ (colA + 4) * heightA + (rowA + 1) ];
		a_T[ (rowA + 1) * widthA + (colA + 5) ] = a[ (colA + 5) * heightA + (rowA + 1) ];
		a_T[ (rowA + 1) * widthA + (colA + 6) ] = a[ (colA + 6) * heightA + (rowA + 1) ];
		a_T[ (rowA + 1) * widthA + (colA + 7) ] = a[ (colA + 7) * heightA + (rowA + 1) ];

		a_T[ (rowA + 2) * widthA + (colA + 0) ] = a[ (colA + 0) * heightA + (rowA + 2) ];
		a_T[ (rowA + 2) * widthA + (colA + 1) ] = a[ (colA + 1) * heightA + (rowA + 2) ];
		a_T[ (rowA + 2) * widthA + (colA + 2) ] = a[ (colA + 2) * heightA + (rowA + 2) ];
		a_T[ (rowA + 2) * widthA + (colA + 3) ] = a[ (colA + 3) * heightA + (rowA + 2) ];
		a_T[ (rowA + 2) * widthA + (colA + 4) ] = a[ (colA + 4) * heightA + (rowA + 2) ]; 
		a_T[ (rowA + 2) * widthA + (colA + 5) ] = a[ (colA + 5) * heightA + (rowA + 2) ]; 
		a_T[ (rowA + 2) * widthA + (colA + 6) ] = a[ (colA + 6) * heightA + (rowA + 2) ]; 
		a_T[ (rowA + 2) * widthA + (colA + 7) ] = a[ (colA + 7) * heightA + (rowA + 2) ]; 

		a_T[ (rowA + 3) * widthA + (colA + 0) ] = a[ (colA + 0) * heightA + (rowA + 3) ];
		a_T[ (rowA + 3) * widthA + (colA + 1) ] = a[ (colA + 1) * heightA + (rowA + 3) ];
		a_T[ (rowA + 3) * widthA + (colA + 2) ] = a[ (colA + 2) * heightA + (rowA + 3) ];
		a_T[ (rowA + 3) * widthA + (colA + 3) ] = a[ (colA + 3) * heightA + (rowA + 3) ];
		a_T[ (rowA + 3) * widthA + (colA + 4) ] = a[ (colA + 4) * heightA + (rowA + 3) ];
		a_T[ (rowA + 3) * widthA + (colA + 5) ] = a[ (colA + 5) * heightA + (rowA + 3) ];
		a_T[ (rowA + 3) * widthA + (colA + 6) ] = a[ (colA + 6) * heightA + (rowA + 3) ];
		a_T[ (rowA + 3) * widthA + (colA + 7) ] = a[ (colA + 7) * heightA + (rowA + 3) ];

		a_T[ (rowA + 4) * widthA + (colA + 0) ] = a[ (colA + 0) * heightA + (rowA + 4) ];
		a_T[ (rowA + 4) * widthA + (colA + 1) ] = a[ (colA + 1) * heightA + (rowA + 4) ];
		a_T[ (rowA + 4) * widthA + (colA + 2) ] = a[ (colA + 2) * heightA + (rowA + 4) ];
		a_T[ (rowA + 4) * widthA + (colA + 3) ] = a[ (colA + 3) * heightA + (rowA + 4) ];
		a_T[ (rowA + 4) * widthA + (colA + 4) ] = a[ (colA + 4) * heightA + (rowA + 4) ];
		a_T[ (rowA + 4) * widthA + (colA + 5) ] = a[ (colA + 5) * heightA + (rowA + 4) ];
		a_T[ (rowA + 4) * widthA + (colA + 6) ] = a[ (colA + 6) * heightA + (rowA + 4) ];
		a_T[ (rowA + 4) * widthA + (colA + 7) ] = a[ (colA + 7) * heightA + (rowA + 4) ];

        a_T[ (rowA + 5) * widthA + (colA + 0) ] = a[ (colA + 0) * heightA + (rowA + 5) ];
        a_T[ (rowA + 5) * widthA + (colA + 1) ] = a[ (colA + 1) * heightA + (rowA + 5) ];
        a_T[ (rowA + 5) * widthA + (colA + 2) ] = a[ (colA + 2) * heightA + (rowA + 5) ];
        a_T[ (rowA + 5) * widthA + (colA + 3) ] = a[ (colA + 3) * heightA + (rowA + 5) ];
        a_T[ (rowA + 5) * widthA + (colA + 4) ] = a[ (colA + 4) * heightA + (rowA + 5) ];
        a_T[ (rowA + 5) * widthA + (colA + 5) ] = a[ (colA + 5) * heightA + (rowA + 5) ];
        a_T[ (rowA + 5) * widthA + (colA + 6) ] = a[ (colA + 6) * heightA + (rowA + 5) ];
        a_T[ (rowA + 5) * widthA + (colA + 7) ] = a[ (colA + 7) * heightA + (rowA + 5) ];

        a_T[ (rowA + 6) * widthA + (colA + 0) ] = a[ (colA + 0) * heightA + (rowA + 6) ];
        a_T[ (rowA + 6) * widthA + (colA + 1) ] = a[ (colA + 1) * heightA + (rowA + 6) ];
        a_T[ (rowA + 6) * widthA + (colA + 2) ] = a[ (colA + 2) * heightA + (rowA + 6) ];
        a_T[ (rowA + 6) * widthA + (colA + 3) ] = a[ (colA + 3) * heightA + (rowA + 6) ];
        a_T[ (rowA + 6) * widthA + (colA + 4) ] = a[ (colA + 4) * heightA + (rowA + 6) ];
        a_T[ (rowA + 6) * widthA + (colA + 5) ] = a[ (colA + 5) * heightA + (rowA + 6) ];
        a_T[ (rowA + 6) * widthA + (colA + 6) ] = a[ (colA + 6) * heightA + (rowA + 6) ];
        a_T[ (rowA + 6) * widthA + (colA + 7) ] = a[ (colA + 7) * heightA + (rowA + 6) ];

        a_T[ (rowA + 7) * widthA + (colA + 0) ] = a[ (colA + 0) * heightA + (rowA + 7) ];
        a_T[ (rowA + 7) * widthA + (colA + 1) ] = a[ (colA + 1) * heightA + (rowA + 7) ];
        a_T[ (rowA + 7) * widthA + (colA + 2) ] = a[ (colA + 2) * heightA + (rowA + 7) ];
        a_T[ (rowA + 7) * widthA + (colA + 3) ] = a[ (colA + 3) * heightA + (rowA + 7) ];
        a_T[ (rowA + 7) * widthA + (colA + 4) ] = a[ (colA + 4) * heightA + (rowA + 7) ];
        a_T[ (rowA + 7) * widthA + (colA + 5) ] = a[ (colA + 5) * heightA + (rowA + 7) ];
        a_T[ (rowA + 7) * widthA + (colA + 6) ] = a[ (colA + 6) * heightA + (rowA + 7) ];
        a_T[ (rowA + 7) * widthA + (colA + 7) ] = a[ (colA + 7) * heightA + (rowA + 7) ];

	}
}
