__kernel void matrixTranspose(const int heightA,
							  const int widthA,
							  __global const float *a,
							  __global float *a_T) {
	const int alpha = 4;
	const int colA = get_global_id(0) * alpha;
	const int rowA = get_global_id(1) * alpha;

	//if ((rowA < (heightA+alpha)) && (colA < (widthA+alpha))) {
	if ((rowA < heightA) && (colA < widthA)) {

		float4 vec1x4_1;
		vec1x4_1 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
	
		vec1x4_1.x = a[ (colA+0) * heightA + (rowA+0) ];
		vec1x4_1.y = a[ (colA+1) * heightA + (rowA+0) ];
		vec1x4_1.z = a[ (colA+2) * heightA + (rowA+0) ];
		vec1x4_1.w = a[ (colA+3) * heightA + (rowA+0) ];

		a_T[(rowA+0)*widthA + (colA+0)] = vec1x4_1.x;
		a_T[(rowA+0)*widthA + (colA+1)] = vec1x4_1.y;
		a_T[(rowA+0)*widthA + (colA+2)] = vec1x4_1.z;
		a_T[(rowA+0)*widthA + (colA+3)] = vec1x4_1.w;

		/**********************************************/

        float4 vec1x4_2;
        vec1x4_2.xyzw = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

        vec1x4_2.x = a[ (colA+0) * heightA + (rowA+1) ];
        vec1x4_2.y = a[ (colA+1) * heightA + (rowA+1) ];
        vec1x4_2.z = a[ (colA+2) * heightA + (rowA+1) ];
        vec1x4_2.w = a[ (colA+3) * heightA + (rowA+1) ];

        a_T[(rowA+1)*widthA + (colA+0)] = vec1x4_2.x;
        a_T[(rowA+1)*widthA + (colA+1)] = vec1x4_2.y;
        a_T[(rowA+1)*widthA + (colA+2)] = vec1x4_2.z;
        a_T[(rowA+1)*widthA + (colA+3)] = vec1x4_2.w;

        /**********************************************/

        float4 vec1x4_3;
        vec1x4_3.xyzw = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

        vec1x4_3.x = a[ (colA+0) * heightA + (rowA+2) ];
        vec1x4_3.y = a[ (colA+1) * heightA + (rowA+2) ];
        vec1x4_3.z = a[ (colA+2) * heightA + (rowA+2) ];
        vec1x4_3.w = a[ (colA+3) * heightA + (rowA+2) ];

        a_T[(rowA+2)*widthA + (colA+0)] = vec1x4_3.x;
        a_T[(rowA+2)*widthA + (colA+1)] = vec1x4_3.y;
        a_T[(rowA+2)*widthA + (colA+2)] = vec1x4_3.z;
        a_T[(rowA+2)*widthA + (colA+3)] = vec1x4_3.w;

        /**********************************************/
    
        float4 vec1x4_4;
        vec1x4_4.xyzw = (float4) (0.0f, 0.0f, 0.0f, 0.0f);

        vec1x4_4.x = a[ (colA+0) * heightA + (rowA+3) ];
        vec1x4_4.y = a[ (colA+1) * heightA + (rowA+3) ];
        vec1x4_4.z = a[ (colA+2) * heightA + (rowA+3) ];
        vec1x4_4.w = a[ (colA+3) * heightA + (rowA+3) ];

        a_T[(rowA+3)*widthA + (colA+0)] = vec1x4_4.x;
        a_T[(rowA+3)*widthA + (colA+1)] = vec1x4_4.y;
        a_T[(rowA+3)*widthA + (colA+2)] = vec1x4_4.z;
        a_T[(rowA+3)*widthA + (colA+3)] = vec1x4_4.w;
	}

}
