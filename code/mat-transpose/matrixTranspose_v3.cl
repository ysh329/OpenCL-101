__kernel void matrixTranspose(const int heightA,
							  const int widthA,
							  __global const float4 *a,
							  __global float4 *a_T) {
	const int alpha = 4;
	const int colA = get_global_id(0);
	const int rowA = get_global_id(1);

	if ((rowA*alpha < heightA) && (colA*alpha < widthA)) {
		a_T[(rowA+0) * widthA/alpha + colA].x = a[colA * heightA/alpha + (rowA+0)].x;
		a_T[(rowA+1) * widthA/alpha + colA].x = a[colA * heightA/alpha + (rowA+0)].y;
		a_T[(rowA+2) * widthA/alpha + colA].x = a[colA * heightA/alpha + (rowA+0)].z;
		a_T[(rowA+3) * widthA/alpha + colA].x = a[colA * heightA/alpha + (rowA+0)].w;

        a_T[(rowA+0) * widthA/alpha + colA].y = a[colA * heightA/alpha + (rowA+1)].x;
        a_T[(rowA+1) * widthA/alpha + colA].y = a[colA * heightA/alpha + (rowA+1)].y;
        a_T[(rowA+2) * widthA/alpha + colA].y = a[colA * heightA/alpha + (rowA+1)].z;
        a_T[(rowA+3) * widthA/alpha + colA].y = a[colA * heightA/alpha + (rowA+1)].w;

        a_T[(rowA+0) * widthA/alpha + colA].z = a[colA * heightA/alpha + (rowA+2)].x;
        a_T[(rowA+1) * widthA/alpha + colA].z = a[colA * heightA/alpha + (rowA+2)].y;
        a_T[(rowA+2) * widthA/alpha + colA].z = a[colA * heightA/alpha + (rowA+2)].z;
        a_T[(rowA+3) * widthA/alpha + colA].z = a[colA * heightA/alpha + (rowA+2)].w;

        a_T[(rowA+0) * widthA/alpha + colA].w = a[colA * heightA/alpha + (rowA+3)].x;
        a_T[(rowA+1) * widthA/alpha + colA].w = a[colA * heightA/alpha + (rowA+3)].y;
        a_T[(rowA+2) * widthA/alpha + colA].w = a[colA * heightA/alpha + (rowA+3)].z;
        a_T[(rowA+3) * widthA/alpha + colA].w = a[colA * heightA/alpha + (rowA+3)].w;
	}
}
