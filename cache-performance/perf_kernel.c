

__kernel void p(__global float *a) {
    const int idx = get_global_id(0);

    float v = 1.3f;
    float w = 1.5f;
    float c = 0.0f;

    int i = 0;
    for (i = 0; i < 1000; i++) {
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 


        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 

        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 


        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 


        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 



        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 


        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 

        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 


        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 


        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c);
        c = mad(v, w, c); 

    }

    a[idx] = c;
}
