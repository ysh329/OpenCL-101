

__kernel void p(__global float *a) {
    const int idx = get_global_id(0);

    float v = 1.3f;
    float w = 1.5f;

    float c0 = 0.5f;
    float c1 = 0.7f;
    float c2 = 0.6f;
    float c3 = 0.77f;
    float c4 = 0.55f;
    float c5 = 0.44f;
    float c6 = 0.32f;
    float c7 = 0.23f;
    float c8 = 0.12f;
    float c9 = 0.43f;

    int i = 0;
    for (i = 0; i < 1000; i++) {
        c0 = mad(v, w, c0);
        c1 = mad(v, w, c1);
        c2 = mad(v, w, c2);
        c3 = mad(v, w, c3);
        c4 = mad(v, w, c4); 
        c5 = mad(v, w, c5);
        c6 = mad(v, w, c6);
        c7 = mad(v, w, c7);
        c8 = mad(v, w, c8);
        c9 = mad(v, w, c9); 

        c0 = mad(v, w, c0);
        c1 = mad(v, w, c1);
        c2 = mad(v, w, c2);
        c3 = mad(v, w, c3);
        c4 = mad(v, w, c4); 
        c5 = mad(v, w, c5);
        c6 = mad(v, w, c6);
        c7 = mad(v, w, c7);
        c8 = mad(v, w, c8);
        c9 = mad(v, w, c9); 

        c0 = mad(v, w, c0);
        c1 = mad(v, w, c1);
        c2 = mad(v, w, c2);
        c3 = mad(v, w, c3);
        c4 = mad(v, w, c4); 
        c5 = mad(v, w, c5);
        c6 = mad(v, w, c6);
        c7 = mad(v, w, c7);
        c8 = mad(v, w, c8);
        c9 = mad(v, w, c9); 

        c0 = mad(v, w, c0);
        c1 = mad(v, w, c1);
        c2 = mad(v, w, c2);
        c3 = mad(v, w, c3);
        c4 = mad(v, w, c4); 
        c5 = mad(v, w, c5);
        c6 = mad(v, w, c6);
        c7 = mad(v, w, c7);
        c8 = mad(v, w, c8);
        c9 = mad(v, w, c9); 

        c0 = mad(v, w, c0);
        c1 = mad(v, w, c1);
        c2 = mad(v, w, c2);
        c3 = mad(v, w, c3);
        c4 = mad(v, w, c4); 
        c5 = mad(v, w, c5);
        c6 = mad(v, w, c6);
        c7 = mad(v, w, c7);
        c8 = mad(v, w, c8);
        c9 = mad(v, w, c9); 

        c0 = mad(v, w, c0);
        c1 = mad(v, w, c1);
        c2 = mad(v, w, c2);
        c3 = mad(v, w, c3);
        c4 = mad(v, w, c4); 
        c5 = mad(v, w, c5);
        c6 = mad(v, w, c6);
        c7 = mad(v, w, c7);
        c8 = mad(v, w, c8);
        c9 = mad(v, w, c9); 

        c0 = mad(v, w, c0);
        c1 = mad(v, w, c1);
        c2 = mad(v, w, c2);
        c3 = mad(v, w, c3);
        c4 = mad(v, w, c4); 
        c5 = mad(v, w, c5);
        c6 = mad(v, w, c6);
        c7 = mad(v, w, c7);
        c8 = mad(v, w, c8);
        c9 = mad(v, w, c9); 

        c0 = mad(v, w, c0);
        c1 = mad(v, w, c1);
        c2 = mad(v, w, c2);
        c3 = mad(v, w, c3);
        c4 = mad(v, w, c4); 
        c5 = mad(v, w, c5);
        c6 = mad(v, w, c6);
        c7 = mad(v, w, c7);
        c8 = mad(v, w, c8);
        c9 = mad(v, w, c9); 

        c0 = mad(v, w, c0);
        c1 = mad(v, w, c1);
        c2 = mad(v, w, c2);
        c3 = mad(v, w, c3);
        c4 = mad(v, w, c4); 
        c5 = mad(v, w, c5);
        c6 = mad(v, w, c6);
        c7 = mad(v, w, c7);
        c8 = mad(v, w, c8);
        c9 = mad(v, w, c9); 

        c0 = mad(v, w, c0);
        c1 = mad(v, w, c1);
        c2 = mad(v, w, c2);
        c3 = mad(v, w, c3);
        c4 = mad(v, w, c4); 
        c5 = mad(v, w, c5);
        c6 = mad(v, w, c6);
        c7 = mad(v, w, c7);
        c8 = mad(v, w, c8);
        c9 = mad(v, w, c9); 

    }

    a[idx] = c0 + c1 + c2 + c3 + c4 +
             c5 + c6 + c7 + c8 + c9;
}
