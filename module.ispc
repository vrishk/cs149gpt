// Copied from 
inline float twoDimRead(float* tensor, int x, int y, const int sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}
inline void twoDimWrite(float* tensor, int x, int y, const int sizeX, float val) {
    tensor[x * (sizeX) + y] = val;
}
inline float fourDimRead(float* tensor, int x, int y, int z, int b, 
        const int sizeX, const int sizeY, const int sizeZ) {    
    return tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b];
}
inline void fourDimWrite(float* tensor, int x, int y, int z, int b, 
        const int sizeX, const int sizeY, const int sizeZ, float val) {
    tensor[x * (sizeX * sizeY * sizeZ) + y * (sizeY * sizeZ) + z * sizeZ + b] = val;
}



// performs matrix multiplication, setting Z where Z = XY^T. X and Y must be
// of the same dimension. Operation done on 4D matrices of size (B, H, N, D).
export void matrixMult( float* uniform X, float* uniform Y, float* uniform Z,
                    int uniform B, int uniform H, int uniform N, int uniform d,
                    int uniform b, int uniform h) {
    // QK^t mul for a given B and Hx
    //for  (i = 0 ... N) {
    for (int i = 0; i < N; i++) {
        foreach (k = 0 ... N) {
            float val = 0.f;
            for (int j = 0; j < d; j++) {
                float a_val = fourDimRead(X, b, h, i, j, H, N, d);
                float b_val = fourDimRead(Y, b, h, k, j, H, N, d);
                val += a_val * b_val;
            }
            twoDimWrite(Z, i, k, N, val);
        }
    }
}

// Performs in-place softmax and row-based normalization on T.
export void softmaxNorm(float* uniform T, int uniform N) {
    // Softmax
    // foreach (i = 0 ... N) {
    for (int i = 0; i < N; i++) {   
        float val = 0.f;
        float sample, normed;
        float min_val = 0.f;
        for (int j = 0; j < N; j++) {
        val += exp(twoDimRead(T, i, j, N) - min_val);
        }
        foreach (j = 0 ... N) {
        normed = exp(twoDimRead(T, i, j, N) - min_val) / val;
        twoDimWrite(T, i, j, N, normed);
        }
    }
}

// performs PV calculation for output in the attention layer, storing it in O.
export void pvCalc(float* uniform P, float* uniform V, float* uniform O,
                    int uniform B, int uniform H, int uniform N, int uniform d,
                    int uniform b, int uniform h) {
    // foreach (i = 0 ... N) {
    for (int i = 0; i < N; i++) {
        // for (int k = 0; k < d; k++) {
        foreach (k = 0 ... d) {
            float val = 0.f;
            for (int j = 0; j < N; j++) {
                float q = twoDimRead(P, i, j, N);
                float v = fourDimRead(V, b, h, j, k, H, N, d);
                val += q * v;
        }
        fourDimWrite(O, b, h, i, k, H, N, d, val);
        }
    }
}
