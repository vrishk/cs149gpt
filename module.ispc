// ------------------------ Warmup code copied from module.cpp ------------------------
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

// ------------------------ Part 1 Code ------------------------

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

// ------------------------ Part 2 Code ------------------------

// performs blocked matrix multiplication, setting Z where Z = XY^T. X and Y must be
// of the same dimension. Operation done on 4D matrices of size (B, H, N, D).
export void blockedMatrixMult( float* uniform X, float* uniform Y, float* uniform Z,
                    int uniform B, int uniform H, int uniform N, int uniform d,
                    int uniform b, int uniform h, const int uniform BLOCK) {
    float val;
    // for (int ii = 0; ii < N; ii += BLOCK) {
    foreach (ii_b = 0 ... (N + BLOCK - 1) / BLOCK) {
        int ii = ii_b * BLOCK;
        uint BLOCK_I = min(ii + BLOCK, N);
        for (int kk = 0; kk < N; kk += BLOCK) {
        // foreach (kk_b = 0 ... (N + BLOCK - 1) / BLOCK) {
            // int kk = kk_b * BLOCK;
            uint BLOCK_K = min(kk + BLOCK, N);
            for (int jj = 0; jj < d; jj += BLOCK) {
                uint BLOCK_J = min(jj + BLOCK, d);
                for(int i = ii; i < BLOCK_I; i++){
                // foreach (i = ii ... BLOCK_I) {
                    for(int k = kk; k < BLOCK_K; k++){
                        val = twoDimRead(Z, i, k, N);
                        for(int j = jj; j < BLOCK_J; j++){
                            val += fourDimRead(X, b, h, i, j, H, N, d) * 
                                        fourDimRead(Y, b, h, k, j, H, N, d);
                        }
                        twoDimWrite(Z, i, k, N, val);
                    }
                }
            }
        }
    }
}

// performs blocked PV calculation for output in the attention layer, storing it in O.
export void blockedPvCalc(float* uniform P, float* uniform V, float* uniform O,
                    int uniform B, int uniform H, int uniform N, int uniform d,
                    int uniform b, int uniform h, const int uniform BLOCK) {
    float val;
    // for (int ii = 0; ii < N; ii += BLOCK) {
    foreach (ii_b = 0 ... (N + BLOCK - 1) / BLOCK) {
        int ii = ii_b * BLOCK;
        uint BLOCK_I = min(ii + BLOCK, N);
        for (int kk = 0; kk < d; kk += BLOCK) {
        // foreach (kk_b = 0 ... (N + BLOCK - 1) / BLOCK) {
            // int kk = kk_b * BLOCK;
            uint BLOCK_K = min(kk + BLOCK, d);
            for (int jj = 0; jj < N; jj += BLOCK) {
                uint BLOCK_J = min(jj + BLOCK, N);
                for(int i = ii; i < BLOCK_I; i++){
                    for(int k = kk; k < BLOCK_K; k++){
                        val = fourDimRead(O, b, h, i, k, H, N, d);
                        for(int j = jj; j < BLOCK_J; j++){
                            val += twoDimRead(P, i, j, N) * fourDimRead(V, b, h, j, k, H, N, d);
                        }
                        fourDimWrite(O, b, h, i, k, H, N, d, val);
                    }
                }
            }
        }
    }
}

// ------------------------ Part 3 Code

export void fusedMatrixMult( float* uniform X, float* uniform Y, float* uniform Z,
                    int uniform B, int uniform H, int uniform N, int uniform d,
                    int uniform b, int uniform h, const int uniform BLOCK,
                    const int uniform i) {
    float val;
    // foreach (kk_b = 0 ... (d + BLOCK - 1) / BLOCK) {
        // int kk = kk_b * BLOCK;
    for (int kk = 0; kk < N; kk += BLOCK) {
        uint BLOCK_K = min(kk + BLOCK, N);
        for (int jj = 0; jj < d; jj += BLOCK) {
            uint BLOCK_J = min(jj + BLOCK, d);
            for(int k = kk; k < BLOCK_K; k++){
                val = Z[k];
                for(int j = jj; j < BLOCK_J; j++){
                    val += fourDimRead(X, b, h, i, j, H, N, d) * fourDimRead(Y, b, h, k, j, H, N, d);
                }
                Z[k] = val;
            }
        }
    }
}

export void fusedSoftmaxNorm(float* uniform T, int uniform N, int uniform i) {
    float val = 0.f;
    float sample, normed;
    float min_val = 0.f;
    for (int j = 0; j < N; j++) {
        val += exp(T[j] - min_val);
    }
    for (int j = 0; j < N; j++) {
        T[j] = exp(T[j] - min_val) / val;
    }
}

export void fusedPvCalc(float* uniform P, float* uniform V, float* uniform O,
                    int uniform B, int uniform H, int uniform N, int uniform d,
                    int uniform b, int uniform h, const int uniform BLOCK,
                    int uniform i) {
          float val = 0.f;
          for (int kk = 0; kk < d; kk += BLOCK) {
            uint BLOCK_K = min(kk + BLOCK, d);
            for (int jj = 0; jj < N; jj += BLOCK) {
              uint BLOCK_J = min(jj + BLOCK, N);
              for(int k = kk; k < BLOCK_K; k++){
                val = fourDimRead(O, b, h, i, k, H, N, d);
                for(int j = jj; j < BLOCK_J; j++){
                  val += P[j] * fourDimRead(V, b, h, j, k, H, N, d);
                }
                fourDimWrite(O, b, h, i, k, H, N, d, val);
              }
            }
          }  
}