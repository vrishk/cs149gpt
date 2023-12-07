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


export void part1(float* uniform Q, float* uniform K, float* uniform V,
                       float* uniform QK_t, float* uniform O, int uniform B, 
                       int uniform H, int uniform N, int uniform d) {
    float val = 0.f;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // QK^t mul with softmax norm!
            // for (int i = 0; i < N; i++) {
            foreach (i = 0 ... N) {
                float total = 0.f;
                float val;
                // foreach (k = 0 ... N) {
                for (int k = 0; k < N; k++) { // does multiplication while holding onto 
                    val = 0.f;
                    for (int j = 0; j < d; j++) {
                        float a_val = fourDimRead(Q, b, h, i, j, H, N, d);
                        float b_val = fourDimRead(K, b, h, k, j, H, N, d);
                        val += a_val * b_val;
                    }
                    // take exp and store val for softmax
                    val = exp(val);
                    twoDimWrite(QK_t, i, k, N, val);
                    total += val; 
                }
                // finished QK_t calculation for row i; now normalize
                for (int k = 0; k < N; k++) {
                  val = twoDimRead(QK_t, i, k, N);
                  twoDimWrite(QK_t, i, k, N, val / total);
                }
            }
            // output multiplication
            for (int i = 0; i < N; i++) {
                foreach (k = 0 ... d) {
                    val = 0.f;
                    for (int j = 0; j < N; j++) {
                        float q = twoDimRead(QK_t, i, j, N);
                        float v = fourDimRead(V, b, h, j, k, H, N, d);
                        val += q * v;
                }
                fourDimWrite(O, b, h, i, k, H, N, d, val);
                }
            }
        }
    }
        
}

// ------------------------ Part 2 Code ------------------------

export void part2(float* uniform Q, float* uniform K, float* uniform V,float* uniform QK_t, 
                    float* uniform O, int uniform B, int uniform H, int uniform N, 
                    int uniform d, const int uniform BLOCK) {
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      float val;
      // blocked matrix mult
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
                          val = twoDimRead(QK_t, i, k, N);
                          for(int j = jj; j < BLOCK_J; j++){
                              val += fourDimRead(Q, b, h, i, j, H, N, d) * 
                                          fourDimRead(K, b, h, k, j, H, N, d);
                          }
                          twoDimWrite(QK_t, i, k, N, val);
                      }
                  }
              }
          }
      }
      // softmax norm
      for (int i = 0; i < N; i++) {   
        val = 0.f;
        float sample, normed;
        float min_val = 0.f;
        for (int j = 0; j < N; j++) {
          val += exp(twoDimRead(QK_t, i, j, N) - min_val);
        }
        foreach (j = 0 ... N) {
          normed = exp(twoDimRead(QK_t, i, j, N) - min_val) / val;
          twoDimWrite(QK_t, i, j, N, normed);
        }
      }
      val = 0.f;
      // output mult
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
                            val += twoDimRead(QK_t, i, j, N) * fourDimRead(V, b, h, j, k, H, N, d);
                        }
                        fourDimWrite(O, b, h, i, k, H, N, d, val);
                    }
                }
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
    foreach (kk_b = 0 ... (N + BLOCK - 1) / BLOCK) {
        int kk = kk_b * BLOCK;
    // for (int kk = 0; kk < N; kk += BLOCK) {
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
    // for (int j = 0; j < N; j++) {
    foreach (j = 0 ... N) {
        T[j] = exp(T[j] - min_val) / val;
    }
}

export void fusedPvCalc(float* uniform P, float* uniform V, float* uniform O,
                    int uniform B, int uniform H, int uniform N, int uniform d,
                    int uniform b, int uniform h, const int uniform BLOCK,
                    int uniform i) {
    float val = 0.f;
    foreach(kk_b = 0 ... (d + BLOCK - 1) / BLOCK) {
        int kk = kk_b * BLOCK;
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

// --- part 4 code --- // 
export void part4(float* uniform O, float* uniform Q, float* uniform K, float* uniform V, 
        float* uniform Sij, float* uniform Pij, float* uniform Kj, float* uniform Vj, 
        float* uniform Qi, float* uniform Oi, float* uniform l, float* uniform PV, 
        float* uniform li, float* uniform lij, float* uniform lnew, int uniform Bc, 
        int uniform Br, int uniform B, int uniform H, int uniform N, int uniform d,
        int uniform b, int uniform h) {

  int step = b * (H * N * d) + h * (N * d);
  foreach(i = 0 ... N) {
      l[i] = 0.f;
  }

  for(int jj = 0; jj < N; jj += Bc) {
  // foreach (jj_b = 0 ... (N + Bc - 1) / Bc) {
  //   int jj = jj_b * Bc;
    
    uint BLOCK_J = min(Bc, N - jj);
    for (int j = 0; j < BLOCK_J; j++) {
      // for (int k = 0; k < d; k++) {
      foreach (k = 0 ... d) {
        Kj[j * d + k] = K[step + (jj + j) * d + k];
        Vj[j * d + k] = V[step + (jj + j) * d + k];
      }
    }
    for (int ii = 0; ii < N; ii += Br) {
      uint BLOCK_I = min(Br, N - ii);

      for (int i = 0; i < BLOCK_I; i++) {
        li[i] = l[ii + i];
        foreach (k = 0 ... d) {
          Qi[i * d + k] = Q[step + (ii + i) * d + k];
          Oi[i * d + k] = O[step + (ii + i) * d + k];
        }
      }

      // QiKi^T
      for (int i = 0; i < BLOCK_I; i++) {
        for (int j = 0; j < BLOCK_J; j++) {
          float val = 0.f;
          for (int k = 0; k < d; k++) {
            val += Qi[i * d + k] * Kj[j * d + k];
          }
          Sij[i * BLOCK_J + j] = val;
        }
      }

      // QiKi^T
      for (int i = 0; i < BLOCK_I; i++) {
        for (int j = 0; j < BLOCK_J; j++) {
          Pij[i * BLOCK_J + j] = exp(Sij[i * BLOCK_J + j]);
        }
      }
    
      // Rowsum
      for (int i = 0; i < BLOCK_I; i++) {
        float val = 0.f;
        for (int j = 0; j < BLOCK_J; j++) {
          val += Pij[i * BLOCK_J + j];
        }
        lij[i] = val;
      }

      for (int i = 0; i < BLOCK_I; i++) {
        lnew[i] = li[i] + lij[i];
      }

      for (int i = 0; i < BLOCK_I; i++) {
        foreach (k = 0 ... d) {
          float val = 0.f;
          for (int j = 0; j < BLOCK_J; j++) {
            val += Pij[i * BLOCK_J + j] * Vj[j * d + k];
          }
          Oi[i * d + k] = (
            li[i] * Oi[i * d + k] + val
          ) / lnew[i];
        }
      }

      for (int i = 0; i < BLOCK_I; i++) {
        foreach (k = 0 ... d) {
          O[step + (ii + i) * d + k] = Oi[i * d + k];
        }
        l[ii + i] = lnew[i];
      }
    }
  }
}