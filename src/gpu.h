#ifndef GPU_H
#define GPU_H

void rodne_call(float *d_D, float *d_R, cudaStream_t stream);
void force_call(float *d_R, float *d_F, cudaStream_t stream);
void integrate_call(float *R, float *DF, float *BdW);
void fix_B_call(float *B);
#endif
