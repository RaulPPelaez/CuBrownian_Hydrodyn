#include"gpu.h"
#include"parameters.h"
#include<cuda_runtime.h>
#include<cuda.h>
#include <stdio.h>

#define TPB 64

__global__ void forceGPU(float *R, float *F){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if(i>particles) return;
  int j;
  float kwave = 2*PI/lambda; /* Wavenumber */
  float kx, ky;
  for(j = 0; j < 3; j++) F[3*i+j] = 0.0f;
  /* Optical force */
  if(i < gold){
    kx = kwave * R[3*i];
    ky = kwave * R[3*i + 1];
    F[3*i] = u0*alpha*cos(kx)*(sin(kx) - beta*sin(ky));
    F[3*i + 1] = u0*alpha*cos(ky)*(sin(ky) + beta*sin(kx));
  }

  /* Confining potential */
  for(j = 0; j < 3; j++){
    if(R[3*i + j] > boxlength/2.0)
      F[3*i + j] -= confpow*confstrength*pow(R[3*i + j] - boxlength/2.0, confpow - 1);
    if(R[3*i + j] < -boxlength/2.0)
      F[3*i + j] -= confpow*confstrength*pow(R[3*i + j] + boxlength/2.0, confpow - 1);
  }
}

__global__ void rotneGPU(float *D, float *R){
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  //int i = blockIdx.x;
  //int j = threadIdx.x;
  int j = id/particles;
  int i = id%particles;
  if(i>particles || j>particles || j==i) return;
  float rij[3];
  float r2 = 0.0f;
  float r;
  float c1, c2;
  for(int k = 0; k<3; k++){
    rij[k] = R[3*j + k] - R[3*i+k];
    r2 += rij[k]*rij[k];
  }
  r = sqrt(r2);
  if(r>=2*rh){
    c1 = 0.75*rh/r*(1.0f + 2.0f*rh*rh/(3.0f*r2));
    c2 = 0.75*rh/r*(1.0f - 2.0f*rh*rh/r2);
  }
  else{
    c1 = 1.0f - 9.0f*r/(32.0f*rh);
    c2 = 3.0f*r/(32.0f*rh);
  }

  for(int k = 0; k < 3; k++)
    for(int l = 0; l < 3; l++)
      D[3*i + k + n*(3*j + l)] = D0*c1*(k==l?1.0f:0.0f) + c2*rij[k]*rij[l]/r2;
}

__global__ void integrate_positions(float *R, float *DF, float *BdW){
  int ii = blockIdx.x*blockDim.x + threadIdx.x;
  //  printf("%d\n",ii);
  //  int i = blockIdx.x;
  //int j = threadIdx.x;
  //int ii = 3*i+j;
  if(ii>=n) return;
  R[ii] += DF[ii] + BdW[ii];
}

__global__ void fix_B(float *B){
  int ii = blockIdx.x*blockDim.x + threadIdx.x;
  int j = ii/particles;
  int i = ii%particles;
  if(i<=j) return;
  
  B[ii] = 0.0f;
}

void rodne_call(float *d_D, float *d_R, cudaStream_t stream){
  rotneGPU<<<(particles*particles)/TPB+1, TPB, 0 ,stream>>>(d_D, d_R);
  //rotneGPU<<<particles, particles>>>(d_D, d_R);
}
void force_call(float *d_R, float *d_F, cudaStream_t stream){
  forceGPU<<<particles/TPB + 1, TPB, 0, stream>>>(d_R, d_F);
}
void integrate_call( float *R, float *DF, float *BdW){
  integrate_positions<<<n/TPB+1,TPB>>>(R, DF, BdW);
}
void fix_B_call(float *B){
  fix_B<<<(n*n)/TPB+1,TPB>>>(B);
}
