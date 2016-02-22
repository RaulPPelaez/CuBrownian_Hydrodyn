/************************************************************************
 *           Brownian motion with hydrodynamic interactions             *
 ************************************************************************/

/*** Standard libraries ***/
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include"Rlib.h"


/*** Simulation parameters ***/
#include "parameters.h"
#include"gpu.h"

/*** Auxiliary functions ***/
cusolverHandle cu;

/* Pseudorandom number generation */
uint64_t s[2]; /* PRNG state */

/* 64-bit (pseudo)random integer */
uint64_t xorshift128plus(void){
  uint64_t x = s[0];
  uint64_t const y = s[1];
  s[0] = y;
  x ^= x << 23; // a
  x ^= x >> 17; // b
  x ^= y ^ (y >> 26); // c
  s[1] = x;
  return x + y;
}

/* Random number from a uniform distribution */
float uniform(float min, float max){
  return min + (xorshift128plus()/((float) RANDOM_MAX))*(max - min);
}
/*** Main program (Brownian dynamics with hydrodynamic interaction) ***/
int main(int argc, char * argv[]){
  if(argc>1)
    cudaSetDevice(atoi(argv[1]));
  else     cudaSetDevice(0);
  Timer tim;
  tim.tic();
  initCUDA(n, cu);
  int i, j, k, l; /* Indices */
  int tstep = 0; /* Time step counter */
  float t; /* Time */
  float c1 = 0, c2 = 0; /* RPY diffusion tensor coefficients */

  Vector R(n); /* Particle coordinates */
  float Rij[3]; /* Vector joining two particles */
  float distRij, R2; /* Distance between two particles and distance squared */
  Vector F(n); /* External forces */
  Vector dW(n); /* Wiener process */
  /*Matrices are stored aligned in memory*/
  //  Matrix K(n,n); /* Shear tensor */ 
  Matrix D(n,n); /* Rodne-Prager-Yamakawa diffusion tensor */
  Matrix B(n,n); /* D = B·B^T */
  //  Vector KR(n); /* K·R */
  Vector DF(n); /* D·F */
  Vector BdW(n); /* B·dW */

  /* The PRNG state must be seeded so that it is not everywhere zero. */
  s[0] = 12679825035178159220u;
  s[1] = 15438657923749336752u;

  /*** Initial conditions ***/
  for(i = 0; i < n; i++)
    R[i] = uniform(-boxlength/2.0, boxlength/2.0);

  /*** Dynamics ***/

  /* Shear tensor */
  //  K.fill_with(0); /* Clear shear tensor */
  //for(i = 0; i < particles; i++)
  // K[3*i + 2][3*i + 1] = shear;
  
  /* Diffusion tensor (diagonal boxes remain unchanged during execution) */
  for(i = 0; i < particles; i++){
    for(k = 0; k < 3; k++){
      for(l = 0; l < 3; l++){
        if(k == l) D[3*i + k][3*i + l] = D0;
        else D[3*i + k][3*i + l] = 0;
      }
    }
  }
  
  //  KR.upload();
  //  K.upload();
  DF.upload();
  BdW.upload();
  D.upload();
  R.upload();
  F.upload();
  dW.upload();
  B.upload();
  //  printf("Initializing time: %.3e\n", tim.toc());
  //tim.tic();
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaStream_t stream2;
  cudaStreamCreate(&stream2);

  for(t = t0; t <= tmax; t += dt){
    force_call(R.d_m, F.d_m, stream);
    cu.gaussian(dW);
    rodne_call(D.d_m, R.d_m, stream2);
    //cu.prod(K, R, KR); /* K·R, is always zero with shear = 0*/
    cudaDeviceSynchronize();
    cu.chol_async(D, B);
    cu.prod(D, F, DF, dt, 0.0); /* D·F·dt */
    cu.chol_finish(B);
    cu.prod(B, dW, BdW, sqrt(2)*sqrt(dt), 0.0); /* B·dW·sqrt(2)·sqrt(dt) */
    cudaDeviceSynchronize();
    /* dR = (K R + D F) dt + sqrt(2) B·dW */
    cu.integrate(R, DF, BdW);
    /* Output results */
      
    if(tstep % sampling == 0){
      R.download();
      printf("#\n");
      for(i = 0; i < particles; i++)
        printf("%f\t%f\t%f\n", R[3*i], R[3*i + 1], R[3*i + 2]);
      printf("\n");
    }
    
    tstep++;
  }
    
  //  printf("Execution time: %.3e\n", tim.toc());
  cudaDeviceReset();
  return 0;
}
