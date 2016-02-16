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

/* Global variables */
int n = 3*particles; /* Number of degrees of freedom */

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

/* Random number from a Gaussian distribution */
float Gaussian(float mean, float stddev){
  double u1, u2, s0 = 2;
  while(s0 >= 1){
    u1 = uniform(-1, 1);
    u2 = uniform(-1, 1);
    s0 = u1*u1 + u2*u2;
  }

  return mean + stddev*u1*sqrt(-2*logf(s0)/s0);
}

/* Kronecker delta function */
int delta(int i, int j){
  if(i != j)
    return 0;
  else
    return 1;
}

/*** Matrix-vector product ***/
int prod(Matrix a, float b[n], float c[n]){
  int i, j; /* Indices */
  float tmpsum;
  for(i = 0; i < n; i++){
    tmpsum = 0;
    for(j = 0; j < n; j++)
      tmpsum += a[i][j]*b[j];
    c[i] = tmpsum;
  }
  return 0;
}

/*** Cholesky decomposition ***/
int Cholesky(Matrix D, Matrix B){
  cu.chol(D, B);
  return 0;
}

/*** External forces ***/
int forces(float R[n], float F[n]){
  int i, j; /* Indices */
  float kwave = 2*PI/lambda; /* Wavenumber */
  float kx, ky;

  for(i = 0; i < n; i++)
    F[i] = 0;

  for(i = 0; i < particles; i++){
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

  return 0;
}


/*** Main program (Brownian dynamics with hydrodynamic interaction) ***/
int main(int argc, char * argv[]){
  initCUDA(n, cu);
  int i, j, k, l; /* Indices */
  int tstep = 0; /* Time step counter */
  float t; /* Time */
  float c1 = 0, c2 = 0; /* RPY diffusion tensor coefficients */

  float R[n]; /* Particle coordinates */
  float Rij[3]; /* Vector joining two particles */
  float distRij, R2; /* Distance between two particles and distance squared */
  float F[n]; /* External forces */
  float dW[n]; /* Wiener process */
  /*Matrices are stored aligned in memory*/
  Matrix K; /* Shear tensor */ 
  Matrix D; /* Rodne-Prager-Yamakawa diffusion tensor */
  Matrix B; /* D = B·B^T */
  float KR[n]; /* K·R */
  float DF[n]; /* D·F */
  float BdW[n]; /* B·dW */

  K.init(n,n); /*You need to init matrices*/
  D.init(n,n);
  B.init(n,n);

  /* Simulation parameters */

  /* The PRNG state must be seeded so that it is not everywhere zero. */
  s[0] = 12679825035178159220u;
  s[1] = 15438657923749336752u;

  /*** Initial conditions ***/
  for(i = 0; i < n; i++)
    R[i] = uniform(-boxlength/2.0, boxlength/2.0);

  /*** Dynamics ***/

  /* Shear tensor */
  K.fill_with(0); /* Clear shear tensor */

  for(i = 0; i < particles; i++){
    K[3*i + 2][3*i + 1] = shear;
  }

  /* Diffusion tensor (diagonal boxes remain unchanged during execution) */
  for(i = 0; i < particles; i++){
    for(k = 0; k < 3; k++){
      for(l = 0; l < 3; l++){
        if(k == l) D[3*i + k][3*i + l] = D0;
        else D[3*i + k][3*i + l] = 0;
      }
    }
  }
  
  /*** Integrate the stochastic differential equation ***/
  /*         dR = (K R + D F) dt + sqrt(2) B dW         */

  /*** Output header ***/
  // printf("# Brownian dynamics with hydrodynamic interactions\n#\n");
  //printf("# x \t\t y \t\t z #\n");
  //printf("#---\t\t---\t\t---#\n");

  for(t = t0; t <= tmax; t += dt){

    /* Calculate the Rodne-Prager-Yamakawa diffusion tensor */
    for(i = 0; i < particles; i++){
      for(j = 0; j < particles; j++){
        if(i > j){ /* The tensor is symmetrical */        
          for(k = 0; k < 3; k++)
            for(l = 0; l < 3; l++)
              D[3*i + k][3*j + l] = D[3*j + l][3*i + k];
        }
        else if(i < j){
          /* Calculate the vector from R_i to R_j */
          for(k = 0; k < 3; k++)
            Rij[k] = R[3*j + k] - R[3*i + k];

          /* Calculate the distance between R_i and R_j */
          R2 = 0;
          for(k = 0; k < 3; k++)
            R2 += Rij[k]*Rij[k];
          distRij = sqrt(R2);

          /* Calculate the factors c1 and c2 (taking into account whether particles overlap) */
          if(distRij >= 2*rh){
            c1 = 0.75*rh/distRij*(1 + 2*rh*rh/(3*R2));
            c2 = 0.75*rh/distRij*(1 - 2*rh*rh/R2);
          }
          else if(distRij < 2*rh){
              c1 = 1 - 9*distRij/(32*rh);
              c2 = 3*distRij/(32*rh);
          }
          /* Fill in the diffusion tensor components */
          for(k = 0; k < 3; k++){
            for(l = 0; l < 3; l++){
              D[3*i + k][3*j + l] = D0*c1*delta(k,l) + c2*Rij[k]*Rij[l]/R2;
            }
          }
        }
      }
    }

    /* Find B through Cholesky decomposition of D */
    //Cholesky(D, B);
    /*Query the GPU for a cholesky decomp. on D. Asynchronously */
    cu.chol_async(D);    
    /* Calculate the optical forces */
    forces(R, F);
    /* Calculate the Wiener displacement */
    for(i = 0; i < n; i++)
      dW[i] = Gaussian(0,1)*sqrt(dt);
    /* Euler scheme integration step */

    prod(K, R, KR); /* K·R */
    prod(D, F, DF); /* D·F */

    /*Wait until cholesky decomp. is finished and query store results in B*/
    cu.chol_retrieve(B);
    prod(B, dW, BdW); /* B·dW */

    /* dR = (K R + D F) dt + sqrt(2) B·dW */
    for(i = 0; i < n; i++)
      R[i] += (KR[i] + DF[i])*dt + sqrt(2)*BdW[i];

    /* Output results */

    if(tstep % sampling == 0){
      printf("#\n");
      for(i = 0; i < particles; i++)
        printf("%f\t%f\t%f\n", R[3*i], R[3*i + 1], R[3*i + 2]);
      printf("\n");
    }
    
    tstep++;
  }

  return 0;
}
