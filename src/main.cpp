/************************************************************************
 *           Brownian motion with hydrodynamic interactions             *
 ************************************************************************/
/* Marc Melendez and Raul P. Pelaez 2016
 *
 *
 *
 */

/*** Standard libraries ***/
#include <stdio.h>
#include "Rlib.h"
#include "parameters.h"
#include "gpu.h"


class CuBrow{
  cuHandle cu; //This is a CUBLAS/CUSOLVER/CURAND wrapper, see Rlib.h
  Xorshift128plus rng;
  Timer tim;
  int tstep; /* Time step counter */
  float t; /* Time */
  Vector R; /* Particle coordinates */
  Vector F; /* External forces */
  Vector dW; /* Wiener process */
  //Matrix K; /* Shear tensor */ 
  Matrix D; /* Rodne-Prager-Yamakawa diffusion tensor */
  Matrix B; /* D = B·B^T */
  //Vector KR(n); /* K·R */
  Vector DF; /* D·F */
  Vector BdW; /* B·dW */
  void init(); /*Initialize all*/
public:
  CuBrow(int argc, char *argv[]);
  ~CuBrow();  /*Free memory and reset GPU*/
  void run(); /*Run simulation*/
  
};

CuBrow::CuBrow(int argc, char *argv[]){
  if(argc>1)
    cudaSetDevice(atoi(argv[1]));
  else     cudaSetDevice(0);
  initCUDA(n,cu);
  tim.tic();
  init();
  //  printf("Initializing time: %.3e\n", tim.toc());
}
CuBrow::~CuBrow(){
  R.freeGPU(); R.freeCPU();
  F.freeGPU();
  dW.freeGPU();
  DF.freeGPU();
  BdW.freeGPU();
  D.freeGPU();
  B.freeGPU();
  cudaDeviceReset();
}


void CuBrow::init(){
  /*Initialize vectors and matrices*/
  /*R is the only variable that we will be downloading during execution
    So it needsto be on pinned memory, thats the second argument*/
  R   = Vector(n, true); 
  F   = Vector(n); 
  dW  = Vector(n); 
  DF  = Vector(n); 
  BdW = Vector(n); 
  D = Matrix(n,n);
  B = Matrix(n,n);
  //KR = Vector(n);
  //K = Matrix(n,n);

  int i, j, k, l; /* Indices */
  /*** Initial conditions ***/
  for(i = 0; i < n; i++)
    R[i] = rng.uniform(-boxlength/2.0, boxlength/2.0);

  /* Shear tensor */
  //K.fill_with(0); /* Clear shear tensor */
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
  /*Upload all information to the GPU*/
  //  KR.upload();
  //  K.upload();

  /*We only need R in the CPU!*/
  R.upload();
  DF.upload();  DF.freeCPU();
  BdW.upload(); BdW.freeCPU();
  D.upload();   D.freeCPU();
  F.upload();   F.freeCPU();
  dW.upload();  dW.freeCPU();
  B.upload();   B.freeCPU();
}


void CuBrow::run(){ 
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaStream_t stream2;
  cudaStreamCreate(&stream2);
  
  tim.tic();
  tstep = 0;
  /*Main loop*/
  for(t = t0; t <= tmax; t += dt){
    /*Compute forces*/
    force_call(R.d_m, F.d_m, stream);//Execute on stream
    /*Fill dW with noise*/
    cu.gaussian(dW);
    /*Perform rodne-prage on D*/
    rodne_call(D.d_m, R.d_m, stream2); //Execute on stream2

    //cu.prod(K, R, KR); /* K·R, is always zero with shear = 0*/
    
    /*Wait for rodne-prage*/
    cudaStreamSynchronize(stream2); 
    /*Perform cholesky decomp. on D, store in B*/
    cu.chol_async(D, B);
    /*DF = D·F*/
    cu.prod(D, F, DF, dt, 0.0); /* D·F·dt */
    /*Fill upper part of B with 0*/
    cu.chol_finish(B);
    /* BdW = B·dW·sqrt(2)·sqrt(dt) */
    cu.prod(B, dW, BdW, sqrt(2)*sqrt(dt), 0.0); 
    /*Wait for all streams to finish*/
    cudaDeviceSynchronize();
    /*Integrate the positions*/
    /* dR = (K R + D F) dt + sqrt(2) B·dW  (now K is zero)*/
    cu.integrate(R, DF, BdW);

    /* Output results */
    if(tstep % sampling == 0){
      R.download();
      printf("#\n");
      for(int i = 0; i < particles; i++)
        printf("%f\t%f\t%f\n", R[3*i], R[3*i + 1], R[3*i + 2]);
      printf("\n");
    }
    
    tstep++;
  }
    
  //  printf("Execution time: %.3e\n", tim.toc());
}
/*** Main program (Brownian dynamics with hydrodynamic interaction) ***/
int main(int argc, char * argv[]){
  CuBrow app(argc, argv);
  app.run();
  return 0;
}
