//Raul P. Pelaez 2016
//Auxiliar code for Marc's Brownian Hydrodynamics. 
//Contains a Matrix wrapper and functions to perform cholesky decomposition using CUDA
//
//Create a cusolverHandle object, cu
//Call initCUDA(n, cu) //n is 3*Nparticles, the size of the matrices
//Call cu.chol(Matrix D, Matrix B) to perform the cholesky decomposition on D
// and store it in B
#include<stdio.h>
#include<stdlib.h>
#include<algorithm>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<curand.h>
#include<cusolverDn.h>
#include"gpu.h"
#include<sys/time.h>
using namespace std;

/*A Matrix handler*/
class Matrix{
public:
  float **M;//Pointer to pointer, this is the access point to the data
  float *data; //The data itself, stored aligned in memory
  float *d_m; //device pointer
  bool GPUinit;
  int n, m; //size of the matrix
  Matrix(int n, int m){
    this->n = n;
    this->m = m;
    /*C style memory handling*/
    M = (float **)malloc(sizeof(float *)*n);
    cudaMallocHost((void **)&data, sizeof(float)*n*m);
    for(int i=0; i<n; i++) M[i] = &data[i*m];
    GPUinit = false;
  }
  
  void fill_with(float x){
    for(int i=0; i<n*m; i++) data[i] = x;
  }
  
  void upload(){
    if(!GPUinit){
      cudaMalloc(&d_m, n*m*sizeof(float));
      GPUinit = true;
    }
    cudaMemcpy(d_m, data, n*m*sizeof(float), cudaMemcpyHostToDevice);
  }
  void download(float *dst = NULL){
    if(dst==NULL)
      cudaMemcpy(data, d_m, n*m*sizeof(float), cudaMemcpyDeviceToHost);
    else{
      cudaMemcpy(dst, d_m, n*m*sizeof(float), cudaMemcpyDeviceToHost);
    }
  }
  void downloadAsync(cudaStream_t stream, float *dst = NULL){
    if(dst==NULL)
      cudaMemcpyAsync(data, d_m, m*n*sizeof(float), cudaMemcpyDeviceToHost, stream);
    else{
      cudaMemcpyAsync(dst, d_m, m*n*sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
  }
  
  void print(){
    for(int i=0; i<n; i++) {
      for(int j=0; j<m; j++) 
	printf("%9.2e ",M[i][j]);
      printf("\n");
    }
    printf("\n");
  }
  /*bracket operator overloading,
    now you can access Matrix a using a[i][j] with no additional cost*/
  float* operator [](const int &i){return M[i];}
};


class Vector{
public:
  float *data; //The data itself, stored aligned in memory
  float *d_m; //device pointer
  bool GPUinit;
  int n; //size of the matrix
  Vector(int n){
    this->n = n;
    /*C style memory handling*/
    cudaMallocHost((void **)&data, sizeof(float)*n);
    GPUinit = false;
  }
  
  void fill_with(float x){
    std::fill(data, data+n, x);
  }
  
  void upload(){
    if(!GPUinit){
      cudaMalloc(&d_m, n*sizeof(float));
      GPUinit = true;
    }
    cudaMemcpy(d_m, data, n*sizeof(float), cudaMemcpyHostToDevice);
  }
  void download(float *dst = NULL){
    if(dst==NULL)
      cudaMemcpy(data, d_m, n*sizeof(float), cudaMemcpyDeviceToHost);
    else{
      cudaMemcpy(dst, d_m, n*sizeof(float), cudaMemcpyDeviceToHost);
    }
  }
  void downloadAsync( cudaStream_t stream, float *dst = NULL){
    if(dst==NULL)
      cudaMemcpyAsync(data, d_m, n*sizeof(float), cudaMemcpyDeviceToHost, stream);
    else{
      cudaMemcpyAsync(dst, d_m, n*sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
  }

  
  void print(){
    for(int i=0; i<n; i++)
      printf("%9.2e ",data[i]);
    printf("\n");
   
  }
  float& operator [](const int &i){return data[i];}
  operator float *() const{return data;}
};


/*cuSolver wrapper*/
struct cusolverHandle{
  //You need a handle to the cusolver instance
  cusolverDnHandle_t solver_handle;
  cublasStatus_t status;
  cublasHandle_t cublas_handle;
  curandGenerator_t rng;
  int *d_info, h_info; //Host and device variables to get information from cusolver
  int h_work_size;  //cusolver variable, related with the expected work load
  float *d_work;  //work load for cusolver

  void chol_async(Matrix D, Matrix B){
    static bool first= true;
    int n = D.n; //size of the matrix
    if(first){
      cudaMalloc(&d_info, sizeof(int));
      h_work_size = 0;
      cusolverDnCreate(&solver_handle);
      cusolverDnSpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, n, B.d_m, n, &h_work_size);
      cudaMalloc(&d_work, h_work_size*sizeof(float));
      first = false;
    }
    cudaMemcpy(B.d_m, D.d_m, n*n*sizeof(float), cudaMemcpyDeviceToDevice);
    cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
		     n, B.d_m, n, d_work, h_work_size, d_info);
  }

  void chol_finish(Matrix B){
    fix_B_call(B.d_m);
  }
  void prod(Matrix A, Vector B, Vector C, float alpha, float beta){
    int n = A.n;
    cublasSgemv(cublas_handle, CUBLAS_OP_N, n, n, &alpha, A.d_m, n, B.d_m, 1, &beta, C.d_m, 1);
  }

  void gaussian(Vector A){
    curandGenerateNormal(rng, A.d_m, A.n, 0 ,1);
  }
  void integrate(Vector R, Vector DF, Vector BdW){
    int n = R.n;
    float alpha = 1.0;
    cublasSaxpy(cublas_handle, n, &alpha, DF.d_m, 1, R.d_m, 1);
    cublasSaxpy(cublas_handle, n, &alpha, BdW.d_m, 1, R.d_m, 1);
    //integrate_call(R.d_m, DF.d_m, BdW.d_m);
  }
};

/*Initialize all cuda variables and enviroment*/
void initCUDA(int n, cusolverHandle &cu){
 
  cu.status = cublasCreate(&(cu.cublas_handle));
  curandCreateGenerator(&(cu.rng), CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed((cu.rng), 1234ULL);
}



class Timer{
public:
  Timer(){}
  void tic(){
    gettimeofday(&start, NULL);
  }
  float toc(){
    gettimeofday(&end, NULL);
    //return std::chrono::duration<float>(t2 - t).count();
    return ((end.tv_sec  - start.tv_sec) * 1000000u + 
	    end.tv_usec - start.tv_usec) / 1.e6;
  }
private:
  //std::chrono::time_point<std::chrono::system_clock> t;
  struct timeval start, end;
};





/*

  int chol_retrieve(Matrix B){
    int n = B.n;
  
    cudaMemcpy(B[0], d_B, n*n*sizeof(float), cudaMemcpyDeviceToHost);
  
    h_info = 0;
    #ifdef DEBUG
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(h_info !=0) printf("Cholesky CUBLAS Error!!! %d \n",h_info);
    #endif
  
    for(int i=0; i<n; i++)
      for(int j=i+1; j<n; j++)
	B[i][j] = 0;
    return h_info;
  }
  int chol_retrieveAsync(Matrix B){
    int n = B.n;
  
    cudaMemcpyAsync(B[0], d_B, n*n*sizeof(float), cudaMemcpyDeviceToHost);
  return 0;
  }

*/
