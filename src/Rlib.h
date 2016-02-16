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
#include<cuda_runtime.h>
#include<cusolverDn.h>

using namespace std;
/*A Matrix handler*/
struct Matrix{
  
  float **M;//Pointer to pointer, this is the access point to the data
  float *data; //The data itself, stored aligned in memory
  int n; //size of the matrix (only square ones for now)
  void init(int n, int m){
    if(n!=m) printf("WARNING, taking Matrix as nxn\n");
    this->n = n;
    /*C style memory handling*/
    M = (float **)malloc(sizeof(float *)*n);
    data = (float *)malloc(sizeof(float)*n*n);
    for(int i=0; i<n; i++) M[i] = &data[i*n];
  }
  
  void fill_with(float x){
    for(int i=0; i<n*n; i++) data[i] = x;
  }
  void print(){
    for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++) 
	printf("%4.2e ",M[i][j]);
      printf("\n");
    }
    printf("\n");
  }
  /*bracket operator overloading,
    now you can access Matrix a using a[i][j] with no additional cost*/
  float* operator [](const int &i){return M[i];}
};



/*cuSolver wrapper*/
struct cusolverHandle{
  //You need a handle to the cusolver instance
  cusolverDnHandle_t solver_handle;
  float *d_B; //Pointer to device matrix
  int *d_info, h_info; //Host and device variables to get information from cusolver
  int h_work_size;  //cusolver variable, related with the expected work load
  float *d_work;  //work load for cusolver
  /*This function uses cusolver to perform a cholesky decomposition 
    on D, storing the result on B*/
  int chol(Matrix D, Matrix B){
    int n = D.n; //size of the matrix
    /*Upload D to d_B pointer on GPU*/
    cudaMemcpy(d_B, D[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
    /*Perform the Cholesky decomposition*/
    cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
		     n, d_B, n, d_work, h_work_size, d_info);
    /*Retrieve the status information from cusolver, only in debug mode*/
    h_info = 0;
    #ifdef DEBUG
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(h_info !=0) printf("Cholesky CUBLAS Error!!! %d \n",h_info);
    #endif
    /*Retrive matrix from GPU, stroe in B*/
    cudaMemcpy(B[0], d_B, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    /*Fix upper part of B*/
    for(int i=0; i<n; i++)
      for(int j=i+1; j<n; j++)
	B[i][j] = 0;
    return h_info;
  }
  /*Starts the cholesky decomposition asynchronously on GPU */
  void chol_async(Matrix D){
    int n = D.n; //size of the matrix
    /*Upload D to d_B pointer on GPU*/
    cudaMemcpy(d_B, D[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
    /*Perform the Cholesky decomposition*/
    cusolverDnSpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER,
		     n, d_B, n, d_work, h_work_size, d_info);
  }
  int chol_retrieve(Matrix B){
    int n = B.n;
    /*Retrive matrix from GPU, stroe in B*/
    cudaMemcpy(B[0], d_B, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    /*Retrieve the status information from cusolver, only in debug mode*/
    h_info = 0;
    #ifdef DEBUG
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(h_info !=0) printf("Cholesky CUBLAS Error!!! %d \n",h_info);
    #endif
    /*Fix upper part of B*/
    for(int i=0; i<n; i++)
      for(int j=i+1; j<n; j++)
	B[i][j] = 0;
  }
};

/*Initialize all cuda variables and enviroment*/
void initCUDA(int n, cusolverHandle &cu){
  cudaSetDevice(1);
  /*Save space for one matrix in d_B (for cholesky results)*/
  cudaMalloc(&(cu.d_B), n*n*sizeof(float));
  /*One int to retrive information about cusolver*/
  cudaMalloc(&(cu.d_info), sizeof(int));
  /*Initialize cusolver enviroment*/
  cu.h_work_size = 0;
  cusolverDnCreate(&(cu.solver_handle));
  /*Compute a work size for the cholesky operation */
  cusolverDnSpotrf_bufferSize(cu.solver_handle, CUBLAS_FILL_MODE_UPPER, n, cu.d_B, n, &(cu.h_work_size));
  /*Reserve the optimal auxiliar space for cusolverDnSpotrf, only once!*/ 
  cudaMalloc(&(cu.d_work), cu.h_work_size*sizeof(float));
}

