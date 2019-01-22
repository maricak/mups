#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <iostream>

#include <functional>
#include <algorithm>

#define NUM_OF_GPU_THREADS 1024
#define ACCURACY 1

double *jacobi_seq(int m, int n)
{
  double d, r, t, *b, *x, *xnew;
  int i, it;

  b = (double *)malloc(n * sizeof(double));
  x = (double *)malloc(n * sizeof(double));
  xnew = (double *)malloc(n * sizeof(double));

  printf("\n");
  printf("JACOBI:\n");
  printf("  C++ sequential version\n");
  printf("  Jacobi iteration to solve A*x=b.\n");
  printf("\n");
  printf("  Number of variables  N = %d\n", n);
  printf("  Number of iterations M = %d\n", m);

  printf("\n");
  printf("  IT     l2(dX)    l2(resid)\n");
  printf("\n");

  b[n - 1] = (double)(n + 1);

  /*
    Initialize the solution estimate to 0.
    Exact solution is (1,2,3,...,N).
  */
  for (i = 0; i < n; i++)
  {
    x[i] = 0.0;
  }

  /* Iterate M times. */
  for (it = 0; it < m; it++)
  {
    /* Jacobi update. */
    for (i = 0; i < n; i++)
    {
      xnew[i] = b[i];
      if (0 < i)
      {
        xnew[i] = xnew[i] + x[i - 1];
      }
      if (i < n - 1)
      {
        xnew[i] = xnew[i] + x[i + 1];
      }
      xnew[i] = xnew[i] / 2.0;
    }
    /* Difference. */
    d = 0.0;
    for (i = 0; i < n; i++)
    {
      d = d + pow(x[i] - xnew[i], 2);
    }
    /* Overwrite old solution. */
    for (i = 0; i < n; i++)
    {
      x[i] = xnew[i];
    }
    /* Residual. */
    r = 0.0;
    for (i = 0; i < n; i++)
    {
      t = b[i] - 2.0 * x[i];
      if (0 < i)
      {
        t = t + x[i - 1];
      }
      if (i < n - 1)
      {
        t = t + x[i + 1];
      }
      r = r + t * t;
    }

    if (it < 10 || m - 10 < it)
    {
      printf("  %8d  %14.6g  %14.6g\n", it, sqrt(d), sqrt(r));
    }
    if (it == 9)
    {
      printf("  Omitting intermediate results.\n");
    }
  }

  /* Write part of final estimate. */
  printf("\n");
  printf("  Part of final solution estimate:\n");
  printf("\n");
  for (i = 0; i < 10; i++)
  {
    printf("  %8d  %14.6g\n", i, x[i]);
  }
  printf("...\n");
  for (i = n - 11; i < n; i++)
  {
    printf("  %8d  %14.6g\n", i, x[i]);
  }

  /* Free memory. */
  free(b);
  //free(x);
  free(xnew);

  return x;
}

__global__ void KernelJacobi(double* cuda_x, double* cuda_xnew, double* cuda_d_array, int n, int iteration) {

  __shared__ double shared_x_array[NUM_OF_GPU_THREADS];

  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  int localId = threadIdx.x;
  int leftId = localId- 1; // local leftId
  int rightId = localId + 1; // local rightId
  double leftValue, rightValue;
  double xNewValue;
  // double bValue; // is equal to (n + 1) only for the last element in cuda_x, otherwise 0 
  // ^ ---> can be avoided as to not use an additional variable in the register file
  
  //printf("myId = %d, localId= %d, leftId = %d, rightId = %d\n", myId, threadIdx.x, leftId, rightId);
  __syncthreads();

  if (myId < n) {  
    // retrieve element for your id and place into shared memory
    shared_x_array[threadIdx.x] = cuda_x[myId];
    __syncthreads();
    
    //printf("shared_x_array[%d] = %d\n", threadIdx.x, shared_x_array[threadIdx.x]);

    // retrieve element from shared memory if leftId/rightId aren't out of bounds 
    // ---> otherwise, grab left/right value if it exists

    if (leftId == -1) {  // local leftId is out of bounds -- get value if it exists (isn't 0-th element of cuda_x)
      if (myId == 0) {
        leftValue = 0.0;
      }
      else {
        leftValue = cuda_x[myId - 1]; // use global id
      }
    }
    else {
      leftValue = shared_x_array[leftId]; // get leftValue from shared array
    }

    if (rightId == NUM_OF_GPU_THREADS) { 
      // local rightId is out of bounds -- get value if it exists (isn't n-1-st element of cuda_x)  
      if (myId == n - 1) {
        rightValue = 0.0;
      }
      else {
        rightValue = cuda_x[myId + 1];   // use global id
      }
    }
    else {
      rightValue = shared_x_array[rightId]; // get rightValue from shared array
    }

    xNewValue = (myId == n - 1) ? n + 1 : 0;

    xNewValue += leftValue;
    xNewValue += rightValue;
    xNewValue /= 2.0;

    // Store new value
    cuda_xnew[myId] = xNewValue;
    // printf("globalId=%d, localId=%d, leftId=%d, rightId=%d, leftValue=%f, rightValue=%f, newValue=%f\n", myId, localId, leftId, rightId, leftValue, rightValue, cuda_xnew[myId]);
    // if it's an iteration when reduction will be performed, send D off to global memory
    if (iteration == 1) {
      cuda_d_array[myId] = pow(shared_x_array[threadIdx.x] - xNewValue, 2);
    }
  
  }

  __syncthreads(); // all threads (even those that don't work) synchronise here
  
}

__global__ void KernelOverwrite(double* cuda_x, double* cuda_xnew, int n) {
    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    if (myId < n) {
      // printf("myId=%d, oldcuda_x=%f, newcuda_x=%f\n", myId, cuda_x[myId], cuda_xnew[myId]);
      cuda_x[myId] = cuda_xnew[myId];
    }
}

__global__ void KernelCalculateR(double* cuda_x, double* cuda_r_array, int n) {

  __shared__ double shared_x_array[NUM_OF_GPU_THREADS];

  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  int localId = threadIdx.x;
  int leftId = localId- 1; // local leftId
  int rightId = localId + 1; // local rightId
  double leftValue, rightValue;
  double t;

  __syncthreads();

  
  if (myId < n) {

    shared_x_array[threadIdx.x] = cuda_x[myId];
    __syncthreads();

    if (leftId == -1) {  // local leftId is out of bounds -- get value if it exists (isn't 0-th element of cuda_x)
      if (myId == 0) {
        leftValue = 0.0;
      }
      else {
        leftValue = cuda_x[myId - 1]; // use global id
      }
    }
    else {
      leftValue = shared_x_array[leftId]; // get leftValue from shared array
    }

    if (rightId == NUM_OF_GPU_THREADS) { 
      // local rightId is out of bounds -- get value if it exists (isn't n-1-st element of cuda_x)  
      if (myId == n - 1) {
        rightValue = 0.0;
      }
      else {
        rightValue = cuda_x[myId + 1];   // use global id
      }
    }
    else {
      rightValue = shared_x_array[rightId]; // get rightValue from shared array
    }

    t = (myId == n - 1) ? n + 1 : 0;
    t -= 2.0 * shared_x_array[localId];
    t += leftValue;
    t += rightValue;
    t *= t;
    cuda_r_array[myId] = t;

  }

  __syncthreads();
}

// each block reduces and then places into cuda_d_result or cuda_r_result 
__global__ void KernelReduction(double* cuda_d_array, double* cuda_r_array, int n,
        double* cuda_d_result_array, double* cuda_r_result_array) {

  __shared__ double shared_d_array[NUM_OF_GPU_THREADS];
  __shared__ double shared_r_array[NUM_OF_GPU_THREADS];

	int myId = blockIdx.x * blockDim.x + threadIdx.x;
        int localId = threadIdx.x;

	__syncthreads();


  if (myId < n) {
  
    shared_d_array[localId] = cuda_d_array[myId];
    shared_r_array[localId] = cuda_r_array[myId];
  }

   __syncthreads();
  
  //if (myId < n) {   
    if (blockIdx.x < gridDim.x - 1) {
     // printf("entered if");
      // 1024 elements to reduce
      for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (localId < s && myId < n) {
          shared_d_array[localId] = shared_d_array[localId] + shared_d_array[localId + s];
          shared_r_array[localId] = shared_r_array[localId] + shared_r_array[localId + s];
        }

        __syncthreads();
      }
    }
    else {
      // find number of elements remaining in array -- then reduce those
	
      int remainingElements = n - blockIdx.x * NUM_OF_GPU_THREADS; 
     // printf("entered else, elems remaining: %d\n", remainingElements);
      int b = 1;
      while (b < remainingElements)     // nearest larger power of 2
      {
        b = b << 1;
      }

      //printf("remaining=%d, b = %d, globalId=%d, localId=%d\n", remainingElements, b, localId, myId);
      for (unsigned int s = b / 2; s > 0; s >>= 1) {
        //printf("s=%d", s);
        if ((localId < s) && (localId + s < remainingElements) && (myId < n)) {
	// printf("globalId: %d, localId: %d, s: %d, shared_d[%d] = %f, shared_d[%d] = %f\n", myId, localId, s, localId, shared_d_array[localId], localId + s, shared_d_array[localId + s]);
         //  printf("Id=%d niz[%d]=niz[%d]+niz[%d] => %d + %d\n", id, id, id, id + s, cudaDeltaArray[id], cudaDeltaArray[id + s]);
          shared_d_array[localId] = shared_d_array[localId] + shared_d_array[localId + s];
          shared_r_array[localId] = shared_r_array[localId] + shared_r_array[localId + s];
        }
        __syncthreads();
      }

    }
   
    // only element with local 0 id places result into resulting arrays
    if (localId == 0) {
	//printf("shared_d_array[%d] = %f\n", localId, shared_d_array[localId]);
      cuda_d_result_array[blockIdx.x] = shared_d_array[0];
      cuda_r_result_array[blockIdx.x] = shared_r_array[0];
    }
 // }
  
  __syncthreads();

}

int main(int argc, char *argv[])
{

  int m, n, i, it;

  float timePar, timeSeq;
  double *xPar, *xSeq;

  cudaEvent_t start = cudaEvent_t();
  cudaEvent_t stop = cudaEvent_t(); 

  cudaEventCreate( &start );
  cudaEventCreate( &stop ); 

  if (argc == 3)
  {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
  }
  else
  {
    m = 5000;
    n = 50000;
  }

  // Use standard jacobi interface
  cudaEventRecord( start, 0 );
  xSeq = jacobi_seq(m, n);
  cudaEventRecord( stop, 0 ); 

  cudaEventSynchronize( stop ); 
  cudaEventElapsedTime( &timeSeq, start, stop ); 

  timeSeq /= 1000;

  printf("\n");
  printf("JACOBI_CUDA:\n");
  printf("  C/CUDA version - %d threads per block\n", NUM_OF_GPU_THREADS);
  printf("  Jacobi iteration to solve A*x=b.\n");
  printf("\n");
  printf("  Number of variables  N = %d\n", n);
  printf("  Number of iterations M = %d\n", m);

  printf("\n");
  printf("  IT     l2(dX)    l2(resid)\n");
  printf("\n");

  // ====== Parallel
  
  // Copy to GPU.

  double d, r;

  double *cuda_x;
  double *cuda_xnew;
  double* cuda_d_array;
  double* cuda_r_array;

  xPar = (double*) calloc (n, sizeof(double));
  cudaMalloc(&cuda_x, n * sizeof(double));
  cudaMemcpy(cuda_x, xPar, n * sizeof(double), cudaMemcpyHostToDevice);


  cudaMalloc(&cuda_xnew, n * sizeof(double));
  cudaMalloc(&cuda_d_array, n * sizeof(double));
  cudaMalloc(&cuda_r_array, n * sizeof(double));  

  dim3 DimGrid((int)((n + NUM_OF_GPU_THREADS - 1) / NUM_OF_GPU_THREADS)); // upper() thread blocks
  dim3 DimBlock(NUM_OF_GPU_THREADS); // 1024 threads per block 

  double* d_result_array = (double*) malloc (DimGrid.x * sizeof(double));
  double* r_result_array = (double*) malloc (DimGrid.x * sizeof(double));
  double* cuda_d_result_array;
  double* cuda_r_result_array;

  cudaMalloc(&cuda_d_result_array, DimGrid.x * sizeof(double)); // 1 element for each block
  cudaMalloc(&cuda_r_result_array, DimGrid.x * sizeof(double)); // 1 element for each block

  printf("Dimgrid: %d, DimBlock: %d\n", DimGrid.x, DimBlock.x);
  // Use parallel jacobi interface
  cudaEventRecord( start, 0 );

  for (it = 0; it < m; it++) {
    // call kernel for processing

    int shouldCalculateD = (it < 10 || m - 10 < it) ? 1 : 0;
    
    // calculate xnew on GPU (and calc. d if needed)
    KernelJacobi<<< DimGrid, DimBlock  >>>(cuda_x, cuda_xnew, cuda_d_array, n, shouldCalculateD);  
    cudaDeviceSynchronize();
    KernelOverwrite<<< DimGrid, DimBlock >>>(cuda_x, cuda_xnew, n); // place xnew into x on GPU
    cudaDeviceSynchronize();
    
    if (it < 10 || m - 10 < it) {
      

      KernelCalculateR<<< DimGrid, DimBlock >>> (cuda_x, cuda_r_array, n);
      cudaDeviceSynchronize();
      KernelReduction <<< DimGrid, DimBlock >>> (cuda_d_array, cuda_r_array, n, cuda_d_result_array, cuda_r_result_array);
      cudaDeviceSynchronize();

	    cudaMemcpy(d_result_array, cuda_d_result_array, DimGrid.x * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(r_result_array, cuda_r_result_array, DimGrid.x * sizeof(double), cudaMemcpyDeviceToHost);

      // calculate final reduction
      d = 0;
      r = 0;
      for (int i = 0; i < DimGrid.x; i++) {
	//printf("d_local=%f, r_local=%f\n", d_result_array[i], r_result_array[i]);
        d += d_result_array[i];
        r += r_result_array[i];
      }

      /*
      d = 1;
      r = 1;
      */

      printf("  %8d  %14.6g  %14.6g\n", it, sqrt(d), sqrt(r));
    }
    if (it == 9)
    {
      printf("  Omitting intermediate results.\n");
    }
  }

  cudaEventRecord( stop, 0 ); 

  cudaEventSynchronize( stop ); 
  cudaEventElapsedTime( &timePar, start, stop ); 

  timePar /= 1000;

  // Copy result.
  cudaMemcpy(xPar, cuda_x, n * sizeof(double), cudaMemcpyDeviceToHost);

  /* Write part of final estimate. */
  printf("\n");
  printf("  Part of final solution estimate:\n");
  printf("\n");
  for (i = 0; i < 10; i++)
  {
    printf("  %8d  %14.6g\n", i, xPar[i]);
  }
  printf("...\n");
  for (i = n - 11; i < n; i++)
  {
    printf("  %8d  %14.6g\n", i, xPar[i]);
  }


  // Comparison
  std::function<bool(double, double)> comparator = [](double left, double right) {
    // Lambda function to compare 2 doubles with ACCURACY
    return fabs(left - right) < ACCURACY;
  };

  std::vector<double> vectorPar(xPar, xPar + n), vectorSeq(xSeq, xSeq + n);

  std::cerr << "********************DZ3Z2**********************" << std::endl;
  std::cerr << "Elapsed time - SEQ: " << timeSeq << "." << std::endl;
  std::cerr << "Elapsed time - PAR(" << NUM_OF_GPU_THREADS << " threads/block): " << timePar << "." << std::endl;
  std::cerr << (std::equal(vectorPar.begin(), vectorPar.end(), vectorSeq.begin(), comparator) ? "TEST PASSED" : "TEST FAILED") << std::endl;
  std::cerr << "***********************************************" << std::endl;


  free(xSeq);
  free(xPar);

  // Cuda Free calls.
  cudaFree(cuda_x);
  cudaFree(cuda_xnew);
  cudaFree(cuda_d_array);
  cudaFree(cuda_r_array);
  cudaFree(cuda_d_result_array);
  cudaFree(cuda_r_result_array);

  return 0;
}

