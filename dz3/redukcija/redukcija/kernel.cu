
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


#define NUM_BLOCKS 800
#define NUM_THREADS 1024

__global__ void localReductionKernel(int* cudaDeltaArray) {
	__shared__ int sharedDeltaArray[NUM_THREADS];

	unsigned int id = threadIdx.x;

	sharedDeltaArray[id] = 1;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (id < s) {
			sharedDeltaArray[id] = sharedDeltaArray[id] + sharedDeltaArray[id + s];
		}

		__syncthreads();
	}

	if (id == 0) {
		cudaDeltaArray[blockIdx.x] = sharedDeltaArray[0];
	}
}

int nextPowerOf2(int a){
	int b = 1;
	while (b < a)
	{
		b = b << 1;
	}
	return b;
}

__global__ void globalReductionKernel(int* cudaDeltaArray) {

	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	int b = 1;
	while (b < gridDim.x)
	{
		b = b << 1;
	}

	//printf("b=%d", b);

	for (unsigned int s = b / 2; s > 0; s >>= 1) {
		//printf("s=%d", s);
		if ((id < s) && (id + s <gridDim.x)) {
			printf("Id=%d niz[%d]=niz[%d]+niz[%d] => %d + %d\n", id, id, id, id + s, cudaDeltaArray[id], cudaDeltaArray[id + s]);
			cudaDeltaArray[id] = cudaDeltaArray[id] + cudaDeltaArray[id + s];
		}
		__syncthreads();
	}
}

int main() {

	int* deltaArray = new int[NUM_BLOCKS];
	int* cudaDeltaArray;

	cudaMalloc(&cudaDeltaArray, NUM_BLOCKS * sizeof(int));
	cudaMemcpy(cudaDeltaArray, deltaArray, NUM_BLOCKS * sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimGrid(NUM_BLOCKS);
	dim3 dimBlock(NUM_THREADS);

	localReductionKernel <<< dimGrid, dimBlock >>> (cudaDeltaArray);
	globalReductionKernel <<< dimGrid, dimBlock >>> (cudaDeltaArray);

	int delta;

	cudaMemcpy(&delta, &cudaDeltaArray[0], sizeof(int), cudaMemcpyDeviceToHost);


	std::cout << "Delta je " << delta;
	return 0;
}