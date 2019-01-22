
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <string>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include <fstream>

#include <functional>
#include <algorithm>

#include <ctime>

#define ACCURACY 0.01
#define NUM_OF_GPU_THREADS 2
#define BLOCK_SIZE 32
#define NUM_OF_GPU_BLOCKS 4

#define MAX 3
#define MIN 0


//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//	int i = threadIdx.x;
//	c[i] = a[i] + b[i];
//}
//
//int main()
//{
//	const int arraySize = 5;
//	const int a[arraySize] = { 1, 2, 3, 4, 5 };
//	const int b[arraySize] = { 10, 20, 30, 40, 50 };
//	int c[arraySize] = { 0 };
//
//	// Add vectors in parallel.
//	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addWithCuda failed!");
//		return 1;
//	}
//
//	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//		c[0], c[1], c[2], c[3], c[4]);
//
//	// cudaDeviceReset must be called before exiting in order for profiling and
//	// tracing tools such as Nsight and Visual Profiler to show complete traces.
//	cudaStatus = cudaDeviceReset();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceReset failed!");
//		return 1;
//	}
//
//	return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//	int *dev_a = 0;
//	int *dev_b = 0;
//	int *dev_c = 0;
//	cudaError_t cudaStatus;
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//		goto Error;
//	}
//
//	// Allocate GPU buffers for three vectors (two input, one output)    .
//	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	// Copy input vectors from host memory to GPU buffers.
//	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	// Launch a kernel on the GPU with one thread for each element.
//	addKernel << <1, size >> > (dev_c, dev_a, dev_b);
//
//	// Check for any errors launching the kernel
//	cudaStatus = cudaGetLastError();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//		goto Error;
//	}
//
//	// cudaDeviceSynchronize waits for the kernel to finish, and returns
//	// any errors encountered during the launch.
//	cudaStatus = cudaDeviceSynchronize();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//		goto Error;
//	}
//
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//Error:
//	cudaFree(dev_c);
//	cudaFree(dev_a);
//	cudaFree(dev_b);
//
//	return cudaStatus;
//}

void checkError(cudaError_t err, int line) {
	if (cudaSuccess != err) {
		std::cerr << "Error " << cudaGetErrorName(err) << " happenend: " << cudaGetErrorString(err) << " at line " << line << std::endl;
		exit(-1);
	}
}

//void matrixInit(float * matrix, int size) {
//	for (int i = 0; i < size; ++i) {
//		matrix[i] = rand() % (MAX - MIN + 1) + MIN;
//	}
//}

__global__ void sgemmKernel(float * cudaA, float * cudaB, float * cudaC, int m, int n, int k, float alpha, float beta) {

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	__shared__ float A[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float B[BLOCK_SIZE][BLOCK_SIZE];

	int row = bx * BLOCK_SIZE + tx;
	int col = by * BLOCK_SIZE + ty;

	float sum = 0;

	for (int i = 0; i < gridDim.y; i++) {

		int a_col = i * BLOCK_SIZE + ty;
		int a_row = row;

		if (a_row < m && a_col < k) {
			A[tx][ty] = cudaA[a_row + a_col * m];
		}

		int b_col = col;
		int b_row = i * BLOCK_SIZE + tx;

		if (b_row < k && b_col < n){
			B[tx][ty] = cudaB[b_row * n + b_col];
		}

		__syncthreads();

		if (row < m && col < n) {

			int j_end = ((i + 1) * BLOCK_SIZE < k) ? BLOCK_SIZE : k - i * BLOCK_SIZE;


			for (int j = 0; j < j_end; j++) {
				sum += A[tx][j] * B[j][ty];
			}
		}
		__syncthreads();
	}
	if (row < m && col < n) {
		cudaC[row + col * m] = 0;
		cudaC[row + col * m] = cudaC[row + col * m] * beta + sum * alpha;
	}
}
bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v)
{
	std::cerr << "Opening file:" << fn << std::endl;
	std::fstream f(fn, std::fstream::in);
	if (!f.good()) {
		return false;
	}

	// Read # of rows and cols
	f >> nr_row;
	f >> nr_col;

	float data;
	std::cerr << "Matrix dimension: " << nr_row << "x" << nr_col << std::endl;
	while (f.good()) {
		f >> data;
		v.push_back(data);
	}
	v.pop_back(); // remove the duplicated last element

	return true;
}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, std::vector<float>&v)
{
	std::cerr << "Opening file:" << fn << " for write." << std::endl;
	std::fstream f(fn, std::fstream::out);
	if (!f.good()) {
		return false;
	}

	// Read # of rows and cols
	f << nr_row << " " << nr_col << " ";

	std::cerr << "Matrix dimension: " << nr_row << "x" << nr_col << std::endl;
	for (int i = 0; i < v.size(); ++i) {
		f << v[i] << ' ';
	}
	f << "\n";
	return true;

}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
	if ((transa != 'N') && (transa != 'n')) {
		std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
		return;
	}

	if ((transb != 'T') && (transb != 't')) {
		std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
		return;
	}

	for (int mm = 0; mm < m; ++mm) {
		for (int nn = 0; nn < n; ++nn) {
			float c = 0.0f;
			for (int i = 0; i < k; ++i) {
				float a = A[mm + i * lda];
				float b = B[nn + i * ldb];
				c += a * b;
			}
			C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
		}
	}
}

void basicSgemm_par(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
	float *cudaA, *cudaB, *cudaC;
	int sizeA = m * k * sizeof(float), sizeB = k * n * sizeof(float), sizeC = m * n * sizeof(float);

	if ((transa != 'N') && (transa != 'n')) {
		std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
		return;
	}

	if ((transb != 'T') && (transb != 't')) {
		std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
		return;
	}

	checkError(cudaMalloc(&cudaA, sizeA), __LINE__);
	checkError(cudaMemcpy(cudaA, A, sizeA, cudaMemcpyHostToDevice), __LINE__);

	checkError(cudaMalloc(&cudaB, sizeB), __LINE__);
	checkError(cudaMemcpy(cudaB, B, sizeB, cudaMemcpyHostToDevice), __LINE__);

	checkError(cudaMalloc(&cudaC, sizeC), __LINE__);

	dim3 dimGrid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	sgemmKernel <<< dimGrid, dimBlock >>> (cudaA, cudaB, cudaC, m, n, k, alpha, beta);

	checkError(cudaMemcpy(C, cudaC, sizeC, cudaMemcpyDeviceToHost), __LINE__);

	checkError(cudaFree(cudaA), __LINE__);
	checkError(cudaFree(cudaB), __LINE__);
	checkError(cudaFree(cudaC), __LINE__);
}

int main(int argc, char *argv[]) {

	int matArow, matAcol;
	int matBrow, matBcol;

	std::vector<float> matA;
	std::vector<float> matBT;

	//int m, n, k;

	float timeSeq = 0.0f, timePar = 0.0f;

	cudaEvent_t start = cudaEvent_t();
	checkError(cudaEventCreate(&start), __LINE__);
	cudaEvent_t stop = cudaEvent_t();
	checkError(cudaEventCreate(&stop), __LINE__);

	if (argc != 4)
	{
		fprintf(stderr, "Expecting three input filenames\n");
		exit(-1);
	}


	/* Read in data */
	// load A
	readColMajorMatrixFile(argv[1], matArow, matAcol, matA);

	// load B^T
	readColMajorMatrixFile(argv[2], matBcol, matBrow, matBT);

	/*m = std::stoi(argv[1]);
	n = std::stoi(argv[2]);
	k = std::stoi(argv[3]);
	// A - m*k
	// B - k*n
	// C - m*n

	matArow = m;
	matAcol = k;
	matBrow = k;
	matBcol = n;
	// allocate space for A and B
	//matrixInit(&matA.front(), m*k);
	//matrixInit(&matBT.front(), k*n);*/

	// allocate space for C and D
	std::vector<float> matC(matArow * matBcol);
	std::vector<float> matD(matArow * matBcol);


	//clock_t begin = clock();
	cudaEventRecord(start, 0);
	// Use standard sgemm interface
	basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);
	clock_t end = clock();
	//timeSeq = float(end - begin) / CLOCKS_PER_SEC;
	checkError(cudaEventRecord(stop, 0), __LINE__);
	checkError(cudaEventSynchronize(stop), __LINE__);
	checkError(cudaEventElapsedTime(&timeSeq, start, stop), __LINE__);
	timeSeq /= 1000;

	cudaEventRecord(start, 0);
	// Use parallel sgemm interface
	basicSgemm_par('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matD.front(), matArow);
	checkError(cudaEventRecord(stop, 0), __LINE__);
	checkError(cudaEventSynchronize(stop), __LINE__);
	checkError(cudaEventElapsedTime(&timePar, start, stop), __LINE__);
	timePar /= 1000;

	checkError(cudaEventDestroy(start), __LINE__);
	checkError(cudaEventDestroy(stop), __LINE__);

	writeColMajorMatrixFile(argv[3], matArow, matBcol, matC);

	std::function<bool(double, double)> comparator = [](double left, double right) {
		// Lambda function to compare 2 doubles with ACCURACY
		return fabs(left - right) < ACCURACY;
	};

	std::cerr << "********************DZ3Z1**********************" << std::endl;
	std::cerr << "Elapsed time - SEQ: " << timeSeq << "." << std::endl;
	std::cerr << "Elapsed time - PAR: " << timePar << "." << std::endl;
	std::cerr << (std::equal(matC.begin(), matC.end(), matD.begin(), comparator) ? "TEST PASSED" : "TEST FAILED") << std::endl;
	std::cerr << "***********************************************" << std::endl;

	return 0;
}

