#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"
#include "cuda_runtime.h"

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#define NUM_OF_GPU_THREADS 1024 

extern double wtime(void);

int find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float **pts,         /* [npts][nfeatures] */
                       int     npts)
{
    int index, i;
    float min_dist=FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i=0; i<npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures);  /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

__device__ __host__  int cuda_find_nearest_point(float *pt, /* [nfeatures] */
	int nfeatures,
	float *pts, /* [npts][nfeatures] */
	int npts)
{
	int index, i;
	float min_dist = FLT_MAX;

	/* find the cluster center id with min distance to pt */
	for (i = 0; i < npts; i++)
	{
		float dist;
		dist = cuda_euclid_dist_2(pt, pts + i * nfeatures, nfeatures); /* no need square root */
		if (dist < min_dist)
		{
			min_dist = dist;
			index = i;
		}
	}
	return (index);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline
float euclid_dist_2(float *pt1,
                    float *pt2,
                    int    numdims)
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

    return(ans);
}


__device__ __host__  __inline float cuda_euclid_dist_2(float *pt1,
	float *pt2,
	int numdims)
{
	int i;
	float ans = 0.0;

	for (i = 0; i < numdims; i++)
		ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);

	return (ans);
}

/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{

    int      i, j, n=0, index, loop=0;
    int     *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float    delta;
    float  **clusters;   /* out: [nclusters][nfeatures] */
    float  **new_centers;     /* [nclusters][nfeatures] */
  

    /* allocate space for returning variable clusters[] */
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;

    /* randomly pick cluster centers */
    for (i=0; i<nclusters; i++) {
        //n = (int)rand() % npoints;
        for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[n][j];
		n++;
    }

    for (i=0; i<npoints; i++)
		membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int*) calloc(nclusters, sizeof(int));

    new_centers    = (float**) malloc(nclusters *            sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;
 
  
    do {
		
        delta = 0.0;

        for (i=0; i<npoints; i++) {
	        /* find the index of nestest cluster centers */
	        index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
	        /* if membership changes, increase delta by 1 */
	        if (membership[i] != index) delta += 1.0;

	        /* assign the membership to object i */
	        membership[i] = index;

	        /* update new cluster centers : sum of objects located within */
	        new_centers_len[index]++;
	        for (j=0; j<nfeatures; j++)          
				new_centers[index][j] += feature[i][j];
        }
      
	for(int i = 0; i < nclusters; i++) {
		printf("new_centers_len[%d]=%d\n", i, new_centers_len[i]);
	}
	printf("\n\n");

	/* replace old cluster centers with new_centers */
        for (i=0; i<nclusters; i++) {
            for (j=0; j<nfeatures; j++) {
                if (new_centers_len[i] > 0)
					clusters[i][j] = new_centers[i][j] / new_centers_len[i];
				new_centers[i][j] = 0.0;   /* set back to 0 */
			}
			new_centers_len[i] = 0;   /* set back to 0 */
		}
            
        //delta /= npoints;
	//printf("delta %f\n", delta);
    } while (delta > threshold);

  
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}


void checkError(cudaError_t err, int line) {
	if (cudaSuccess != err) {
		printf("Error ");
	      	printf("%s", cudaGetErrorName(err));
	       	printf(" happenend: ");
	      	printf("%s", cudaGetErrorString(err));
	        printf(" at line %d", line);
		exit(-1);
	}
}

__global__ void IntReduction(int* cuda_delta_array, int n, int* cuda_delta_result_array) {


  __shared__ int shared_delta_array[NUM_OF_GPU_THREADS];
  
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  int localId = threadIdx.x;

  __syncthreads();


  if (myId < n) {

    shared_delta_array[localId] = cuda_delta_array[myId];

  }

   __syncthreads();

  //if (myId < n) {
    if (blockIdx.x < gridDim.x - 1) {
     // printf("entered if");
      // 1024 elements to reduce
      for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (localId < s && myId < n) {
          shared_delta_array[localId] = shared_delta_array[localId] + shared_delta_array[localId + s];
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
        
        if ((localId < s) && (localId + s < remainingElements) && (myId < n)) {
		shared_delta_array[localId] = shared_delta_array[localId] + shared_delta_array[localId + s];
        }
        __syncthreads();
      }

    }

    // only element with local 0 id places result into resulting arrays
    if (localId == 0) {
	//printf("shared_d_array[%d] = %f\n", localId, shared_d_array[localId]);
      cuda_delta_result_array[blockIdx.x] = shared_delta_array[0];
      
    }


  __syncthreads();

}
__global__ void DeltaReduction(float* cuda_delta_array, int n, float* cuda_delta_result_array) {


  __shared__ float shared_delta_array[NUM_OF_GPU_THREADS];
  
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  int localId = threadIdx.x;

  __syncthreads();


  if (myId < n) {

    shared_delta_array[localId] = cuda_delta_array[myId];

  }

   __syncthreads();

  //if (myId < n) {
    if (blockIdx.x < gridDim.x - 1) {
     // printf("entered if");
      // 1024 elements to reduce
      for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (localId < s && myId < n) {
          shared_delta_array[localId] = shared_delta_array[localId] + shared_delta_array[localId + s];
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
        
        if ((localId < s) && (localId + s < remainingElements) && (myId < n)) {
		shared_delta_array[localId] = shared_delta_array[localId] + shared_delta_array[localId + s];
        }
        __syncthreads();
      }

    }

    // only element with local 0 id places result into resulting arrays
    if (localId == 0) {
	//printf("shared_d_array[%d] = %f\n", localId, shared_d_array[localId]);
      cuda_delta_result_array[blockIdx.x] = shared_delta_array[0];
      
    }


  __syncthreads();

}

__global__ void updateNewCenters(int npoints, float* cuda_delta_array, int* cuda_membership, float* cuda_features, int nfeatures, float* cuda_clusters, int nclusters, int* cuda_new_centers_len, float* cuda_new_centers) {


	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;

	float delta = 0;

	if(global_id < npoints) {
		int membership = cuda_membership[global_id]; // global mem access

		int index = cuda_find_nearest_point(cuda_features + global_id * nfeatures, nfeatures, cuda_clusters, nclusters);

		//printf("id=%d membership=%d index=%d\n", global_id, membership, index);
		if (membership != index) {
			delta = 1; // local delta every thread 
	
		// send to global delta array
		}
		cuda_membership[global_id] = membership;
		cuda_delta_array[global_id] = delta;
		if(delta != 0) {
		//	printf("global_id=%d, local_id=%d, delta=%d\n", global_id, local_id, delta);
		}
	
		cuda_delta_array[global_id] = delta;
		cuda_membership[global_id] = index;
	
	

		for(int i = 0; i < nclusters; i++) {
			cuda_new_centers_len[i * npoints + global_id] = 0;
		}
		cuda_new_centers_len[index * npoints + global_id] = 1; // race condition
	}/*for (j = 0; j < nfeatures; j++)
		new_centers[index][j] += feature[id][j];*/
}


/*----< kmeans_clustering() >---------------------------------------------*/
float **kmeans_clustering_par(float **feature, /* in: [npoints][nfeatures] */
	int nfeatures,
	int npoints,
	int nclusters,
	float threshold,
	int *membership) /* out: [npoints] */
{

	int i, j, n = 0, index, loop = 0;
	int *new_centers_len; /* [nclusters]: no. of points in each cluster */
	float delta;
	float **clusters;    /* out: [nclusters][nfeatures] */
	float **new_centers; /* [nclusters][nfeatures] */

	dim3 gridDim((npoints + NUM_OF_GPU_THREADS - 1) / NUM_OF_GPU_THREADS);
	dim3 blockDim(NUM_OF_GPU_THREADS);
	
	/* allocate space for returning variable clusters[] */
	/* allocate space for returning variable clusters[] */
	clusters = (float **)malloc(nclusters * sizeof(float *));
	clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
	for (i = 1; i < nclusters; i++)
		clusters[i] = clusters[i - 1] + nfeatures;

	/* randomly pick cluster centers */
	for (i = 0; i < nclusters; i++) {
		//n = (int)rand() % npoints;
		for (j = 0; j < nfeatures; j++)
			clusters[i][j] = feature[n][j];
		n++;
	}

	// INICIJALIZUJE CPU I KOPIRA U GLOBALNU MEMORIJU
	float * cuda_clusters;
	checkError(cudaMalloc(&cuda_clusters, nclusters * nfeatures * sizeof(float)), __LINE__);
	checkError(cudaMemcpy(cuda_clusters, clusters[0], nclusters * nfeatures * sizeof(float), cudaMemcpyHostToDevice), __LINE__);;

	for (i = 0; i < npoints; i++)
		membership[i] = -1;
	int* cuda_membership;
	checkError(cudaMalloc(&cuda_membership, npoints * sizeof(int)), __LINE__);
	checkError(cudaMemcpy(cuda_membership, membership, npoints * sizeof(int), cudaMemcpyHostToDevice), __LINE__);

	// init cuda features
	float* cuda_features;
	checkError(cudaMalloc(&cuda_features, npoints * nfeatures * sizeof(float)), __LINE__);
	checkError(cudaMemcpy(cuda_features, feature[0], npoints * nfeatures * sizeof(float), cudaMemcpyHostToDevice), __LINE__);

	// INICIJALIZUJE CPU
	/* need to initialize new_centers_len and new_centers[0] to all 0 */
	new_centers_len = (int *)calloc(nclusters, sizeof(int));
	int* cuda_new_centers_len;
	checkError(cudaMalloc(&cuda_new_centers_len, nclusters * npoints * sizeof(int)), __LINE__);
	int* cuda_new_centers_len_result;
	checkError(cudaMalloc(&cuda_new_centers_len_result, nclusters * gridDim.x * sizeof(int)), __LINE__);
	int* new_centers_len_result;
	new_centers_len_result = (int*)calloc(nclusters * gridDim.x, sizeof(int));

	new_centers = (float **)malloc(nclusters * sizeof(float *));
	new_centers[0] = (float *)calloc(nclusters * nfeatures, sizeof(float));
	for (i = 1; i < nclusters; i++)
		new_centers[i] = new_centers[i - 1] + nfeatures;

	float * cuda_new_centers;
	checkError(cudaMalloc(&cuda_new_centers, nclusters * nfeatures * sizeof(float)), __LINE__);
	checkError(cudaMemcpy(cuda_new_centers, new_centers[0], nclusters * nfeatures * sizeof(float), cudaMemcpyHostToDevice), __LINE__);


	// create cuda_delta array
	float* delta_result_array;
	delta_result_array = (float*)malloc(gridDim.x * sizeof(float));
	float* cuda_delta_array;
	checkError(cudaMalloc(&cuda_delta_array, npoints * sizeof(float)), __LINE__);
	float* cuda_delta_result_array;
	checkError(cudaMalloc(&cuda_delta_result_array, gridDim.x * sizeof(float)), __LINE__);
	
	
	// create new_centers_len

	//printf("KERNEL pre\n");
	printf("gridDim.x=%d, blockDim.x=%d\n", gridDim.x, blockDim.x);

	updateNewCenters <<< gridDim, blockDim >>>(npoints, cuda_delta_array, cuda_membership, cuda_features, nfeatures, cuda_clusters, nclusters, cuda_new_centers_len, cuda_new_centers);
	checkError(cudaDeviceSynchronize(), __LINE__);

	DeltaReduction <<< gridDim, blockDim >>>(cuda_delta_array, npoints, cuda_delta_result_array);
	checkError(cudaDeviceSynchronize(), __LINE__);
	checkError(cudaMemcpy(delta_result_array, cuda_delta_result_array, gridDim.x*sizeof(float), cudaMemcpyDeviceToHost), __LINE__);
	

	delta = 0.0f;
	for(int i = 0; i < gridDim.x; i++) {
	//	printf("delta[%d]=%f\n", i, delta_result_array[i]);
		delta += delta_result_array[i];
	}

	printf("KERNEL posle delta = %f\n", delta);
		

	for(int i = 0; i < nclusters; i++) {
		IntReduction <<<gridDim, blockDim>>> (cuda_new_centers_len + i * npoints, npoints, cuda_new_centers_len_result + i * gridDim.x);
		checkError(cudaDeviceSynchronize(), __LINE__);
	}
	checkError(cudaDeviceSynchronize(), __LINE__);
	checkError(cudaMemcpy(new_centers_len_result, cuda_new_centers_len_result, gridDim.x * nclusters * sizeof(int), cudaMemcpyDeviceToHost ), __LINE__);

	
	for(int i = 0; i < nclusters; i++) {
		for(int j = 0; j < gridDim.x; j++) {
			new_centers_len[i] += new_centers_len_result[i*gridDim.x + j];
		}
	}

	for(int i = 0; i < nclusters; i++) {
		printf("new_centers_len[%d]=%d\n", i, new_centers_len[i]);
	}

//do
	//{

	//	delta = 0.0;

	//	for (i = 0; i < npoints; i++)
	//	{
			/* find the index of nestest cluster centers */
	//		index = cuda_find_nearest_point(feature[i], nfeatures, clusters[0], nclusters);

			//printf("point(%d) index(%d)\n",  i, index);

			/* if membership changes, increase delta by 1 */
	//		if (membership[i] != index)
	//			delta += 1.0;

			/* assign the membership to object i */
	//		membership[i] = index;

			/* update new cluster centers : sum of objects located within */
	//		new_centers_len[index]++;
	//		for (j = 0; j < nfeatures; j++)
	//			new_centers[index][j] += feature[i][j];
	//	}

		/* replace old cluster centers with new_centers */
	//	for (i = 0; i < nclusters; i++)
	//	{
	//		for (j = 0; j < nfeatures; j++)
	//		{
	//			if (new_centers_len[i] > 0)
	//				clusters[i][j] = new_centers[i][j] / new_centers_len[i];
	//			new_centers[i][j] = 0.0; /* set back to 0 */
	//		}
	//		new_centers_len[i] = 0; /* set back to 0 */
	//	}

		//delta /= npoints;
//	} while (delta > threshold);

	free(new_centers[0]);
	free(new_centers);
	free(new_centers_len);

	return clusters;
}

