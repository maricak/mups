#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"

#define RANDOM_MAX 2147483647
#define NUM_OF_GPU_THREADS 1024

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

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

__device__ __inline float cuda_euclid_dist_2(float *pt1,
	float *pt2,
	int numdims)
{
	int i;
	float ans = 0.0;

	for (i = 0; i < numdims; i++)
		ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);

	return (ans);
}

__device__  int cuda_find_nearest_point(float *pt, /* [nfeatures] */
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
    } while (delta > threshold);

  
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}




__global__ void updateNewCenters(float* cuda_delta_array, int* cuda_membership, float* cuda_features, int nfeatures, float* cuda_clusters, int nclusters, int* cuda_new_centers_len, float* cuda_new_centers) {

	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	int local_id = threadIdx.x;

	float delta = 0;
	int membership = cuda_membership[global_id]; // global mem access

	int index = cuda_find_nearest_point(cuda_features + global_id * nfeatures, nfeatures, cuda_clusters, nclusters);

	if (membership != index) {
		delta = 1; // local delta every thread 

		// send to global delta array
	}

	cuda_delta_array[global_id] = delta;
	cuda_membership[global_id] = index;


	/*cuda_new_centers_len[index]++; // race condition
	for (j = 0; j < nfeatures; j++)
		new_centers[index][j] += feature[id][j];*/
}


/*----< kmeans_clustering_par() >---------------------------------------------*/
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
	cudaMalloc(&cuda_clusters, nclusters * nfeatures * sizeof(float));
	cudaMemcpy(cuda_clusters, clusters[0], nclusters * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

	for (i = 0; i < npoints; i++)
		membership[i] = -1;
	int* cuda_membership;
	cudaMalloc(&cuda_membership, npoints * sizeof(int));
	cudaMemcpy(cuda_membership, membership, npoints * sizeof(int), cudaMemcpyHostToDevice);

	// init cuda features
	float* cuda_features;
	cudaMalloc(&cuda_features, npoints * nfeatures * sizeof(float));
	cudaMemcpy(cuda_features, feature[0], npoints * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

	// INICIJALIZUJE CPU
	/* need to initialize new_centers_len and new_centers[0] to all 0 */
	new_centers_len = (int *)calloc(nclusters, sizeof(int));

	int* cuda_new_centers_len;
	cudaMalloc(&cuda_new_centers_len, nclusters * sizeof(int));
	cudaMemcpy(cuda_new_centers_len, new_centers_len, nclusters * sizeof(int), cudaMemcpyHostToDevice);

	new_centers = (float **)malloc(nclusters * sizeof(float *));
	new_centers[0] = (float *)calloc(nclusters * nfeatures, sizeof(float));
	for (i = 1; i < nclusters; i++)
		new_centers[i] = new_centers[i - 1] + nfeatures;

	float * cuda_new_centers;
	cudaMalloc(&cuda_new_centers, nclusters * nfeatures * sizeof(float));
	cudaMemcpy(cuda_new_centers, new_centers[0], nclusters * nfeatures * sizeof(float), cudaMemcpyHostToDevice);

	// create cuda_delta array
	float* cuda_delta_array;
	cudaMalloc(&cuda_delta_array, npoints * sizeof(float));

	// SVAKA NIT KOPIRA SVOJ FEATURE[i] U SHARED MATRICU SHARED_FEATURES[1024][30]
	// SAMO NITI SA ID-EM 0..NCLUSTERS-1 KOPIRAJU CLUSTERS U SHARED_CLUSTERS

	dim3 gridDim((npoints + NUM_OF_GPU_THREADS - 1) / NUM_OF_GPU_THREADS);
	dim3 blockDim(NUM_OF_GPU_THREADS);

	//do
	//{
	// copy new_centers, new_centers_len, clusters		


	updateNewCenters <<< gridDim, blockDim >>> (cuda_delta_array, cuda_membership, cuda_features, nfeatures, cuda_clusters, nclusters, cuda_new_centers_len, cuda_new_centers);

	//	// KRAJ KERNELA
	//	// IMAMO REDUKOVANO NEW_CENTER I NEW_CENTERS_LEN KOJI SU U GLOBALNOJ CUDA MEMORIJI. KOPIRAMO IH U CPU MEMORIJU, IZRACUNAMO NOVO I NA OSNOVU
	//	// DELTA ODREDIMO DAL TREBANOVA ITERACIJA. AKO TREBA U CUDA PROSTOR KOPIRAMO CLUSTERS KOJI SMO IZRACUNALI
	//	// reduced new_centers and new_centers_len
	//	cudaMemcpy(new_centers_len, cuda_new_centers_len, nclusters * sizeof(int), cudaMemcpyDeviceToHost);
	//	cudaMemcpy(new_centers[0], cuda_new_centers, nclusters * nfeatures * sizeof(float), cudaMemcpyDeviceToHost);
	//	// CPU
	//	/* replace old cluster centers with new_centers */
	//	/*int xi, xj;
	//	for (xi = 0; xi < nclusters; xi++)
	//	{
	//		for (xj = 0; xj < nfeatures; xj++)
	//		{
	//			if (new_centers_len[xi] > 0)
	//				clusters[xi][xj] = new_centers[xi][xj] / new_centers_len[xi];
	//			new_centers[xi][xj] = 0.0; /* set back to 0 */
	//		}
	//		/*new_centers_len[xi] = 0; /* set back to 0 */
	//	//}
	//	// delta reduction
	//	// deltaReduction<<< gridDim, blockDim >>>();
	//	//delta /= npoints;
	//} while (delta > threshold);*/
	//free(new_centers[0]);
	//free(new_centers);
	//free(new_centers_len);
	//return clusters;*/
}