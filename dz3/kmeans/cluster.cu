#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#include "kmeans.h"

/*---< cluster() >-----------------------------------------------------------*/
int cluster(int      numObjects,      /* number of input objects */
            int      numAttributes,   /* size of attribute of each object */
            float  **attributes,      /* [numObjects][numAttributes] */
            int      num_nclusters,
            float    threshold,       /* in:   */
            float ***cluster_centres /* out: [best_nclusters][numAttributes] */
    
            )
{
    int     nclusters;
    int    *membership;
    float **tmp_cluster_centres;

    membership = (int*) malloc(numObjects * sizeof(int));

    nclusters=num_nclusters;

    srand(7);
	
    tmp_cluster_centres = kmeans_clustering(attributes,
                                            numAttributes,
                                            numObjects,
                                            nclusters,
                                            threshold,
                                            membership);

    if (*cluster_centres) {
		free((*cluster_centres)[0]);
        free(*cluster_centres);
	}
	*cluster_centres = tmp_cluster_centres;

   
    free(membership);

    return 0;
}

