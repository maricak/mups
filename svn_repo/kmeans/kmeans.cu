#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
//#include <omp.h>
#include "getopt.h"

#include "kmeans.h"

extern double wtime(void);


#define ACCURACY 1
#define N 1024

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
        "Usage: %s [switches] -i filename\n"
        "       -i filename     :  file containing data to be clustered\n"
        "       -b                 :input file is in binary format\n"
		"       -k                 : number of clusters (default is 8) \n"
        "       -t threshold    : threshold value\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     nclusters=5;
           char   *filename = 0;           
           float  *buf;
           float **attributes;
           float **cluster_centres_seq=NULL;
	   float **cluster_centres_par=NULL;
           int     i, j;           
		   
	   int different = 0;

           int     numAttributes;
           int     numObjects;           
           char    line[1024];
           int     isBinaryFile = 0;
           int     nloops;
           float   threshold = 0.001;
		   double  timing;

	float timeSeq = 0.0f, timePar = 0.0f;
	cudaEvent_t start = cudaEvent_t(), stop = cudaEvent_t();

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	while ( (opt=getopt(argc,argv,"i:k:t:b"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'k': nclusters = atoi(optarg);
                      break;
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }

    if (filename == 0) usage(argv[0]);

    numAttributes = numObjects = 0;

    /* from the input file, get the numAttributes and numObjects ------------*/
   
    if (isBinaryFile) {
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
//        read(infile, &numObjects,    sizeof(int));
//        read(infile, &numAttributes, sizeof(int));
   

        /* allocate space for attributes[] and read attributes of all objects */
        buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
        attributes    = (float**)malloc(numObjects*             sizeof(float*));
        attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));
        for (i=1; i<numObjects; i++)
            attributes[i] = attributes[i-1] + numAttributes;

//        read(infile, buf, numObjects*numAttributes*sizeof(float));

//        close(infile);
    }
    else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        while (fgets(line, 1024, infile) != NULL)
            if (strtok(line, " \t\n") != 0)
                numObjects++;
        rewind(infile);
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first attribute): numAttributes = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) numAttributes++;
                break;
            }
        }
     

        /* allocate space for attributes[] and read attributes of all objects */
        buf           = (float*) malloc(numObjects*numAttributes*sizeof(float));
        attributes    = (float**)malloc(numObjects*             sizeof(float*));
        attributes[0] = (float*) malloc(numObjects*numAttributes*sizeof(float));
        for (i=1; i<numObjects; i++)
            attributes[i] = attributes[i-1] + numAttributes;
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue; 
            for (j=0; j<numAttributes; j++) {
                buf[i] = atof(strtok(NULL, " ,\t\n"));
                i++;
            }
        }
        fclose(infile);
    }
  
    nloops = 1;	
	printf("I/O completed\n");

	memcpy(attributes[0], buf, numObjects*numAttributes*sizeof(float));

	cudaEventRecord( start, 0 );
//	timing = omp_get_wtime();
    for (i=0; i<nloops; i++) {
        		
        cluster_centres_seq = NULL;
        cluster(numObjects,
                numAttributes,
                attributes,           /* [numObjects][numAttributes] */
                nclusters,
                threshold,
                &cluster_centres_seq   
               );

     
    }

    cudaEventRecord( stop, 0 );

  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &timeSeq, start, stop );

  timeSeq /= 1000;
 //   timing = omp_get_wtime() - timing;
	printf("===================================SEQ=======================================\n\n");
	printf("number of Clusters %d\n",nclusters); 
	printf("number of Attributes %d\n\n",numAttributes); 
    /*printf("Cluster Centers Output\n"); 
	printf("The first number is cluster number and the following data is arribute value\n");
	printf("=============================================================================\n\n");
	
    for (i=0; i<nclusters; i++) {
		printf("%d: ", i);
        for (j=0; j<numAttributes; j++)
            printf("%f ", cluster_centres[i][j]);
        printf("\n\n");
    }*/
	printf("Time for process: %f\n", timeSeq);


  cudaEventRecord( start, 0 );
	for (i = 0; i < nloops; i++)
	{

		cluster_centres_par = NULL;
		cluster_par(numObjects,
			numAttributes,
			attributes, /* [numObjects][numAttributes] */
			nclusters,
			threshold,
			&cluster_centres_par);
	}
//	timing = omp_get_wtime() - timing;

  cudaEventRecord( stop, 0 );

  cudaEventSynchronize( stop );
  cudaEventElapsedTime( &timePar, start, stop );

  timePar /= 1000;

	printf("===================================PAR=======================================\n\n");
	printf("number of threads: %d\n", N);
	printf("number of Clusters %d\n", nclusters);
	printf("number of Attributes %d\n\n", numAttributes);
	/*
	printf("Cluster Centers Output\n");
	printf("The first number is cluster number and the following data is arribute value\n");
	printf("=============================================================================\n\n");

	for (i = 0; i < nclusters; i++)
	{
	  printf("%d: ", i);
	  for (j = 0; j < numAttributes; j++)
		printf("%f ", cluster_centres_par[i][j]);
	  printf("\n\n");
	}
	*/
	printf("Time for process: %f\n", timePar);

	for (i = 0; i < nclusters; i++)
	{
		for (j = 0; j < numAttributes; j++)
		{
			if (!(fabs(cluster_centres_par[i][j] - cluster_centres_seq[i][j]) < ACCURACY))
			{
				different = 1;
				printf("ERR: (%d,%d) par=%f seq=%f\n", i, j, cluster_centres_par[i][j], cluster_centres_seq[i][j]);
				//break;
			}
		}
		//if (different)
		//  break;
	}

	printf(different ? "TEST FAILED\n" : "TEST PASSED\n");
	free(attributes);
	free(cluster_centres_seq[0]);
	free(cluster_centres_par[0]);
	free(cluster_centres_seq);
	free(cluster_centres_par);
	free(buf);
	return (0);

}

