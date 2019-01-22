# C compiler
CC = nvcc
CC_FLAGS = -g -fopenmp -O2 

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o 
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o  -o kmeans

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.cu -c
	
getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c
	
kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) kmeans.cu -c

kmeans_clustering.o: kmeans_clustering.cu kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering.cu -c

clean:
	rm -f *.o *~ kmeans 
