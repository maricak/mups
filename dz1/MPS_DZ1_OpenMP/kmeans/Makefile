# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2 

dz1z5: dz1z5.o 
	$(CC) $(CC_FLAGS) dz1z5.o -o dz1z5

dz1z5.o: dz1z5.c dz1z5.h
	$(CC) $(CC_FLAGS) dz1z5.c -c	

clean:
	rm -f *.o *~ dz1z5 
