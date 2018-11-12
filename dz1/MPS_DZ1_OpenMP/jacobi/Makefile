# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O2

prime: 
	$(CC) $(CC_FLAGS) jacobi.c -o jacobi -lm 
	$(CC) $(CC_FLAGS) dz1z4.c -o dz1z4 -lm 

clean:
	rm -f jacobi dz1z4
