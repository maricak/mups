# C compiler
CC = g++
C = gcc 
CC_FLAGS = -fopenmp -O2

prime: 
	$(CC) $(CC_FLAGS) dz1z1.cpp -o dz1z1 -lm 
	$(CC) $(CC_FLAGS) dz1z2.cpp -o dz1z2 -lm 
	$(CC) $(CC_FLAGS) dz1z3.cpp -o dz1z3 -lm 	
	$(CC) $(CC_FLAGS) dz1z4.cpp -o dz1z4 -lm 
	
	$(C) $(CC_FLAGS) dz1z5.o -o dz1z5


dz1z5.o: dz1z5.c dz1z5.h
	$(CC) $(CC_FLAGS) dz1z5.c -c

clean:
	rm -f sgemm dz1z1 dz1z2 dz1z3 dz1z4 dz1z5 result*
