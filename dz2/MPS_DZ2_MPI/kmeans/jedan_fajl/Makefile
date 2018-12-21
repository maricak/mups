# C compiler

C = mpicc
CC_FLAGS = -lm 

prime: $(C) $(CC_FLAGS) dz1z5.o -o dz1z5


dz1z5.o: dz1z5.c dz1z5.h
	$(C) $(CC_FLAGS) dz1z5.c -c

clean:
	rm -f sgemm dz1z1 dz1z2 dz1z3 dz1z4 dz1z5 result*
