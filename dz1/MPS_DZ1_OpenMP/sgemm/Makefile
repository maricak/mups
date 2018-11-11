# C compiler
CC = g++
CC_FLAGS = -fopenmp -O2

prime: 
	$(CC) $(CC_FLAGS) dz1z1.cpp -o dz1z1 -lm 
	$(CC) $(CC_FLAGS) dz1z2.cpp -o dz1z2 -lm 
	$(CC) $(CC_FLAGS) dz1z3.cpp -o dz1z3 -lm 
	
	$(CC) $(CC_FLAGS) sgemm.cc -o sgemm -lm 

	$(CC) $(CC_FLAGS) cmp_files.cpp -o cmp_files -lm 

clean:
	rm -f sgemm dz1z1 dz1z2 dz1z3 result*
