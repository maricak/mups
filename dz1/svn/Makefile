# C compiler
CC = g++
CC_FLAGS = -fopenmp

prime: 
	$(CC) $(CC_FLAGS) dz1z1_1.cpp -o dz1z1_1 -lm 
	$(CC) $(CC_FLAGS) dz1z1_2.cpp -o dz1z1_2 -lm 
	$(CC) $(CC_FLAGS) dz1z1_4.cpp -o dz1z1_4 -lm 
	$(CC) $(CC_FLAGS) dz1z1_8.cpp -o dz1z1_8 -lm 
	
	$(CC) $(CC_FLAGS) dz1z2_1.cpp -o dz1z2_1 -lm 
	$(CC) $(CC_FLAGS) dz1z2_2.cpp -o dz1z2_2 -lm 
	$(CC) $(CC_FLAGS) dz1z2_4.cpp -o dz1z2_4 -lm 
	$(CC) $(CC_FLAGS) dz1z2_8.cpp -o dz1z2_8 -lm 
	
	$(CC) $(CC_FLAGS) sgemm.cc -o sgemm -lm 

	$(CC) $(CC_FLAGS) cmp_files.cpp -o cmp_files -lm 

clean:
	rm -f sgemm dz1z1_1 dz1z1_2 dz1z1_4 dz1z1_8 dz1z2_1 dz1z2_2 dz1z2_4 dz1z2_8 result*
