C = mpicc
CC = mpic++
CCFLAGS = -lm

prime: 
	$(CC) $(CCFLAGS) dz2z1.cpp -o dz2z1
	$(CC) $(CCFLAGS) dz2z2.cpp -o dz2z2
	$(CC) $(CCFLAGS) dz2z3.cpp -o dz2z3 
	$(C) $(CCFLAGS) dz2z4.c dz2z4.h -o dz2z4

clean:
	rm -f dz2z1 dz2z2 dz2z3 dz2z4 result*
