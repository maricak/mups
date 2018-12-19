#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) {
	int rank, size;
	
	/* Inicijalizujemo MPI */
	...
	
	/* Trazimo svoj rang unutar MPI sveta (prvi je 0) */
	...
	
	/* Trazimo velicinu MPI sveta */
	... 
	
	/* Ispisujemo poruku na standardnom izlazu */
	printf("Zdravo svima! Ja sam broj %d od ukupno %d\n", rank, size);
	
	/* Zavrsavamo MPI */
	...
	
	return 0;
}

