#include <mpi.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include <fstream>

#include <functional>
#include <algorithm>

#define MASTER 0
#define ACCURACY 0.01

bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float> &v)
{
  std::cerr << "Opening file:" << fn << std::endl;
  std::fstream f(fn, std::fstream::in);
  if (!f.good())
  {
    return false;
  }

  // Read # of rows and cols
  f >> nr_row;
  f >> nr_col;

  float data;
  std::cerr << "Matrix dimension: " << nr_row << "x" << nr_col << std::endl;
  while (f.good())
  {
    f >> data;
    v.push_back(data);
  }
  v.pop_back(); // remove the duplicated last element
}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, std::vector<float> &v)
{
  std::cerr << "Opening file:" << fn << " for write." << std::endl;
  std::fstream f(fn, std::fstream::out);
  if (!f.good())
  {
    return false;
  }

  // Read # of rows and cols
  f << nr_row << " " << nr_col << " ";

  float data;
  std::cerr << "Matrix dimension: " << nr_row << "x" << nr_col << std::endl;
  for (int i = 0; i < v.size(); ++i)
  {
    f << v[i] << ' ';
  }
  f << "\n";
  return true;
}

// sequential iterface for basicSgemm
void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
  if ((transa != 'N') && (transa != 'n'))
  {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }

  if ((transb != 'T') && (transb != 't'))
  {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  for (int mm = 0; mm < m; ++mm)
  {
    for (int nn = 0; nn < n; ++nn)
    {
      float c = 0.0f;
      for (int i = 0; i < k; ++i)
      {
        float a = A[mm + i * lda];
        float b = B[nn + i * ldb];
        c += a * b;
      }
      C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
    }
  }
}

// parallel interface for basicSgemm
void basicSgemm_par(char transa, char transb, int m, int n, int k, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
{
  int rank, size;
  int chunk;

  // get rank and size
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == MASTER)
  {
    if ((transa != 'N') && (transa != 'n'))
    {
      std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if ((transb != 'T') && (transb != 't'))
    {
      std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -2);
    }
    if (m % size)
    {
      printf("Inadequate number of processes");
      MPI_Abort(MPI_COMM_WORLD, -3);
    }
  }

  // svi procesim treba da znaj dimenzije matrica  parametre alpha i beta
  MPI_Bcast(&m, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&k, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

  //MPI_Bcast(&lda, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  //MPI_Bcast(&ldb, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  //MPI_Bcast(&ldc, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

  MPI_Bcast(&alpha, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&beta, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  // svaki proces obradjuje chunk redova izlazne matrice
  chunk = m / size;

  // alociranje bafera za komunikaciju
  float *Abuff, *Bbuff, *Cbuff, *Atmp;
  Abuff = new float[chunk * k]; // redovi matrice A
  if (rank != MASTER)
    B = new float[k * n];       // matrica B
  Cbuff = new float[chunk * n]; // redovi matrice C

  // chunk redova matrice A
  MPI_Datatype row_matA, chunk_row_matA;
  MPI_Type_vector(k, chunk, m, MPI_FLOAT, &row_matA);
  MPI_Type_create_resized(row_matA, 0, chunk * sizeof(float), &chunk_row_matA); // dodato zbog scattera
  MPI_Type_commit(&chunk_row_matA);

  
  /* chunk redova matrice C
  MPI_Datatype row_matC, chunk_row_matC;
  MPI_Type_vector(n, chunk, m, MPI_FLOAT, &row_matC);
  MPI_Type_create_resized(row_matC, 0, chunk * sizeof(float), &chunk_row_matC); // dodato zbog gathera
  MPI_Type_commit(&chunk_row_matC);
  */

  // matricaA - chunk redova
  MPI_Scatter(A, 1, chunk_row_matA, Abuff, chunk * k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  /* resenje bez tipa chunk_row_matA
    for (int i = 0; i < k; i++)
    {
      MPI_Scatter(&A[i * m], chunk, MPI_FLOAT, &Abuff[i * chunk], chunk, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    }
  */

  // matricaB - cela
  MPI_Bcast(B, k * n, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  Bbuff = B;

  // izracunavanje
  for (int mm = 0; mm < chunk; ++mm)
  {
    for (int nn = 0; nn < n; ++nn)
    {
      float c = 0.0f;
      for (int i = 0; i < k; ++i)
      {
        float a = Abuff[mm + i * chunk];
        float b = Bbuff[nn + i * n];
        c += a * b;
      }
      Cbuff[mm + nn * chunk] = Cbuff[mm + nn * chunk] * beta + alpha * c;
    }
  }

  // matricaC - chunk redova
  for (int i = 0; i < n; i++)
  {
    MPI_Gather(&Cbuff[i * chunk], chunk, MPI_FLOAT, &C[i * m], chunk, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  }
  //MPI_Gather(Cbuff, chunk * n, MPI_FLOAT, C, 1, chunk_row_matC, MASTER, MPI_COMM_WORLD);

  MPI_Type_free(&chunk_row_matA);
  //MPI_Type_free(&chunk_row_matC)

  // oslobadjanje bafera
  if (rank != MASTER)
  {
    delete Bbuff;
  }
  delete Abuff;
  delete Cbuff;
}

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);

  int rank, size;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int matArow, matAcol;
  int matBrow, matBcol;

  double timeSeq, timePar;
  std::vector<float> matA, matBT;

  if (rank == MASTER)
  {
    if (argc != 4)
    {
      fprintf(stderr, "Expecting three input filenames\n");
      MPI_Abort(MPI_COMM_WORLD, -4);
      //exit(-1);
    }

    /* Read in data */
    // load A
    readColMajorMatrixFile(argv[1], matArow, matAcol, matA);

    // load B^T
    readColMajorMatrixFile(argv[2], matBcol, matBrow, matBT);

    // allocate space for C
    std::vector<float> matC(matArow * matBcol);
    std::vector<float> matD(matArow * matBcol);

    // Use standard sgemm interface
    timeSeq = MPI_Wtime();
    basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);
    timeSeq = MPI_Wtime() - timeSeq;

    // Use parallel sgemm interface
    timePar = MPI_Wtime();
    basicSgemm_par('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matD.front(), matArow);
    timePar = MPI_Wtime() - timePar;

    writeColMajorMatrixFile(argv[3], matArow, matBcol, matD);

    std::function<bool(double, double)> comparator = [](double left, double right) {
      // Lambda function to compare 2 doubles with ACCURACY
      return fabs(left - right) < ACCURACY;
    };

    std::cerr << "********************DZ2Z1**********************" << std::endl;
    std::cerr << "Elapsed time - SEQ: " << timeSeq << "." << std::endl;
    std::cerr << "Elapsed time - PAR(" << size << "): " << timePar << "." << std::endl;
    std::cerr << (std::equal(matC.begin(), matC.end(), matD.begin(), comparator) ? "TEST PASSED" : "TEST FAILED") << std::endl;
    std::cerr << "***********************************************" << std::endl;
  }
  else
  {
    basicSgemm_par('N', 'T', 0, 0, 0, 1.0f, 0, 0, 0, 0, 0.0f, 0, 0);
  }

  MPI_Finalize();

  return 0;
}
