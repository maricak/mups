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

void basicSgemm_par(char transa, char transb, int m, int n, int k, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
{
  int rank, size;
  int chunk;

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
  MPI_Bcast(&m, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&k, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

  //MPI_Bcast(&lda, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  // MPI_Bcast(&ldb, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  //MPI_Bcast(&ldc, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

  MPI_Bcast(&alpha, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&beta, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  chunk = m / size; // svaki proces obradjuje chunk redova izlazne matrice
  //printf("RANK=%d, chunk=%d, m=%d, n=%d, k=%d, lds=%d, ldb=%d, ldc=%d, alpha-%f, beta=%f\n", rank, chunk, m, n, k, lda, ldb, ldc, alpha, beta);

  float *Abuff, *Bbuff, *Cbuff, *Atmp;
  // alocirati prostor
  Abuff = new float[chunk * k];
  Atmp = new float[chunk * k];
  if (rank != MASTER)
    B = new float[k * n];
  Cbuff = new float[chunk * n];

  /*MPI_Datatype chunk_col; // kolona sa chunk elemenata
  MPI_Type_vector(k, chunk, m, MPI_FLOAT, &chunk_col);
  MPI_Type_commit(&chunk_col);
*/
  // matricaA - chunk redova
  //MPI_Scatter(A, 1, chunk_col, Abuff, 1, chunk_col, MASTER, MPI_COMM_WORLD);
  for (int i = 0; i < k; i++)
  {
    MPI_Scatter(&A[i * m], chunk, MPI_FLOAT, &Abuff[i * chunk], chunk, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  }  
  //MPI_Scatter(A, 1, chunk_col, Atmp, chunk * k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  /*printf("Proces %d: Abuff ", rank);
  for (int i = 0; i < chunk * k; i++)
  {
    printf("%f ", Abuff[i]);
  }
  printf("\n");
  printf("Proces %d: Atmp ", rank);
  for (int i = 0; i < chunk * k; i++)
  {
    printf("%f ", Atmp[i]);
  }

  printf("\n");
*/
  // matricaB - cela
  MPI_Bcast(B, k * n, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  Bbuff = B;

  /*printf("Proces %d: matB ", rank);
  for (int i = 0; i < n * k; i++)
  {
    printf("%f ", Bbuff[i]);
  }
  printf("\n");
*/
  for (int mm = 0; mm < chunk; ++mm)
  {
    for (int nn = 0; nn < n; ++nn)
    {
      float c = 0.0f;
      for (int i = 0; i < k; ++i)
      {
        //printf("RANK(%d): A(%d)B(%d)\n", rank, mm+i*chunk, nn+i*ldb);
        float a = Abuff[mm + i * chunk];
        float b = Bbuff[nn + i * n];
        c += a * b;
      }
      Cbuff[mm + nn * chunk] = Cbuff[mm + nn * chunk] * beta + alpha * c;
      //printf("RANK(%d): C(%d) = %f\n", rank, mm+nn*chunk, Cbuff[mm+nn*chunk]);
    }
  }
  /*
  printf("Proces %d: matCbuff ", rank);
  for (int i = 0; i < n * chunk; i++)
  {
    printf("%f ", Cbuff[i]);
  }
  printf("\n");
*/
  // matricaC - chunk redova
  for (int i = 0; i < n; i++)
  {
    MPI_Gather(&Cbuff[i * chunk], chunk, MPI_FLOAT, &C[i * m], chunk, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  }

  /*if (rank == MASTER)
  {
    printf("Proces %d: matC ", rank);
    for (int i = 0; i < n * m; i++)
    {
      printf("%f ", C[i]);
    }
    printf("\n");
  }*/
  //MPI_Type_free(&chunk_col);
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
    //printf("Finished seq!\n");
    timeSeq = MPI_Wtime() - timeSeq;

    // parallel
    timePar = MPI_Wtime();
    basicSgemm_par('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matD.front(), matArow);
    //printf("Finished par!\n");
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
