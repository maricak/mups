// Zbog get_num_threads() i get_thread_num() funkcija.
#include <omp.h>

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include <fstream>

//#define DEBUG
#define NUM_THREADS 2

bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float> &v)
{
  // std::cerr << "Opening file:"<< fn << std::endl;
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

#ifdef DEBUG
  std::cerr << "Closed file:" << fn << std::endl;
#endif
}

bool writeColMajorMatrixFile(const char *fn, int nr_row, int nr_col, std::vector<float> &v)
{
  // std::cerr << "Opening file:"<< fn << " for write." << std::endl;
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

#ifdef DEBUG
  std::cerr << "Closed file:" << fn << std::endl;
#endif

  return true;
}

/* 
 * Base C implementation of MM
 */

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

#pragma omp parallel for collapse(2)

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

#ifdef DEBUG
#endif
}
int main(int argc, char *argv[])
{

  double timeStart, timeEnd;

  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  omp_set_num_threads(NUM_THREADS);

  if (argc != 4)
  {
    fprintf(stderr, "Expecting three input filenames\n");
    exit(-1);
  }

  /* Read in data */
  // load A
  readColMajorMatrixFile(argv[1], matArow, matAcol, matA);

  // load B^T
  readColMajorMatrixFile(argv[2], matBcol, matBrow, matBT);

  // allocate space for C
  std::vector<float> matC(matArow * matBcol);

  timeStart = omp_get_wtime();
  // Use standard sgemm interface
  basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);

  timeEnd = omp_get_wtime();
  
  writeColMajorMatrixFile(argv[3], matArow, matBcol, matC);


  std::cerr << "********************DZ1Z2**********************" << std::endl;
  std::cerr << "Elapsed time: " << timeEnd - timeStart << "." << std::endl;
  std::cerr << "Number of threads: " << NUM_THREADS << std::endl;
  std::cerr << "***********************************************" << std::endl;

  return 0;
}
