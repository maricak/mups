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

#include <functional>
#include <algorithm>

//#define DEBUG
#define N 1
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

#ifdef DEBUG
  std::cerr << "Closed file:" << fn << std::endl;
#endif
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

#ifdef DEBUG
  std::cerr << "Closed file:" << fn << std::endl;
#endif

  return true;
}

/* 
 * Parallel C, openMP implementation of MM
 */

void basicSgemm_par(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
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

#pragma omp parallel
  {

#pragma omp single
    {
      for (int mm = 0; mm < m; ++mm)
      {

#pragma omp task default(none) shared(A, B, C, n, k, lda, ldb, ldc, alpha, beta), firstprivate(mm)
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
        } // task
      }
    } // single
  }   // parallel
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

int main(int argc, char *argv[])
{

  double timeStart, timePar, timeSeq;

  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  omp_set_num_threads(N);

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

  // allocate space for C - output for standard interface
  std::vector<float> matC(matArow * matBcol);

  // allocate space for D - output for parallel interface
  std::vector<float> matD(matArow * matBcol);

  // Use standard sgemm interface
  timeStart = omp_get_wtime();
  basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);
  timePar = omp_get_wtime() - timeStart;

  // Use parallel sgemm interface
  timeStart = omp_get_wtime();
  basicSgemm_par('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matD.front(), matArow);
  timeSeq = omp_get_wtime() - timeStart;

  /* Write result */
  writeColMajorMatrixFile(argv[3], matArow, matBcol, matD);

  std::function<bool(double, double)> comparator = [](double left, double right) {
    // Lambda function to compare 2 doubles with ACCURACY
    return fabs(left - right) < ACCURACY;
  };

  std::cerr << "********************DZ1Z3**********************" << std::endl;
  std::cerr << "Elapsed time - SEQ: " << timeSeq << "." << std::endl;
  std::cerr << "Elapsed time - PAR(" << N << "): " << timePar << "." << std::endl;
  std::cerr << (std::equal(matC.begin(), matC.end(), matD.begin(), comparator) ? "TEST PASSED" : "TEST FAILED") << std::endl;
  std::cerr << "***********************************************" << std::endl;

  return 0;
}
