#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <iostream>

#include <functional>
#include <algorithm>

#include <omp.h>

#define N 8
#define ACCURACY 0.01

double *jacobi_seq(int m, int n)
{
  double d, r, t, *b, *x, *xnew;
  int i, it;

  b = (double *)calloc(n, sizeof(double));
  x = (double *)malloc(n * sizeof(double));
  xnew = (double *)malloc(n * sizeof(double));

  printf("\n");
  printf("JACOBI:\n");
  printf("  C++ sequential version\n");
  printf("  Jacobi iteration to solve A*x=b.\n");
  printf("\n");
  printf("  Number of variables  N = %d\n", n);
  printf("  Number of iterations M = %d\n", m);

  printf("\n");
  printf("  IT     l2(dX)    l2(resid)\n");
  printf("\n");

  b[n - 1] = (double)(n + 1);

  /*
    Initialize the solution estimate to 0.
    Exact solution is (1,2,3,...,N).
  */
  for (i = 0; i < n; i++)
  {
    x[i] = 0.0;
  }

  /* Iterate M times. */
  for (it = 0; it < m; it++)
  {
    /* Jacobi update. */
    for (i = 0; i < n; i++)
    {
      xnew[i] = b[i];
      if (0 < i)
      {
        xnew[i] = xnew[i] + x[i - 1];
      }
      if (i < n - 1)
      {
        xnew[i] = xnew[i] + x[i + 1];
      }
      xnew[i] = xnew[i] / 2.0;
    }
    /* Difference. */
    d = 0.0;
    for (i = 0; i < n; i++)
    {
      d = d + pow(x[i] - xnew[i], 2);
    }
    /* Overwrite old solution. */
    for (i = 0; i < n; i++)
    {
      x[i] = xnew[i];
    }
    /* Residual. */
    r = 0.0;
    for (i = 0; i < n; i++)
    {
      t = b[i] - 2.0 * x[i];
      if (0 < i)
      {
        t = t + x[i - 1];
      }
      if (i < n - 1)
      {
        t = t + x[i + 1];
      }
      r = r + t * t;
    }

    if (it < 10 || m - 10 < it)
    {
      printf("  %8d  %14.6g  %14.6g\n", it, sqrt(d), sqrt(r));
    }
    if (it == 9)
    {
      printf("  Omitting intermediate results.\n");
    }
  }

  /* Write part of final estimate. */
  printf("\n");
  printf("  Part of final solution estimate:\n");
  printf("\n");
  for (i = 0; i < 10; i++)
  {
    printf("  %8d  %14.6g\n", i, x[i]);
  }
  printf("...\n");
  for (i = n - 11; i < n; i++)
  {
    printf("  %8d  %14.6g\n", i, x[i]);
  }

  /* Free memory. */
  free(b);
  //free(x);
  free(xnew);

  return x;
}

double *jacobi_par(int m, int n)
{
  double d, r, t, *b, *x, *xnew;
  int i, it;

  b = (double *)malloc(n * sizeof(double));
  x = (double *)malloc(n * sizeof(double));
  xnew = (double *)malloc(n * sizeof(double));

  printf("\n");
  printf("JACOBI_OPENMP:\n");
  printf("  C/OpenMP version\n");
  printf("  Jacobi iteration to solve A*x=b.\n");
  printf("\n");
  printf("  Number of variables  N = %d\n", n);
  printf("  Number of iterations M = %d\n", m);

  printf("\n");
  printf("  IT     l2(dX)    l2(resid)\n");
  printf("\n");

/*
  Set up the right hand side.
  +
  Initialize the solution estimate to 0.
  Exact solution is (1,2,3,...,N).
*/
#pragma omp parallel for
  for (i = 0; i < n; i++)
  {
    b[i] = 0.0;
    x[i] = 0.0;
  }

  // ---> End parallel.

  b[n - 1] = (double)(n + 1);

  /* Iterate M times. */
  for (it = 0; it < m; it++)
  {
    /*
      Jacobi update + Difference.
    */
    d = 0.0;
#pragma omp parallel
    {
#pragma omp for reduction(+ \
                          : d)
      for (i = 0; i < n; i++)
      {
        xnew[i] = b[i];
        if (0 < i)
        {
          xnew[i] = xnew[i] + x[i - 1];
        }
        if (i < n - 1)
        {
          xnew[i] = xnew[i] + x[i + 1]; // radi se samo read nad nizom x -- nema potrebe za sinhronizacijom
        }
        xnew[i] = xnew[i] / 2.0;
        d = d + pow(x[i] - xnew[i], 2);
      }
// ---> End for.

/*  Synchronise point + Overwrite old solution. */
#pragma omp for
      for (i = 0; i < n; i++)
      {
        x[i] = xnew[i];
      }

/* Residual -- reduction. */
#pragma omp single
      r = 0.0;

#pragma omp for reduction(+ \
                          : r) private(t)
      for (i = 0; i < n; i++)
      {
        t = b[i] - 2.0 * x[i];
        if (0 < i)
        {
          t = t + x[i - 1];
        }
        if (i < n - 1)
        {
          t = t + x[i + 1];
        }
        r = r + t * t;
      }
    } // ---> End parallel.

    if (it < 10 || m - 10 < it)
    {
      printf("  %8d  %14.6g  %14.6g\n", it, sqrt(d), sqrt(r));
    }
    if (it == 9)
    {
      printf("  Omitting intermediate results.\n");
    }
  }

  /* Write part of final estimate. */
  printf("\n");
  printf("  Part of final solution estimate:\n");
  printf("\n");
  for (i = 0; i < 10; i++)
  {
    printf("  %8d  %14.6g\n", i, x[i]);
  }
  printf("...\n");
  for (i = n - 11; i < n; i++)
  {
    printf("  %8d  %14.6g\n", i, x[i]);
  }

  /* Free memory. */
  free(b);
  //free(x);
  free(xnew);

  return x;
}

int main(int argc, char *argv[])
{

  int m, n, i;

  double timeStart, timePar, timeSeq;
  double *xPar, *xSeq;

  if (argc == 3)
  {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
  }
  else
  {
    m = 5000;
    n = 50000;
  }

  omp_set_num_threads(N);

  // Use standard jacobi interface
  timeStart = omp_get_wtime();
  xSeq = jacobi_seq(m, n);
  timeSeq = omp_get_wtime() - timeStart;

  // Use parallel jacobi interface
  timeStart = omp_get_wtime();
  xPar = jacobi_par(m, n);
  timePar = omp_get_wtime() - timeStart;

  std::function<bool(double, double)> comparator = [](double left, double right) {
    // Lambda function to compare 2 doubles with ACCURACY
    return fabs(left - right) < ACCURACY;
  };

  std::vector<double> vectorPar(xPar, xPar + n), vectorSeq(xSeq, xSeq + n);

  std::cerr << "********************DZ1Z4**********************" << std::endl;
  std::cerr << "Elapsed time - SEQ: " << timeSeq << "." << std::endl;
  std::cerr << "Elapsed time - PAR(" << N << "): " << timePar << "." << std::endl;
  std::cerr << (std::equal(vectorPar.begin(), vectorPar.end(), vectorSeq.begin(), comparator) ? "TEST PASSED" : "TEST FAILED") << std::endl;
  std::cerr << "***********************************************" << std::endl;


  free(xSeq);
  free(xPar);

  return 0;
}
