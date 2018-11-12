#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

int main(int argc, char *argv[])
{

  double *b;
  double d;
  int i;
  int it;
  int m;
  int n;
  double r;
  double t;
  double *x;
  double *xnew;

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

#pragma omp parallel for
  /*
  Set up the right hand side.
  +
  Initialize the solution estimate to 0.
  Exact solution is (1,2,3,...,N).
*/
  for (i = 0; i < n; i++)
  {
    b[i] = 0.0;
    x[i] = 0.0;
  }

  // ---> End parallel.

  b[n - 1] = (double)(n + 1);

  /*
  Iterate M times.
*/
  for (it = 0; it < m; it++)
  {
    /*
      Jacobi update + Difference.
    */
    d = 0.0;
    #pragma omp parallel
    {
      #pragma omp for
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

        /*
          Synchronise point + Overwrite old solution.
        */
        #pragma omp for
        for (i = 0; i < n; i++)
        {
          x[i] = xnew[i];
        }

        /*
        Residual -- reduction.
        */
        #pragma omp single
          r = 0.0;
        
        #pragma omp for reduction(+:r) private(t)
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


  } // ---> End M-iteration loop.

  /*
  Write part of final estimate.
  */
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
  /*
  Free memory.
*/
  free(b);
  free(x);
  free(xnew);

  return 0;
}
