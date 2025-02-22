#include <mpi.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <iostream>

#include <functional>
#include <algorithm>

#define MASTER 0
#define ACCURACY 0.01

#define TAG_INCOMING_FROM_RIGHT 1000
#define TAG_INCOMING_FROM_LEFT 2000

double *jacobi_seq(int m, int n)
{
  double d, r, t, *b, *x, *xnew;
  int i, it;

  b = (double *)malloc(n * sizeof(double));
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

  /*
    Initialize the solution estimate to 0.
    Exact solution is (1,2,3,...,N).
  */
  for (i = 0; i < n; i++)
  {
    b[i] = 0.0;
    x[i] = 0.0;
  }
  b[n - 1] = (double)(n + 1);

  b[n - 1] = (double)(n + 1);

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

void parallelJacobiUpdateAndDifference(double *xnew_buffer, const double *x_buffer, double &d_local, int chunk, int n, int rank, int N)
{

  MPI_Request firstSendReq, lastSendReq;
  MPI_Status firstSendStatus, lastSendStatus;
  MPI_Request prevRecvReq, nextRecvReq;
  MPI_Status prevRecvStatus, nextRecvStatus;
  int receivedPrev = 0;
  int receivedNext = 0;
  int skippedFirst = 0;
  int skippedLast = 0;
  double xPrev, xNext;

  // for i == 0
  xnew_buffer[0] = 0;
  xnew_buffer[chunk - 1] = 0;
  if (rank == N - 1)
    xnew_buffer[chunk - 1] = n + 1;

  if (rank == 0)
  {
    if (N == 1)
    {
      xPrev = 0;
      receivedPrev = 1;
      xNext = 0;
      receivedNext = 1;
    }
    else
    {
      MPI_Isend(&x_buffer[chunk - 1], 1, MPI_DOUBLE, rank + 1, TAG_INCOMING_FROM_LEFT, MPI_COMM_WORLD, &lastSendReq);
      MPI_Irecv(&xNext, 1, MPI_DOUBLE, rank + 1, TAG_INCOMING_FROM_RIGHT, MPI_COMM_WORLD, &nextRecvReq); // attempt receiving the first element from the right-rank process
      xPrev = 0;
      receivedPrev = 1;
    }
  }
  else if (rank == N - 1)
  {
    MPI_Isend(&x_buffer[0], 1, MPI_DOUBLE, rank - 1, TAG_INCOMING_FROM_RIGHT, MPI_COMM_WORLD, &firstSendReq);
    MPI_Irecv(&xPrev, 1, MPI_DOUBLE, rank - 1, TAG_INCOMING_FROM_LEFT, MPI_COMM_WORLD, &prevRecvReq);
    xNext = 0;
    receivedNext = 1;
  }
  else
  {
    MPI_Isend(&x_buffer[0], 1, MPI_DOUBLE, rank - 1, TAG_INCOMING_FROM_RIGHT, MPI_COMM_WORLD, &firstSendReq);
    MPI_Isend(&x_buffer[chunk - 1], 1, MPI_DOUBLE, rank + 1, TAG_INCOMING_FROM_LEFT, MPI_COMM_WORLD, &lastSendReq);

    MPI_Irecv(&xPrev, 1, MPI_DOUBLE, rank - 1, TAG_INCOMING_FROM_LEFT, MPI_COMM_WORLD, &prevRecvReq);
    MPI_Irecv(&xNext, 1, MPI_DOUBLE, rank + 1, TAG_INCOMING_FROM_RIGHT, MPI_COMM_WORLD, &nextRecvReq);
  }

  for (int i = 1; i < chunk - 1; i++)
  {
    xnew_buffer[i] = 0; // replacement for b array

    xnew_buffer[i] = xnew_buffer[i] + x_buffer[i - 1];
    xnew_buffer[i] = xnew_buffer[i] + x_buffer[i + 1];
    xnew_buffer[i] = xnew_buffer[i] / 2.0;
    d_local = d_local + pow(x_buffer[i] - xnew_buffer[i], 2);
  }

  if (!receivedPrev)
  {
    MPI_Wait(&prevRecvReq, &prevRecvStatus);
  }
  xnew_buffer[0] = xnew_buffer[0] + xPrev;
  xnew_buffer[0] = xnew_buffer[0] + x_buffer[1];
  xnew_buffer[0] = xnew_buffer[0] / 2.0;
  d_local = d_local + pow(x_buffer[0] - xnew_buffer[0], 2);

  if (!receivedNext)
  {
    MPI_Wait(&nextRecvReq, &nextRecvStatus);
  }
  xnew_buffer[chunk - 1] = xnew_buffer[chunk - 1] + xNext;
  xnew_buffer[chunk - 1] = xnew_buffer[chunk - 1] + x_buffer[chunk - 2];
  xnew_buffer[chunk - 1] = xnew_buffer[chunk - 1] / 2.0;
  d_local = d_local + pow(x_buffer[chunk - 1] - xnew_buffer[chunk - 1], 2);
}

void parallelResidual(const double *xnew_buffer, double &r_local, int chunk, int n, int rank, int N)
{
  MPI_Request firstSendReq, lastSendReq;
  MPI_Status firstSendStatus, lastSendStatus;
  MPI_Request prevRecvReq, nextRecvReq;
  MPI_Status prevRecvStatus, nextRecvStatus;
  int receivedPrev = 0;
  int receivedNext = 0;
  int skippedFirst = 0;
  int skippedLast = 0;
  double xPrev, xNext;

  double t;
  double b = 0.0;

  // b = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 16
  if (rank == 0)
  {
    if (N == 1)
    {
      xPrev = 0;
      receivedPrev = 1;
      xNext = 0;
      receivedNext = 1;
    }
    else
    {
      MPI_Isend(&xnew_buffer[chunk - 1], 1, MPI_DOUBLE, rank + 1, TAG_INCOMING_FROM_LEFT, MPI_COMM_WORLD, &lastSendReq);
      MPI_Irecv(&xNext, 1, MPI_DOUBLE, rank + 1, TAG_INCOMING_FROM_RIGHT, MPI_COMM_WORLD, &nextRecvReq); // attempt receiving the first element from the right-rank process
      xPrev = 0;
      receivedPrev = 1;
    }
  }
  else if (rank == N - 1)
  {
    MPI_Isend(&xnew_buffer[0], 1, MPI_DOUBLE, rank - 1, TAG_INCOMING_FROM_RIGHT, MPI_COMM_WORLD, &firstSendReq);
    MPI_Irecv(&xPrev, 1, MPI_DOUBLE, rank - 1, TAG_INCOMING_FROM_LEFT, MPI_COMM_WORLD, &prevRecvReq);
    xNext = 0;
    receivedNext = 1;
  }
  else
  {
    MPI_Isend(&xnew_buffer[0], 1, MPI_DOUBLE, rank - 1, TAG_INCOMING_FROM_RIGHT, MPI_COMM_WORLD, &firstSendReq);
    MPI_Isend(&xnew_buffer[chunk - 1], 1, MPI_DOUBLE, rank + 1, TAG_INCOMING_FROM_LEFT, MPI_COMM_WORLD, &lastSendReq);

    MPI_Irecv(&xPrev, 1, MPI_DOUBLE, rank - 1, TAG_INCOMING_FROM_LEFT, MPI_COMM_WORLD, &prevRecvReq);
    MPI_Irecv(&xNext, 1, MPI_DOUBLE, rank + 1, TAG_INCOMING_FROM_RIGHT, MPI_COMM_WORLD, &nextRecvReq);
  }

  for (int i = 1; i < chunk - 1; i++)
  {
    t = -2.0 * xnew_buffer[i];
    t = t + xnew_buffer[i - 1];
    t = t + xnew_buffer[i + 1];
    r_local = r_local + t * t;
  }

  // Wait for first.
  if (!receivedPrev)
  {
    MPI_Wait(&prevRecvReq, &prevRecvStatus);
  }
  t = -2.0 * xnew_buffer[0];
  t = t + xPrev;
  t = t + xnew_buffer[1];
  r_local = r_local + t * t;

  // Wait for last.
  if (!receivedNext)
  {
    MPI_Wait(&nextRecvReq, &nextRecvStatus);
  }
  if (rank == N - 1)
    t = n + 1 - 2.0 * xnew_buffer[chunk - 1];
  else
  {
    t = -2.0 * xnew_buffer[chunk - 1];
  }
  t = t + xNext;
  t = t + xnew_buffer[chunk - 2];
  r_local = r_local + t * t;
}

void overwriteLocalArray(double **x_buffer, double **xnew_buffer)
{
  double *temp = *x_buffer;
  *x_buffer = *xnew_buffer;
  *xnew_buffer = temp;
}

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);

  double d;
  int i;
  int it;

  double r;
  double t;
  int m;
  int n;
  double *x;

  double timeStart, timePar, timeSeq;
  double *xPar, *xSeq;

  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc == 3)
  {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    if (n % size)
    {
      printf("Number n = %d must be divisible by %d!\n", n, size);
      MPI_Abort(MPI_COMM_WORLD, -1); // -1 -- error code
    }
  }
  else
  {
    m = 5000;
    n = 50000;
  }

  // Info every process will need
  int chunk = n / size;

  if (rank == MASTER)
  {
    // Use standard jacobi interface
    timeStart = MPI_Wtime();
    xSeq = jacobi_seq(m, n);
    timeSeq = MPI_Wtime() - timeStart;

    // Prepare structures for parallel implementation
    x = (double *)malloc(n * sizeof(double));

    printf("\n");
    printf("JACOBI_MPI:\n");
    printf("  C/MPI version - %d processes\n", size);
    printf("  Jacobi iteration to solve A*x=b.\n");
    printf("\n");
    printf("  Number of variables  N = %d\n", n);
    printf("  Number of iterations M = %d\n", m);

    printf("\n");
    printf("  IT     l2(dX)    l2(resid)\n");
    printf("\n");
  }

  // Everyone allocates buffers.
  double *x_buffer = (double *)calloc(chunk, sizeof(double));
  double *xnew_buffer = (double *)malloc(chunk * sizeof(double));
  double d_local;
  double r_local;

  //MPI_Barrier(MPI_COMM_WORLD);
  // Use parallel jacobi interface
  if (rank == MASTER)
  {
    timeStart = MPI_Wtime();
  }

  // Implement parallel

  MPI_Request gatherRequest;
  MPI_Request diffRequest;
  MPI_Request redRequest;
  MPI_Status diffStatus, redStatus, gatherStatus;

  for (it = 0; it < m; it++)
  {

    // Jacobi update and difference.
    d_local = 0;
    parallelJacobiUpdateAndDifference(xnew_buffer, x_buffer, d_local, chunk, n, rank, size);
    // Residual.
    r_local = 0;
    parallelResidual(xnew_buffer, r_local, chunk, n, rank, size);

    // Overwrite local X array.
    overwriteLocalArray(&x_buffer, &xnew_buffer);

    if (rank == MASTER)
    {
      if (it < 10 || m - 10 < it)
      {
        // Call reduces yourself.
        MPI_Ireduce(&d_local, &d, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD, &diffRequest);
        MPI_Ireduce(&r_local, &r, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD, &redRequest);
        // wait for difference and reduction values
        MPI_Wait(&diffRequest, &diffStatus);
        MPI_Wait(&redRequest, &redStatus);

        printf("  %8d  %14.6g  %14.6g\n", it, sqrt(d), sqrt(r));
      }
      if (it == 9)
      {
        printf("  Omitting intermediate results.\n");
      }
    }
    else
    {
      if (it < 10 || m - 10 < it)
      {
        MPI_Ireduce(&d_local, &d, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD, &diffRequest);
        MPI_Ireduce(&r_local, &r, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD, &redRequest);
      }
    }
  }

  // When the process is finished, gather results to master.
  MPI_Gather(x_buffer, chunk, MPI_DOUBLE, x, chunk, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

  if (rank == MASTER)
  {
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

    timePar = MPI_Wtime() - timeStart;

    xPar = x;

    std::function<bool(double, double)> comparator = [](double left, double right) {
      // Lambda function to compare 2 doubles with ACCURACY
      return fabs(left - right) < ACCURACY;
    };

    /*for(int i = 0; i < n; i++) {
      double left = xPar[i];
      double right = xSeq[i];
      if(!(fabs(left - right) < ACCURACY)) {
        std::cerr << "ERR:(" << i << ")" << left << " " << right << std::endl;
      }
    }*/

    std::vector<double> vectorPar(xPar, xPar + n), vectorSeq(xSeq, xSeq + n);

    std::cerr << "********************DZ1Z4**********************" << std::endl;
    std::cerr << "Elapsed time - SEQ: " << timeSeq << "." << std::endl;
    std::cerr << "Elapsed time - PAR(" << size << "): " << timePar << "." << std::endl;
    std::cerr << (std::equal(vectorPar.begin(), vectorPar.end(), vectorSeq.begin(), comparator) ? "TEST PASSED" : "TEST FAILED") << std::endl;
    std::cerr << "***********************************************" << std::endl;

    free(xSeq);
    free(xPar);
  }

  free(x_buffer);
  free(xnew_buffer);

  MPI_Finalize();

  return 0;
}
