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

enum Tags
{
  ROW_TAG = 1000,
  COLUMN_TAG,
  ROW_INDEX_TAG,
  COLUMN_INDEX_TAG,
  RESULT_TAG,
  END_TAG
};

MPI_Datatype Arow;
MPI_Datatype Bcol;
MPI_Datatype Crow;

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

void basicSgemm_par(int rank, int size, char transa, char transb, int m, int n, int k, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
{
  int chunk;

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

    //printf("m=%d, n=%d, k=%d\n", m, n, k);
  }

  // svi procesi treba da znaju velicine matrica i parametre alpha i beta
  // mat A mxk --- matB kxn --- matC mxn
  MPI_Bcast(&k, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&alpha, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&beta, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  // tip koji predstavla red matrice A
  MPI_Type_vector(k, 1, m, MPI_FLOAT, &Arow);
  MPI_Type_commit(&Arow);

  // tip koji predstavlja kolonu matice B
  MPI_Type_vector(k, 1, n, MPI_FLOAT, &Bcol);
  //  MPI_Type_contiguous(k, MPI_FLOAT, &Bcol);
  MPI_Type_commit(&Bcol);

  // tip koji predstavlja red matrice C
  MPI_Type_vector(n, 1, m, MPI_FLOAT, &Crow);
  MPI_Type_commit(&Crow);

  if (rank != MASTER)
  {
    // Slave procesi alociraju prostor za matricu B
    B = new float[n * k];
  }
  // svim procesima treba cela matrica B
  MPI_Bcast(B, n * k, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  if (rank == MASTER)
  {
    int count = m; // broj redova matice C
    int sendCount = 0; // broj poslatih taskova
    int receiveCount = 0; // broj primljenih rezultata

    for (sendCount = 0; sendCount < count && sendCount + 1 < size; sendCount++)
    {
      // svakom procesu posalji taks - izracunavanje jednog reda matrice C 
      //int row = sendCount / n;
      int row = sendCount;
      //int col = sendCount % n;

      MPI_Request request;

      // salje se redni broj reda
      MPI_Send(&row, 1, MPI_INT, sendCount + 1, ROW_INDEX_TAG, MPI_COMM_WORLD);
      //MPI_Send(&col, 1, MPI_INT, sendCount + 1, COLUMN_INDEX_TAG, MPI_COMM_WORLD);
      // salje se red matice A
      MPI_Isend(&A[row], 1, Arow, sendCount + 1, ROW_TAG, MPI_COMM_WORLD, &request);
      //MPI_Isend(&B[col], 1, Bcol, sendCount + 1, COLUMN_TAG, MPI_COMM_WORLD, &request);
      //printf("Proces %d sends Arow=%d to process %d\n", rank, row, sendCount + 1);
      //printf("Proces %d sends Bcol=%d to process %d\n", rank, col, sendCount + 1);
      //    printf("MASTER sends (%d, %d) to process%d\n", row, col, sendCount + 1);
    }
    for (; sendCount + 1 < size; sendCount++)
    {
      // posalji END procesima koji su visak
      MPI_Request request;
      int dummy;
      MPI_Isend(&dummy, 1, MPI_INT, sendCount + 1, END_TAG, MPI_COMM_WORLD, &request);
    }
    //printf("-----------------------------------\n");

    int row, col;
    float result;
    MPI_Status status;
    // dok ne primis sve rezultate
    while (receiveCount < count)
    {
      // cekaj podatak od slavea
      //printf("MASTER CEKA\n");
      // prima se redni broj reda rezultata
      MPI_Recv(&row, 1, MPI_INT, MPI_ANY_SOURCE, ROW_INDEX_TAG, MPI_COMM_WORLD, &status);
     // MPI_Recv(&col, 1, MPI_INT, status.MPI_SOURCE, COLUMN_INDEX_TAG, MPI_COMM_WORLD, 0);
     // prime sa red matice C
      MPI_Recv(&C[row], 1, Crow, status.MPI_SOURCE, RESULT_TAG, MPI_COMM_WORLD, 0);
      /*printf("MASTER prmio C(%d,%d)=%f\n", row, col, result);

      printf("MASTER primio: C: ");
      for(int i = 0; i < n * m; i++) {
        printf("%f ", C[i]);
      }
      printf("\n");
*/
      //C[col * m + row] = result;
      receiveCount++;
      // ako ima jos taskova da se salju, posalji task procesu od koga si primio rezultat
      if (sendCount < count)
      {
        //int row = sendCount / n;
        int row = sendCount;
        //int col = sendCount % n;

        MPI_Request request;
        // salje se redni broj reda
        MPI_Send(&row, 1, MPI_INT, status.MPI_SOURCE, ROW_INDEX_TAG, MPI_COMM_WORLD);
        //printf("Proces %d sends Arow=%d to process %d\n", rank, row, status.MPI_SOURCE);
        //MPI_Send(&col, 1, MPI_INT, status.MPI_SOURCE, COLUMN_INDEX_TAG, MPI_COMM_WORLD);
        //printf("Proces %d sends Dcol=%d to process %d\n", rank, col, status.MPI_SOURCE);
        // salje se red matrice A
        MPI_Isend(&A[row], 1, Arow, status.MPI_SOURCE, ROW_TAG, MPI_COMM_WORLD, &request);
        //MPI_Isend(&B[col], 1, Bcol, status.MPI_SOURCE, COLUMN_TAG, MPI_COMM_WORLD, &request);
        // printf("MASTER sends (%d, %d) to process%d\n", row, col, status.MPI_SOURCE);

        sendCount++;
      }
      else
      {
        // posalji END
        MPI_Request request;
        int dummy;
       // printf("Send END to %d\n", status.MPI_SOURCE);
        MPI_Isend(&dummy, 1, MPI_INT, status.MPI_SOURCE, END_TAG, MPI_COMM_WORLD, &request);
      }
    }
  }
  else
  { // SLAVE
    //printf("SLAVE %d\n", rank);
    float *Abuff, *Bbuff, *Cbuff;
    int row, col;
    bool needed = true;
    MPI_Status status;

    // alocirati prostor za bafere
    Abuff = new float[k]; // red matrice A
    //Bbuff = new float[k]; // kolona
    Bbuff = B; // matrica B
    Cbuff = new float[n]; // red matice C
/*
    printf("SLAVE%d B:", rank);
    for (int i = 0; i < n * k; i++)
      printf("%f ", B[i]);
    printf("\n");
*/
    // dok ima posla
    while (needed)
    {
      //printf("Slave %d ceka\n", rank);
      // cekaj poruku od mastera
      MPI_Recv(&row, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      if (status.MPI_TAG == END_TAG)
      {
       // printf("Primljen END\n");
       // nema vise posla
        needed = false;
        continue;
      }
      //MPI_Recv(&col, 1, MPI_INT, MASTER, COLUMN_INDEX_TAG, MPI_COMM_WORLD, 0);
      // prima se red matice A
      MPI_Recv(Abuff, k, MPI_FLOAT, MASTER, ROW_TAG, MPI_COMM_WORLD, 0);
      //printf("SLAVE%d Primljen Abuff(%d)", rank, row);
      //for (int i = 0; i < k; printf("%F ", Abuff[i++]))
        //;
      //printf("\n");

      //MPI_Recv(Bbuff, k, MPI_FLOAT, MASTER, COLUMN_TAG, MPI_COMM_WORLD, 0);
      /*
      printf("SLAVE%d Primljen bBUFF(%d)", rank, col);
      for (int i = 0; i < k; printf("%F ", Bbuff[i++]))
        ;
      printf("\n");
*/
      // izracunavanje
      for (int i = 0; i < n; i++)
        Cbuff[i] = 0;
      for (int j = 0; j < n; j++)
      {
        float c = 0.0f;
       // printf("Kolona B: ");

        for (int i = 0; i < k; i++)
        {
         // printf("%f ", Bbuff[n * i + j]);
          c += Abuff[i] * Bbuff[n * i + j];
          //printf("c+=%f * %f", Abuff[i], Bbuff[k * j + i]);
        }
       // printf("\n");
        // zbir na kraju pomnoziti sa alfa i beta
        Cbuff[j] = Cbuff[j] * beta + alpha * c;
      }
      //printf("SLAVE%d Cbuff:", rank);
     // for (int i = 0; i < n; i++)
        //printf("%f ", Cbuff[i]);

      // posalji rezultat -- prvo redni broj reda pa sam red matice C
      MPI_Send(&row, 1, MPI_INT, MASTER, ROW_INDEX_TAG, MPI_COMM_WORLD);
     // MPI_Send(&col, 1, MPI_INT, MASTER, COLUMN_INDEX_TAG, MPI_COMM_WORLD);
      MPI_Send(Cbuff, n, MPI_FLOAT, MASTER, RESULT_TAG, MPI_COMM_WORLD);

      //printf("SLAVE%d salje C(%d, %d)=%f\n", rank, row, col, Cbuff);
    }
    // slave dealocira svoje bafere
    delete Abuff;
    delete Bbuff;
    delete Cbuff;
  }

  // dealociranje tipova podataka
  MPI_Type_free(&Arow);
  MPI_Type_free(&Bcol);
  MPI_Type_free(&Crow);
}

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);

  int rank, size;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size == 1)
  {
    // mora postojati barem jedan slave proces
    printf("Inadequate number of processes\n");
    MPI_Abort(MPI_COMM_WORLD, -10);
  }

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

    //printf("Arow=%d, Acol=%d, Brow=%d, Bcol=%d\n", matArow, matAcol, matBrow, matBcol);

    // allocate space for C
    std::vector<float> matC(matArow * matBcol);
    std::vector<float> matD(matArow * matBcol);

    // Use standard sgemm interface
    timeSeq = MPI_Wtime();
    basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matC.front(), matArow);
    //printf("Finished seq!\n");
    timeSeq = MPI_Wtime() - timeSeq;

    // parallel sgemm interface
    timePar = MPI_Wtime();
    basicSgemm_par(rank, size, 'N', 'T', matArow, matBcol, matAcol, 1.0f, &matA.front(), matArow, &matBT.front(), matBcol, 0.0f, &matD.front(), matArow);
    //printf("Finished par!\n");
    timePar = MPI_Wtime() - timePar;

    writeColMajorMatrixFile(argv[3], matArow, matBcol, matD);

    std::function<bool(double, double)> comparator = [](double left, double right) {
      // Lambda function to compare 2 doubles with ACCURACY
      return fabs(left - right) < ACCURACY;
    };

/*    for (int i = 0; i < matArow * matBcol; i++)
      printf("(%f, %f)\n", matC[i], matD[i]);

    printf("\n");
*/    /*    printf("matrica D:");
    for (int i = 0; i < matArow * matBcol; printf("%f ", matD[i++]))
      ;
    printf("\n");
*/
    std::cerr << "********************DZ2Z2**********************" << std::endl;
    std::cerr << "Elapsed time - SEQ: " << timeSeq << "." << std::endl;
    std::cerr << "Elapsed time - PAR(" << size << "): " << timePar << "." << std::endl;
    std::cerr << (std::equal(matC.begin(), matC.end(), matD.begin(), comparator) ? "TEST PASSED" : "TEST FAILED") << std::endl;
    std::cerr << "***********************************************" << std::endl;
  }
  else
  {
    // i master i svi slave-ovi pozivaju ovu funkciju
    basicSgemm_par(rank, size, 'N', 'T', 0, 0, 0, 1.0f, 0, 0, 0, 0, 0.0f, 0, 0);
  }

  MPI_Finalize();

  return 0;
}
