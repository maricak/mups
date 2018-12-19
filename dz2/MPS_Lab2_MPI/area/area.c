// area.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
struct compl {
  double real;
  double imag;
};
int main(){
  int i, j, iter, numoutside = 0;
  int npoints, maxiter;
  double area, error, ztemp;
  struct compl z, c;
  // Should be done by the master only
  printf("Number of points? "); scanf("%d", &npoints);
  printf("Maximum number of iterations? ");  scanf("%d", &maxiter);

  // Calculate & broadcast number of points for every process
  // Broadcast maximum number of iterations

/*  Outer loops run over npoints, initialise z=c
 *  Inner loop has the iteration z=z*z+c, and threshold test
 */
  for (i=0; i<npoints; i++) {
    for (j=0; j<npoints; j++) {
      c.real = -2.0+2.5*(double)(i)/(double)(npoints)+1.0e-7;
      c.imag = 1.125*(double)(j)/(double)(npoints)+1.0e-7;
      z=c;
      for (iter=0; iter<maxiter; iter++){
	ztemp=(z.real*z.real)-(z.imag*z.imag)+c.real;
	z.imag=z.real*z.imag*2+c.imag; 
	z.real=ztemp; 
	if ((z.real*z.real+z.imag*z.imag)>4.0e0) {
	  numoutside++; 
	  break;
	}
      }
    }
  }

  // Reduce numoutside variable

/*
 *  Calculate area and error and output the results in the MASTER process
 */

      area=2.0*2.5*1.125*((double)(npoints*npoints-numoutside)/(double)(npoints*npoints));
      error=area/(double)npoints;

      printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
}

