#include "Kernel_estimator.h"
#include "Poisson_Process.h"
#include <iostream>
#include <string>
#include <gsl/gsl_vector.h>

double intensity(double x)
{
  return 0.2*exp(-(x-30.)*(x-30.)/80) + 0.8*exp(-(x-60.)*(x-60.)/50);
}

int main(int argc, char *argv[])
{	
  Poisson_Process NHPP;
  Poisson_Process HPP;
  size_t k=0;
  double h = 5;

  // Computes the HPP simulation
  HPP.SamplePath(0.1);
  gsl_vector *HPPSamplePath=HPP.GetSamplePath();
  std::cout << "Estimated intensity = " <<  ((double)HPPSamplePath->size) / gsl_vector_get(HPPSamplePath,HPPSamplePath->size-1) << std::endl;

  // Computes the NHPP simulation
  NHPP.SamplePath(&intensity, 1);
  gsl_vector *NHPPSamplePath=NHPP.GetSamplePath();

  // Display in the console
  std::cout << "HPP = " << std::endl << "[";
  for(k=0;k<HPPSamplePath-> size;++k)
    std::cout << gsl_vector_get(HPPSamplePath,k) << " " ;
  std::cout << "]^T" << std::endl << std:: endl ;

  std::cout << "NHPP = " << std::endl << "[";
  for(k=0;k<NHPPSamplePath-> size;++k)
    std::cout << gsl_vector_get(NHPPSamplePath,k) << " " ;
  std::cout << "]^T" << std::endl << std:: endl ;

  // CHecks the density
  {
    Kernel_Estimator K(NHPPSamplePath,0,100,1,h,"gaussian",1,NHPPSamplePath->size);
    K.ComputeKernelEstimate();
    gsl_vector *pdf=K.ReturnKernelEstimate();
    gsl_vector_scale(pdf,(double)NHPPSamplePath->size);

    FILE * f = fopen ("test.dat", "w");
    gsl_vector_fprintf (f, pdf, "%.5g");
    fclose (f);
  }

  gsl_vector_free(HPPSamplePath);
  gsl_vector_free(NHPPSamplePath);

}
