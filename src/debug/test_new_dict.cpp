#include "Spectro_Signal.h"
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{	
  gsl_vector *a;
  SpectroSignal test("~/projets/adonis/acqui_avril_2012/pidie5e5_1_1334916199_084037",500,(unsigned long)GSL_POSINF,0,0,0,"block_2_1334916229_091777",true);
  std::cout << "Classe test creee :)" << std::endl;
  test.SetDC(450);
  test.SetRSTThresh(-50);
  test.ComputeSigma(100000,-500,2500,30,1);
  test.SetSigma(4);
  a=test.GetParameters();
  std::cout << "Parameters updated..." << std::endl;
  std::cout << "DC=" << gsl_vector_get(a,0) << std::endl;
  std::cout << "Sigma=" <<  gsl_vector_get(a,2) << std::endl;
  std::cout << "RST thresh=" <<  gsl_vector_get(a,1) << std::endl;
  test.ProcessData();
  std::cout << "Data prepared for further analysis..." << std::endl;
  test.ExtractPiledupEnergies(10);
  std::cout << "Energies Created" << std::endl;
  std::cout << "Fin du test !" << std::endl;
  gsl_vector_free(a);
  return EXIT_SUCCESS;
}
