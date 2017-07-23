#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <iomanip>
#include <string>
#include "Post_Processing.h"


int main(int argc, char *argv[])
{
  unsigned long n=0;
  gsl_vector* beta=gsl_vector_calloc(100);
  gsl_vector* beta2=gsl_vector_calloc(100);
  unsigned long nb_shapes=10;

  for(n=0;n<beta->size;n++)
    {
      gsl_vector_set(beta,n,n);
      gsl_vector_set(beta2,n,(n%2 == 0 ? (double)n: -(double)n));
    }

  // Test Post_Processing
  {
    Post_Processing pp1(beta,"L1","tentimesthreshold");
    gsl_vector *Tn=pp1.Performs_PP(nb_shapes,100);
    std::cout << " estimated Tn (sum,tentimesthreshold) for beta = [";
    for(n=0;n<Tn->size;++n)
      std::cout << std::setprecision(16) << gsl_vector_get(Tn,n) << " " ;
    std::cout << "]^T" << std::endl << std::endl ;
    
    gsl_vector_free(Tn);
  }

  {
    Post_Processing pp2(beta2,"L1","tentimesthreshold");
    gsl_vector* Tn=pp2.Performs_PP(nb_shapes,100);
    std::cout << " estimated Tn (sum,tentimesthreshold) for beta = [";
    for(n=0;n<Tn->size;++n)
      std::cout << std::setprecision(16) << gsl_vector_get(Tn,n) << " " ;
    std::cout << "]^T" << std::endl << std::endl ;
    
    gsl_vector_free(Tn);
  }
  
  
  gsl_vector_free(beta);
  gsl_vector_free(beta2);
	
  return EXIT_SUCCESS;
}
