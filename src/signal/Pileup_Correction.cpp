#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>
#include <cmath>
#include "Pileup_Correction.h"

//#define DEBUG 1

// Constructors
Pileup_Correction::Pileup_Correction()
{
  individuals=gsl_vector_calloc(500);
  max_ind=500;
  correction_method="sparseregression";
  scaling=1;
}

Pileup_Correction::Pileup_Correction(gsl_vector *a, unsigned long b, std::string c, double d): individuals(a), max_ind(b), correction_method(c), scaling(d)
{
}

Pileup_Correction::Pileup_Correction(unsigned long b):  max_ind(b)
{
  individuals=gsl_vector_calloc(b);
  correction_method="sparseregression";
  scaling=1;
}

// Destructor
Pileup_Correction::~Pileup_Correction()
{
}

///////////////////////////////////////////////////////////////////////
gsl_vector* Pileup_Correction::GetEnergies()
{
  unsigned long n,size=1;
  gsl_vector *E;

  for(n=0;n<individuals->size;++n)
    if(gsl_vector_get(individuals,n)>0)
      ++size;

  E=gsl_vector_calloc(size);

  size=0;
  for(n=0;n<individuals->size;++n)
    if(gsl_vector_get(individuals,n)>0)
      {
	gsl_vector_set(E,size,gsl_vector_get(individuals,n));
	++size;
      }

  return E;

}

/////////////////////////////////////////////////////////////////////////

void Pileup_Correction::ScaleEnergies()
{
  gsl_vector_scale(individuals,scaling);
}

void Pileup_Correction::ScaleEnergies(double userscale)
{
  gsl_vector_scale(individuals,userscale);
}

/////////////////////////////////////////////////////////////////////////
void Pileup_Correction::SparseEnergies(gsl_matrix *A,gsl_vector *beta, gsl_vector *times)
{
  unsigned long n=0,k=0,m=0;
  // number of shapes
  unsigned long nb_shapes = (unsigned long)((A->size2) / (A->size1));
  // creates temporary vectors
  gsl_vector *sol=gsl_vector_calloc(A->size1);
  gsl_vector *ones=gsl_vector_calloc(A->size1);
  gsl_vector *betapart=gsl_vector_calloc(beta->size);

  // Reset the individuals vector
  gsl_vector_set_zero(individuals);
  gsl_vector_set_all(ones,1);

  for(n=0;n<times->size-1;++n)
    {
      unsigned long start_subv=((unsigned long)gsl_vector_get(times,n)-1)*nb_shapes;
      unsigned long end_subv=((unsigned long)gsl_vector_get(times,n+1)-1)*nb_shapes;
      double e=0;
      gsl_vector_set_zero(betapart);

      for(k=start_subv;k<end_subv;++k)
	gsl_vector_set(betapart,k,gsl_vector_get(beta,k));

      // Multiply A*beta_temp to get the energies of one photon
      gsl_blas_dgemv(CblasNoTrans,1,A,betapart,0,sol);

      // Sets the energy
      gsl_blas_ddot(sol,ones,&e);
      gsl_vector_set(individuals,m,e);
      ++m;
    }

  gsl_vector_free(sol);
  gsl_vector_free(betapart);
  gsl_vector_free(ones);
}
