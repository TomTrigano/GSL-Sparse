#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <sys/types.h>
#include "Poisson_Process.h"

//#define DEBUG 1

// Constructors

Poisson_Process::Poisson_Process()
{
  Tmax=100;
  nb_points_max=1000;
  storage=gsl_vector_calloc(1000);
  seed=0;
}

Poisson_Process::Poisson_Process(double a,size_t b,gsl_vector *c, size_t d): Tmax(a), nb_points_max(b), storage(c), seed(d)
{
}


// Destructor
Poisson_Process::~Poisson_Process()
{
}

/* ///////////////////////////////////////////////////////////////////////
// HPP sample path */

void Poisson_Process::SamplePath(double lambda)
{
  gsl_rng *rng;  // random number generator
  rng = gsl_rng_alloc (gsl_rng_rand48);     // pick random number generator
  gsl_rng_set (rng, seed);                  // set seed

  gsl_vector_set_zero(storage);

  size_t k=0;
  double acc=0;

  while((k<nb_points_max) && (acc<Tmax))
    {
      acc+= gsl_ran_exponential(rng,1. / lambda);
      gsl_vector_set(storage,k,acc);
      ++k;
    }

  gsl_rng_free (rng); 
}

//////////////////////////////////////////////////////////////////////////
void Poisson_Process::SamplePath(double (*intensity)(double),double UpperBound)
{
  gsl_rng *rng;  // random number generator
  rng = gsl_rng_alloc (gsl_rng_rand48);     // pick random number generator
  gsl_rng_set (rng, seed);                  // set seed

  gsl_vector_set_zero(storage);
  gsl_vector *workspace=gsl_vector_calloc(storage->size);

  size_t k=0;
  double acc=0;

  while((k<nb_points_max) && (acc<Tmax))
    {
      acc+= gsl_ran_exponential(rng,1. / UpperBound);
      gsl_vector_set(workspace,k,acc);
      ++k;
    }

  size_t m=0;
  size_t n=0;
  for(m=0;m<k;++m)
    {
      if((*intensity)(gsl_vector_get(workspace,m)) > UpperBound*gsl_ran_flat(rng,0,1))
	{
	  gsl_vector_set(storage,n,gsl_vector_get(workspace,m));
	  ++n;
	}
    }

  gsl_rng_free (rng); 
  gsl_vector_free(workspace);
}

////////////////////////////////////////////////////////////////////////
gsl_vector* Poisson_Process::GetSamplePath()
{
  unsigned long k=gsl_vector_max_index(storage);
  if(k>0)
    {
      gsl_vector *result = gsl_vector_calloc(k);
      size_t m=0;

      for(m=0;m<k;++m)
	gsl_vector_set(result,m,gsl_vector_get(storage,m));
      return result;
    }
  else
    {
      gsl_vector *result = gsl_vector_calloc(1);
      gsl_vector_set(result,0,-M_PI);
      return result;
    }

}
