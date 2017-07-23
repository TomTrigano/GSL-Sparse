#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include "Kernel_estimator.h"

//#define DEBUG 1

// Constructors

Kernel_Estimator::Kernel_Estimator()
{
	data=gsl_vector_calloc(500);
	pdf=gsl_vector_calloc(101);
	min_x=0; 
	max_x=100; 
	x_step=1; 
	bandwidth=1;
	kernel_type="gaussian"; 
	scaling=1;
	nb_data = 500;
}

Kernel_Estimator::Kernel_Estimator(gsl_vector *a,double c,double d,double e, double f, std::string g, double h, size_t n): data(a), min_x(c), max_x(d), x_step(e), bandwidth(f), kernel_type(g), scaling(h), nb_data(n)
{
  pdf = gsl_vector_calloc((size_t)((d-c)/e)+1);
}

Kernel_Estimator::Kernel_Estimator(double c,double d,double e, double f, std::string g, double h, size_t n): min_x(c), max_x(d), x_step(e), bandwidth(f), kernel_type(g), scaling(h), nb_data(n)
{
  data=gsl_vector_calloc(1);
  pdf = gsl_vector_calloc((size_t)((d-c)/e)+1);
}

Kernel_Estimator::Kernel_Estimator(gsl_vector *a,gsl_vector *b,double c,double d,double e, double f, std::string g, double h, size_t n): data(a), pdf(b), min_x(c), max_x(d), x_step(e), bandwidth(f), kernel_type(g), scaling(h), nb_data(n)
{
}

Kernel_Estimator::Kernel_Estimator(gsl_vector *a): data(a)
{
	pdf=gsl_vector_calloc(101);
	min_x=0; 
	max_x=100; 
	x_step=1; 
	bandwidth=1;
	kernel_type="gaussian"; 
	scaling=1;
	nb_data = a->size;
}



// Destructor
Kernel_Estimator::~Kernel_Estimator()
{
  gsl_vector_free(pdf);
}

/* ///////////////////////////////////////////////////////////////////////
// Computes the kernel estimator */

void Kernel_Estimator::ComputeKernelEstimate()
{
  gsl_vector_set_zero(pdf);
  gsl_matrix *W=gsl_matrix_calloc(pdf->size,data->size);
  size_t m,n;

  gsl_vector *norm_cst=gsl_vector_calloc(data->size);
  gsl_vector_set_all(norm_cst,1./(bandwidth*((double)data->size)));

  for(m=0;m<W->size1;++m)
    {
      for(n=0;n<W->size2;++n)
	{
	  double valexp=-(min_x+m*x_step-gsl_vector_get(data,n))/(bandwidth*scaling);
	  if(kernel_type.compare("gaussian")==0)
	    gsl_matrix_set(W,m,n,exp(-valexp*valexp/2)/sqrt(2*M_PI));

	  if(kernel_type.compare("rectangular")==0)
	    gsl_matrix_set(W,m,n,abs(valexp)<0.5 ? 1. : 0.);

	}
    }

  gsl_blas_dgemv(CblasNoTrans,1.,W,norm_cst,0,pdf);

  gsl_matrix_free(W);
  gsl_vector_free(norm_cst);

}


//////////////////////////////////////////////////////////////////////////
gsl_vector* Kernel_Estimator::ReturnKernelEstimate()
{
	gsl_vector *result=gsl_vector_calloc(pdf->size);
	gsl_vector_memcpy(result,pdf);
	return result;
}

////////////////////////////////////////////////////////////////////////
void Kernel_Estimator::AddValue2Histogram(double value)
{
	unsigned long k,n;

	for(k=0;k<pdf->size;++k)
	{
	  double pdf_value = gsl_vector_get(pdf,k);

		if(kernel_type.compare("gaussian")==0)
		  pdf_value += gsl_ran_gaussian_pdf((value/scaling - (min_x+(double)k*x_step))/bandwidth,1);

		if(kernel_type.compare("rectangular")==0)
		  pdf_value += (value/scaling - (min_x+(double)k*x_step))/bandwidth < 0.5 ? 1 : 0;

		gsl_vector_set(pdf,k,pdf_value);

	}
	++nb_data;
}

void Kernel_Estimator::ScaleHistogram()
{
  gsl_vector_scale(pdf,1./((double)nb_data * bandwidth));
}
