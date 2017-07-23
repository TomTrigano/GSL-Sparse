#include <iostream>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>

double EstimateSigma0(gsl_matrix *A,double tolest);

int main()
{

  gsl_matrix *R=gsl_matrix_calloc(3,5);
  size_t Rsize=3;
  size_t n=0,m=0;
  double normest=0.;

  for(n=0;n<Rsize;++n)
    gsl_matrix_set(R,n,n,n+1);
  
  gsl_matrix_set(R,0,1,1);
  gsl_matrix_set(R,0,2,1);
  gsl_matrix_set(R,1,2,1);
  gsl_matrix_set(R,0,3,1);
  gsl_matrix_set(R,0,4,1);
  gsl_matrix_set(R,1,3,1);
  gsl_matrix_set(R,1,4,1);
  gsl_matrix_set(R,2,3,1);
  gsl_matrix_set(R,2,4,1);
  
  std::cout << "R=" << Rsize << std::endl;
  for(n=0;n<Rsize;++n)
    {
      for(m=0;m<Rsize;++m)
	std::cout << gsl_matrix_get(R,n,m) << " " ;
      std::cout << std::endl;
    }
  std::cout << std::endl << std::endl;

  normest=EstimateSigma0(R,1e-6);
  std::cout << "sigma0 = " << normest << std::endl;

 gsl_matrix_free(R);
}


double EstimateSigma0(gsl_matrix *A, double tolest)
{
  gsl_vector *Ax = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size2);
  size_t m=0,n=0;  

  double e=0., cnt=0., e0 =0., tmp=0. ;

  for(m=0;m<A->size2;m++)
    {
      for(n=0;n<A->size1;++n)
	{
	  tmp = 0.;
	  tmp += fabs(gsl_matrix_get(A,n,m));
	}
      gsl_vector_set(x,m,tmp);      
    }
  
  e = gsl_blas_dnrm2(x);
  gsl_vector_scale(x,1./e);
  
  while ( (fabs(e-e0)) > (tolest * e))
    {
      e0 = e;
      // Sx = S*x
      gsl_blas_dgemv(CblasNoTrans, 1., A, x, 0., Ax);
      // e = norm(Sx)
      e = gsl_blas_dnrm2(Ax);
      // x = S'*Sx
      gsl_blas_dgemv(CblasTrans, 1., A, Ax, 0., x);

      double tmp=gsl_blas_dnrm2(x);      
      gsl_vector_scale(x,(double)1./tmp);
      
      cnt+=1;
    }
  
  gsl_vector_free(x);  
  gsl_vector_free(Ax);
  return e*e / (double)A->size1;
}
