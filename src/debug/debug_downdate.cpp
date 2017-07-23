#include <iostream>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>

void downdateChol(unsigned long j,gsl_matrix *R, size_t &Rsize);

int main()
{
  gsl_matrix *R=gsl_matrix_calloc(10,10);
  size_t Rsize=3;
  size_t n=0,m=0;

  for(n=0;n<Rsize;++n)
    gsl_matrix_set(R,n,n,n+1);

  gsl_matrix_set(R,0,1,1);
  gsl_matrix_set(R,0,2,1);
  gsl_matrix_set(R,1,2,1);

  std::cout << "R=" << Rsize << std::endl;
  for(n=0;n<Rsize;++n)
    {
      for(m=0;m<Rsize;++m)
	std::cout << gsl_matrix_get(R,n,m) << " " ;
      std::cout << std::endl;
    }
  std::cout << std::endl << std::endl;

  downdateChol(1,R,Rsize);

  std::cout << "R=" << Rsize << std::endl;

  for(n=0;n<Rsize;++n)
    {
      for(m=0;m<Rsize;++m)
	std::cout << gsl_matrix_get(R,n,m) << " " ;
      std::cout << std::endl;
    }
  std::cout << std::endl << std::endl;


  gsl_matrix_free(R);
}

void downdateChol(unsigned long j,gsl_matrix *R, size_t &Rsize)
{
	gsl_matrix *Rtemp=gsl_matrix_calloc(Rsize,Rsize-1);
	gsl_vector *temp=gsl_vector_calloc(R->size1);

	unsigned long k;
	unsigned long n=Rsize-1;
	
	for(k=j+1;k<R->size2;k++)
	{
		gsl_matrix_get_col(temp,R,k);
		gsl_matrix_set_col(R,k-1,temp);

	}
	
	// Givens rotations to cancel the violating nonzeros
	
	for(k=j;k<n;++k)
	{
		double r=gsl_hypot(gsl_matrix_get(R,k,k),gsl_matrix_get(R,k+1,k));
		double c=gsl_matrix_get(R,k,k) / r;
		double s=-gsl_matrix_get(R,k+1,k)/r;
		
		double t[4]={c,-s,s,c};
		
		gsl_matrix_set(R,k,k,r);
		gsl_matrix_set(R,k+1,k,0);
		
		if(k<n)
		{
			gsl_matrix_view G = gsl_matrix_view_array(t,2,2);
			gsl_matrix_view subR=gsl_matrix_submatrix(R,k,k+1,2,n-k);
			gsl_matrix *subRcpy=gsl_matrix_calloc(2,n-k);
			
			gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,1.0, &G.matrix, &subR.matrix, 0.0, subRcpy);
			gsl_matrix_memcpy(&subR.matrix,subRcpy);
			
			gsl_matrix_free(subRcpy);
		}
	}
	
	for(k=0;k<R->size2;++k)
		gsl_matrix_set(R,n,k,0);
	
	gsl_matrix_free(Rtemp);
	gsl_vector_free(temp);

	--Rsize;
}
