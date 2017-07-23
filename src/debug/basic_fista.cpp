#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <iomanip>
#include <string>
#include "FISTA_solver.h"


int main(int argc, char *argv[])
{
	unsigned long nb_lines=10;
	unsigned long nb_cols=10;
	gsl_matrix *A=gsl_matrix_calloc(nb_lines,nb_cols);
	gsl_vector *y=gsl_vector_calloc(nb_lines);
	unsigned long N=nb_cols;
	gsl_vector *groups=gsl_vector_calloc(N);
	gsl_vector *beta_real=gsl_vector_calloc(N);
	gsl_vector *beta_est=gsl_vector_calloc(N);
	unsigned long maxIters=10000; // maximal number of interations
	
	const gsl_rng_type * T;
	gsl_rng * r;
	unsigned long k,i,j;
		
	for(i=0;i<nb_lines;++i)
		for(j=0;j<=i;++j)
			gsl_matrix_set(A,i,j,(pow(double(j+2),double(i)/double(10))));
			
	gsl_vector_set(beta_real, 2, 1.5);
	gsl_vector_set(beta_real, 7, 2);
	//gsl_vector_set(beta_real, 22, 0.8);
	gsl_blas_dgemv(CblasNoTrans,1,A,beta_real,0,y);

	gsl_vector_set_all(groups,1);
	gsl_vector_set(groups,5,2);
	gsl_vector_set(groups,6,2);
	gsl_vector_set(groups,7,2);
	gsl_vector_set(groups,8,3);
	gsl_vector_set(groups,9,3);


	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	
	// for(k=0;k<y->size;++k)
	// {
		// double tmp=gsl_ran_gaussian (r,0.05);
		// double tmp2=gsl_vector_get(y,k);
		// gsl_vector_set(y,k,tmp+tmp2);
	// }
	
	std::cout << "y = [";
	for(k=0;k<y->size;++k)
		std::cout << " " << gsl_vector_get(y,k) << " ";
	std::cout << "]" << std::endl << std::endl;
 
	// Test LASSO-FISTA
	{
	  FISTA_solver FLs(A,y,"lasso",groups,maxIters,1);
	  std::cout << "Solver initialized..." << std::endl;
	  FLs.Solve(0.01);
	  FLs.GetResult(beta_est);
	  std::cout << " estimated beta 0.01 = [";
	  for(k=0;k<beta_est->size;++k)
	    std::cout << std::setprecision(16) << gsl_vector_get(beta_est,k) << " " ;
	  std::cout << "]^T" << std::endl << std::endl ;
	  FLs.ReinitializeValues();

	  FLs.Solve(1);
	  FLs.GetResult(beta_est);
	  std::cout << " estimated beta 1 = [";
	  for(k=0;k<beta_est->size;++k)
	    std::cout << std::setprecision(16) << gsl_vector_get(beta_est,k) << " " ;
	  std::cout << "]^T" << std::endl << std::endl ;
	  FLs.ReinitializeValues();

	  FLs.Solve(1,1);
	  FLs.GetResult(beta_est);
	  std::cout << " estimated beta 1,1 = [";
	  for(k=0;k<beta_est->size;++k)
	    std::cout << std::setprecision(16) << gsl_vector_get(beta_est,k) << " " ;
	  std::cout << "]^T" << std::endl << std::endl ;
	  FLs.ReinitializeValues();

	  FLs.Solve(0.0001,1);
	  FLs.GetResult(beta_est);
	  std::cout << " estimated beta 0.0001 = [";
	  for(k=0;k<beta_est->size;++k)
	    std::cout << std::setprecision(16) << gsl_vector_get(beta_est,k) << " " ;
	  std::cout << "]^T" << std::endl << std::endl ;
	  FLs.ReinitializeValues();

	  FLs.GroupSolve(10);
	  FLs.GetResult(beta_est);
	  std::cout << " estimated beta 0.1 = [";
	  for(k=0;k<beta_est->size;++k)
	    std::cout << std::setprecision(16) << gsl_vector_get(beta_est,k) << " " ;
	  std::cout << "]^T" << std::endl << std::endl ;
	  FLs.ReinitializeValues();


	  std::cout << " true beta = [";
	  for(k=0;k<beta_real->size;++k)
	    std::cout << gsl_vector_get(beta_real,k) << " " ;
	  std::cout << "]^T" << std::endl << std::endl ;
	}


	gsl_vector_free(beta_real);
	gsl_vector_free(beta_est);
	gsl_matrix_free(A);
	gsl_vector_free(y);

	return EXIT_SUCCESS;
}
