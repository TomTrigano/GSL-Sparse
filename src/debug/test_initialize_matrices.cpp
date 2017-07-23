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
#include "LASSO_LARS_Solver.h"

int main(int argc, char *argv[])
{
	unsigned long nb_lines=10;
	unsigned long nb_cols=10;
	gsl_matrix *A=gsl_matrix_calloc(nb_lines,nb_cols);
	gsl_vector *y=gsl_vector_calloc(nb_lines);
	unsigned long N=nb_cols;
	gsl_vector *beta=gsl_vector_calloc(N);
	gsl_vector *beta_real=gsl_vector_calloc(N);
	std::string algType="LASSO";
	unsigned long maxIters=10; // maximal number of interations
	double lambdaStop=0; // when the Lagrange multiplier <= lambdaStop.
	double resStop=1; // L2 norm of the residual <= resStop. 
	int verbose=1; // 1 to print out detailed progress at each iteration, 0 for no output.
	
	const gsl_rng_type * T;
	gsl_rng * r;
	int k,i,j;
		
	for(i=0;i<nb_lines;++i)
		for(j=0;j<=i;++j)
			gsl_matrix_set(A,i,j,(pow(double(j+2),double(i)/double(10))));
			
	gsl_vector_set(beta_real, 2, 1.5);
	gsl_vector_set(beta_real, 7, 2);
	//gsl_vector_set(beta_real, 22, 0.8);
	gsl_blas_dgemv(CblasNoTrans,1,A,beta_real,0,y);


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
 
	// Test InitializeActiveSet
	LASSO_LARS_solver Ls(A,y,algType,maxIters,lambdaStop,resStop,1);
	std::cout << "Solver initialized..." << std::endl;
	Ls.InitializeActiveSet(0);
	Ls.DisplayPrivateMembers();
	
	gsl_vector_free(beta);
	gsl_vector_free(beta_real);
	gsl_matrix_free(A);
	gsl_vector_free(y);
	return EXIT_SUCCESS;
}
