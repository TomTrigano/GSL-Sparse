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
	gsl_matrix * R;
	gsl_matrix * A, *B, *C;
	gsl_vector * aset;
	int i,j;
	
	// essai de cholesky decomp
	A=gsl_matrix_calloc(10,10);
	B=gsl_matrix_calloc(10,10);
	C=gsl_matrix_calloc(10,10);
	aset=gsl_vector_calloc(10);
	gsl_matrix_set_identity(A);
	
	for(i=0;i<A->size1;++i)
		for(j=0;j<i;++j)
			gsl_matrix_set(A,i,j,1);
	
	gsl_matrix_transpose_memcpy(B,A);
	gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,A,B,0,C);
	
	
	for(i=0;i<aset->size;++i)
		gsl_vector_set(aset,i,10-i);

	std::cout << "aset = " << std::endl;
	for(i=0;i<aset->size;++i)
		std::cout << "[ " << gsl_vector_get(aset,i) << "]" << std::endl;

		
	std::cout << "C = " << std::endl;
	for(i=0;i<C->size1;++i)
	{
		std::cout << "[ ";
		for(j=0;j<C->size2;++j)
			std::cout << gsl_matrix_get(C,i,j) << " " ;
		std::cout << "]" << std::endl;
	}

	std::cout << "Coin";
	gsl_linalg_cholesky_decomp(C);
	gsl_linalg_cholesky_svx(C,aset);
	std::cout << "Coin" << std::endl;

	
	std::cout << "LL^T = " << std::endl;
	for(i=0;i<C->size1;++i)
	{
		std::cout << "[ ";
		for(j=0;j<C->size2;++j)
			std::cout << gsl_matrix_get(C,i,j) << " " ;
		std::cout << "]" << std::endl;
	}

	std::cout << "aset = " << std::endl;
	for(i=0;i<aset->size;++i)
		std::cout << "[ " << gsl_vector_get(aset,i) << "]" << std::endl;
	
	for(i=0;i<A->size1;++i)
		for(j=0;j<i;++j)
			gsl_matrix_set(A,i,j,0);

	std::cout << "L^T = " << std::endl;
	for(i=0;i<B->size1;++i)
	{
		std::cout.precision(2);
		std::cout << "[ ";
		for(j=0;j<B->size2;++j)
			std::cout << gsl_matrix_get(B,i,j) << " " ;
		std::cout << "]" << std::endl;
	}
	
	for(i=0;i<aset->size;++i)
		gsl_vector_set(aset,i,10-i);

	
	gsl_blas_dtrsv(CblasUpper,CblasNoTrans,CblasNonUnit,B,aset);

	std::cout << "aset = " << std::endl;
	for(i=0;i<aset->size;++i)
		std::cout << "[ " << gsl_vector_get(aset,i) << "]" << std::endl;
	
	
	gsl_matrix_free(A);
	gsl_matrix_free(B);
	gsl_matrix_free(C);
	gsl_vector_free(aset);
	
	return 0;
}
