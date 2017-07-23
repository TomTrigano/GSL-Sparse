#ifndef LASSO_LARS_SOLVER
#define LASSO_LARS_SOLVER

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
#include <cmath>

/*!
\class LASSO_LARS_solver
\brief C++ class for solving a sparse regression problem using either LARS or LASSO, possibly with nonnegativity constraints on the covariance
Private members which are user defined:
\param A: the dictionary matrix (gsl_matrix, default: 10x10 identity)
\param y: the observed signal (gsl_vector, default: 0)
\param algType: the algorithm used (string= lasso, lars, nnlasso, nnlars)
\param maxIters: maximum number of iterations
\param lambdaStop: stops the solver when the Lagrange multiplier <= lambdaStop (default: 0)
\param resStop: stops the solver when the L2 norm of the residual <= resStop (default: 0)
\param verbose: 1 to print out detailed progress at each iteration, 0 for no output.
Private members automatically initialized
\param beta: solution of the LASSO
\param Aactive: the dictionary matrix (gsl_matrix, default: 10x10 null)
\param Aactive_size: the number of active vectors
\param newIndices,removeIndices,activeSet,inactiveSet,collinearSet: vector with 0's and 1's
\param R: cholesky matrix with the active columns
\param Rsize: size of the matrix, basically number of active columns of A
\param residual: vector which contains y-A.beta, initially y
\param corr: vector which contains the correlations
\param lambda: sparsity parameter
\param OptTol: error tolerence (default: 1e-5)
*/
class LASSO_LARS_solver{
private:
	gsl_matrix *A; // dictionary
	gsl_vector *y; // output signal
	std::string algType; // can be lars, lasso, nnlars, nnlasso
	unsigned long maxIters; // maximal number of interations
	double lambdaStop; // when the Lagrange multiplier <= lambdaStop.
	double resStop; // L2 norm of the residual <= resStop. 
	int verbose; // 1 to print out detailed progress at each iteration, 0 for no output.
	gsl_matrix *Aactive;
	long Aactive_size;
	gsl_vector *newIndices;
	gsl_vector *removeIndices;
	gsl_vector *activeSet;
	gsl_vector *inactiveSet;
	gsl_vector *collinearSet;
	gsl_matrix *R;
	unsigned long Rsize;
	gsl_vector *residual;
	gsl_vector *corr;
	double lambda;
	gsl_vector *beta; // solution of the LASSO
	double OptTol; // error tolerance
public:
	//! Default constructor
	LASSO_LARS_solver();
	//! Manual constructor
	LASSO_LARS_solver(gsl_matrix *a,gsl_vector *b, std::string c, unsigned long d, double e, double f, int g);
	//! Default destruction
	~LASSO_LARS_solver();
	//! Solves the LASSO by means of the algorithm decribed in Efron et al. (2004, Annals of Statistics)
	void SolveLASSO();
	// Subfunctions called by SolveLASSO
	void updateChol(unsigned long newIndex, int &flag);
	void downdateChol(unsigned long j);
	int InitializeActiveSet(int nonNegative);
	int CheckStoppingCriterion(double gammamin);
	void InitializeCholeskyFactor(unsigned long &iter);
	void ComputeLARSDirection(gsl_vector *dx,gsl_vector* ATv,gsl_vector *v);
	void FindFirstVector2Activate(double &gammaIc,int nonNegative, gsl_vector *ATv);
	void AugmentActiveSet(unsigned long &iter);
	void ReduceActiveSet(unsigned long &iter);
	void DisplayVector(gsl_vector *v);
	void DisplayPrivateMembers(std::string input);
	void SetLambdaMaxCorr(int nntest);
	void GetRegressor(gsl_vector *dest);
	void ResetRegressor();
};

#endif
