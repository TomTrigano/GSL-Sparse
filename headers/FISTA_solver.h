#ifndef FISTA_SOLVER
#define FISTA_SOLVER

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

/*!
\class FISTA_solver
\brief C++ class for solving a sparse regression problem using proximal methods (FISTA)
Private members which are user defined:
\param A: the dictionary matrix (gsl_matrix, default: 10x10 identity)
\param y: the observed signal (gsl_vector, default: 0)
\param regType: the type of regression used used (string= lasso, elnet, glasso)
\param params: gsl_vector which includes the paraneters used for the regression (default: [0 0])
\param groups: gsl_vector which regroups the columns of A into groups, used for Group LASSO methods (default: 0)
\param maxIters: maximum number of iterations
\param verbose: 1 to print out detailed progress at each iteration, 0 for no output.
*/
class FISTA_solver{
private:
	gsl_matrix *A; // dictionary
	gsl_vector *y; // output signal
	std::string regType; // can be lasso, elnet, glasso
	gsl_vector *groups;
	unsigned long maxIters; // maximal number of interations
	int verbose; // 1 to print out detailed progress at each iteration, 0 for no output.
	gsl_vector *beta;
	double sigma0;
	double tol;
public:
	//! Default constructor
	FISTA_solver();
	//! Partial manual constructor
	FISTA_solver(gsl_matrix *a,gsl_vector *b, std::string c, gsl_vector *d,unsigned long f, int g);
	//! Full manual constructor
	FISTA_solver(gsl_matrix *a,gsl_vector *b, std::string c, gsl_vector *d,unsigned long f, int g, gsl_vector *i, double j, double k);
	//! Resets all the parameters used in the solver for further use.
	void ReinitializeValues();
	//! Returns the solution of the optimization problem
	void GetResult(gsl_vector *dest) const;
	//! Default destruction
	~FISTA_solver();
	//! Estimates sigma0 as the 2-norm (spectral radius) of A squared and divided by its nb of lines -- used for the gradient step
	void EstimateSigma0(double tolest);
	//! Solves the LASSO problem by means of the algorithm decribed in Beck et al. (2009)
	void Solve(double tau);
	//! Solves the Elastic Net problem by means of the algorithm decribed in Beck et al. (2009)
	void Solve(double tau, double smooth_par);
	//! Solves the Group LASSO problem by means of the algorithm decribed in Beck et al. (2009) 
	void GroupSolve(double tau);
};

#endif
