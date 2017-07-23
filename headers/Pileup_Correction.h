#ifndef PILEUP
#define PILEUP

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>
#include <cmath>

/*!
\class Pileup_Correction
\brief C++ class which includes several methods for pileup correction.
\param individuals the energies associated with individual photons (gsl_vector *)
\param max_ind the maximal number of individual photons expected (unsigned long, default = 500)
\param correction_method the method used for the pile-up correction (string, default="sparseregression")
\param scaling a scale parameter which can be used for a better resolution in the energy spectra in case of little data (double, default = 1)
\todo add the known methods for pileup correction (analytic, bayesian, sparse regression)
*/
class Pileup_Correction{
private:
	gsl_vector *individuals;
	unsigned long max_ind;
	std::string correction_method; 
	double scaling;
public:
	//! Default constructor
	Pileup_Correction();
	//! Full manual constructor
	Pileup_Correction(gsl_vector *a,unsigned long b, std::string c, double d);
	//! Partial manual constructor
	Pileup_Correction(unsigned long b);
	//! Default destruction
	~Pileup_Correction();
	//! Gets the positive elements in the vector individuals 
	gsl_vector* GetEnergies();
	//! Scales the elements in the vector individuals 
	void ScaleEnergies();
	//! Scales the elements in the vector individuals by a user-given scaling factor 
	void ScaleEnergies(double userscale);
	//! Computes the energies by the sparse regression method, given a dictionary (gsl_matrix *A), estimated arrival times (gsl_vector *times) and a regressor beta (gsl_vector *beta) obtained by any sparse regression algorithm (Trigano and Sepulcre, 2012)
	void SparseEnergies(gsl_matrix *A, gsl_vector *beta, gsl_vector *times);
};

#endif
