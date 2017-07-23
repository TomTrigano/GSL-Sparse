#ifndef POST_PROCESSING
#define POST_PROCESSING

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>
#include <cmath>

/*!
\class Post_Processing
\brief C++ class for several methods of post-processing
Private members which are user defined:
\param beta the regressor obtained by a sparse regression method (gsl_vector)
\param regroupmethod the method used for regrouping the inner coefficients of each block (string, default = "sum")
\param samplepathmethod the method used to estimate the sample path of the Poisson processes (string, default="sigmathreshold")
*/
class Post_Processing{
private:
	gsl_vector *beta; // dictionary
	std::string regroupmethod; 
	std::string samplepathmethod;
public:
	//! Default constructor
	Post_Processing();
	//! Full manual constructor
	Post_Processing(gsl_vector *a, std::string b, std::string c);
	//! Performs the post-processing accordingly to the class members (noise-independent)
	gsl_vector* Performs_PP(unsigned long block_size);
	//! Performs the post-processing accordingly to the class members (noise-dependent)
	gsl_vector* Performs_PP(unsigned long block_size, double sigma);
	//! Estimates the arrival times values (in terms of sampling points index)
	gsl_vector* EstimateArrivalTimes(gsl_vector *Tn);
	//! Withdraw some arrival times too close (can be used as a pileup rejector)
	gsl_vector *ThinningArrivalTimes(gsl_vector *Tn, unsigned int min_distance);
	//! Default destruction
	~Post_Processing();
};

#endif
