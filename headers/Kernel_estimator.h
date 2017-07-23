#ifndef KERNEL_ESTIMATOR
#define KERNEL_ESTIMATOR

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>
#include <cmath>

/*!
\class Kernel_estimator
\brief C++ class to estimate the energy (probabilistic) density.
\param data the energies associated with individual photons (gsl_vector *)
\param pdf the values of the estimated pdf (gsl_vector *)
\param min_x the minimal energy of individual photons expected (double, default = 0)
\param max_x the maximal energy of individual photons expected (double, default = 500)
\param x_step the step between two bins (double, default = 1)
\param bandwidth the standard bandwidth parameter of kernel estimates (double, default = 1)
\param kernel_type the type of kernel (string, default="gaussian" ; for possible choices: gaussian, rectangular)
\param scaling a scale parameter which can be used for a better resolution in the energy spectra in case of little data (double, default = 1)
\param nb_data the number of used data (useful if we need to build the kernel estimate from extern files, or one value at a time)
\todo implement other kernels than gauss
\todo implement a kernel estimate based of for loops (right now, uses matrix multiplication and needs lots of RAM)
*/

class Kernel_Estimator{
private:
	gsl_vector *data;
	gsl_vector *pdf;
	double min_x; 
	double max_x; 
	double x_step; 
	double bandwidth;
	std::string kernel_type; 
	double scaling;
	size_t nb_data;
public:
	//! Default constructor
	Kernel_Estimator();
	//! Full manual constructors
	Kernel_Estimator(gsl_vector *a,double c,double d,double e, double f, std::string g, double h, size_t n);
	Kernel_Estimator(double c,double d,double e, double f, std::string g, double h, size_t n);
	Kernel_Estimator(gsl_vector *a,gsl_vector *b,double c,double d,double e, double f, std::string g, double h, size_t n);
	//! Partial manual constructor
	Kernel_Estimator(gsl_vector *a);
	//! Default destruction
	~Kernel_Estimator();
	//! Computes the pdf
	void ComputeKernelEstimate();
	//! Returns a copy of the pdf
	gsl_vector* ReturnKernelEstimate();
	//! Adds a single observation and updates the unnormalized kernel estimate accordingly
	void AddValue2Histogram(double value);
	//! Scales an histogram (unnormalized histogram) by 1/ nh, where n is the number of data and h the bandwidth
	void ScaleHistogram();
	};
	
	#endif
