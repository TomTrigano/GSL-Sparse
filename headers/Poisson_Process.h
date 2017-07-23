#ifndef POISSON
#define POISSON

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>
#include <cmath>

/*!
\class Poisson_Process
\brief C++ class to simulate the sample path of a Poisson process, homogeneous or non-homogeneous, on a given interval [0;Tmax]
\param Tmax the upper limit of the interval on which the sample path is computed (double, default = 100)
\param nb_points_max the maximum number of points to save in the sample path (size_t, default = 1000)
\param storage a vector on which the value of the points of the sample path will be stored
\param seed the seed for the random number generators
*/

class Poisson_Process{
private:
  double Tmax;
  size_t nb_points_max;
  gsl_vector *storage;
  size_t seed;
public:
	//! Default constructor
	Poisson_Process();
	//! Full manual constructor
	Poisson_Process(double a, size_t b,gsl_vector *c, size_t d);
	//! Default destruction
	~Poisson_Process();
	//! Simulates the sample path of a homogeneous Poisson process with intensity lambda
	void SamplePath(double lambda);
	//! Simulates the sample path of a non-homogeneous Poisson process with varying intensity lambda (acceptation-reject method)
	void SamplePath(double (*intensity)(double),double UpperBound);
	//! Gets the obtained sample path
	gsl_vector *GetSamplePath();
	};
	
	#endif
