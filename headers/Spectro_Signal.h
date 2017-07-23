#ifndef SPECTRO_SIGNAL
#define SPECTRO_SIGNAL

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sort.h>
//#include <gsl/gsl_statistics.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

/*!
\class SpectroSignal
\brief C++ class for basic operations on spectrometric data such as: DC removal, reset removal, cutting to signal adapted to a dictionary, reading and writing results into binary files
\param filename: the filename (string)
\param chunk_size: the maximal size of vectors to be used for sparse reconstruction afterwards (default: 500)
\param nb_chunks: the number of vectors to obtain (integer; if taken to GSL_INF, then the whole raw data is analyzed -- default: GSL_INF)
\param DC: the DC voltage added to the signal; can be estimated by the median or set up manually (default: 0)
\param sigma: the standard deviation of the additive noise, assumed to be Gaussian (default: 10)
\param RST_lower_threshold: the lower threshold used to localized the reset undershoots; can be estimated of set up manually (default: 0)
\param savename: the name used for saving the vector (e.g., if savename = foo, then the GSL_vector used will be foo1.bin, foo2.bin and so forth - default: "block")
\param saturation_value: value used for the detection of saturating peaks 
\param nb_effective_chunks The actual number of chunks that have been processed in the end
*/
class SpectroSignal{
private:
	std::string filename; // raw signal file to process
	unsigned long chunk_size; // the size of signal chunks to process afterwards
	unsigned long nb_chunks; // the number of chunks to be used (set to -1) if all the raw signal is used
	double DC ; // The DC value, set to 0 initially
	double sigma;
	double RST_lower_threshold; // the lower threshold for the reset
	std::string savename; // the basic name for saving all the vectors obtained
	bool binary_save; // true to save the file in a binary file, false to a text file (default: false)
	double saturation_value; // value used for saturating peaks
	unsigned long nb_effective_chunks; 
public:
	//! Default constructor
	SpectroSignal();
	//! Manual constructor
	SpectroSignal(std::string s, unsigned long a, unsigned long b, double c, double d, double e, std::string f, bool g, double h);
	//! Default destruction
	~SpectroSignal();
	//! Uses the median as the DC (due to the fact that the signal is sparse, so the DC estimate can be pretty robust by taking the median)
	void ComputeMedianDC(long piece_size);
	//! Sets the DC value manually
	void SetDC(double value);
	//! Approximate the value of a lower Threshold for the reset by checking the possible negative values obtained on a large chunk
	void ComputeRSTThreshold(long piece_size, double quantile);
	//! Sets the lower threshold manually
	void SetRSTThresh(double value);
	
	//! Approximation of the standard noise deviation
	void ComputeSigma(unsigned long piece_size,double minval,double maxval, unsigned long nb_bins,double normalize);
	//! Set the std of the noise manual
	void SetSigma(double value);
	//! Gets the main parameters (DC, sigma, RSTThresh, nb_effective_chunks) out of the class
	gsl_vector *GetParameters() const;
	//! Performs the preparation of the signal to be later on analyzed (cutno = "cut", the bad parts of the signal are discarded, cutno="replace", the bad parts of the signal are replaced by noise)
	void ProcessData(std::string cutno);
	//! Puts the energies obtained by thresholding (w/o pileup correction) in a file
	void ExtractPiledupEnergies(double thresh);
};

#endif
