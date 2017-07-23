#ifndef SPEECH_SIGNAL
#define SPEECH_SIGNAL

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>

/*!
\class TIMIT_Signal
\brief C++ class for basic operations on speech signals from the TIMIT database, and basic operations: DC removal, Pre-emphasis, splitting and windowing, reading and writing results into binary files, fft and so forth.
\param filename the filename (string)
\param window_size the window size used (unsigned long, default: 256)
\param overlapping the overlapping used (unsigned long, default: 128)
\param nb_frames the number of frames to analyze (unsigned long, default =1)
\param signal the signal to be analyzed (gsl_vector *)
\param frames a matrix which contains all the frames (gsl_matrix *)
\param current_frame a vector which contains the frame for analysis
\param savename the name used for saving the vector (e.g., if savename = foo, then the GSL_vector used will be foo1.bin, foo2.bin and so forth - default: "block")
\param binary_save specify if the data should be saved in text mode or in binary (architecture dependent, default = TRUE)
*/
class TIMIT_Signal{
private:
	std::string filename; // raw signal file to process
	unsigned long window_size; // the size of signal chunks to process afterwards
	unsigned long overlapping; // the number of chunks to be used (set to -1) if all the raw signal is used
	unsigned long nb_frames; //The number of frames to analyze
	gsl_vector *sound; // the vector in which the whole data will be saved
	gsl_matrix *frames; // the matrix in which we will stock all the frame
	gsl_vector *current_frame; // A vector which saves the current frame to work on it
	std::string savename; // the basic name for saving all the vectors obtained
	bool binary_save; // true to save the file in a binary file, false to a text file (default: false)
public:
	//! Default constructor
	TIMIT_Signal();
	//! Manual constructor
	TIMIT_Signal(std::string s, unsigned long a, unsigned long b, unsigned long bb,gsl_vector* c, gsl_matrix* d, gsl_vector* e, std::string f, bool g);
	//! Partially manual constructor
	TIMIT_Signal(std::string s, unsigned long a, unsigned long b, unsigned long bb, std::string f, bool g);
	//! Default destructor
	~TIMIT_Signal();
	//! Removes the DC from the input signal
	void RemoveDC();
	//! Removes the DC from the input signal with a user-defined value
	void RemoveDC(double DCvalue);
	//! High-pass FIR filtering given by the transfer function H(z) = 1- a z^{-1}
	void Preemphasis(double a);
	//! Multiplies the frames by a Hamming window
	void Hamming();
	//! Reads the WAV file and puts the frames in the relevent matrix if stock_matrix is TRUE - otherwise saves only the vector
	void ReadAll(bool stock_matrix);
	//! Reads the n-th frame in the current WAV file
	void ReadFrame(unsigned long n);

};

#endif
