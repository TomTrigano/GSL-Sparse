#ifndef CONST_DICTIONARY
#define CONST_DICTIONARY

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
//#include <gsl/gsl_statistics.h>
//#include <gsl/gsl_statistics_double.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstdio>

/*!
\class Const_Dictionary
\brief C++ class for basic operations on dictionaries used to process spectrometric signales such as: creating dictionaries; selecting subdictionaries; evaluating pulses energies on some local part of the signal; updating the dictionary
\param filename: the filename (string)
\param shape_reset: the known detector reset shape whose occurrences are to be removed 
\param typeshape: in spectrometry we use dictionary of gamma shapes 
\param shape_length: all basic shapes have a common limited length, though the shapes are theoretically infinite 
\param rangesave:  
*/

class Const_Dictionary{
private:
  unsigned long IncludeShapeReset;
  gsl_vector *shape_reset; // the known detector reset shape
  unsigned long nb_shapes;
  unsigned long nb_params;
  std::string typeshape; // the type of the shapes: exponential or gamma
  gsl_matrix *params; //=gsl_matrix_calloc(nb_params,nb_shapes); // row vector collecting the parameters 
  unsigned long signal_size; // the size of the original signal
  unsigned long shape_length; // the (finite) length of each dictionary shape ; corresponds to $length(t)$ ine the Matlab code Create_DictionaryNHPP
  unsigned long rangesave; // used to set the maximal possible start for a shape
  gsl_matrix *shapes;
  gsl_matrix *BigA; // The generated dictionary - Maximal size so that we can only get smaller dictionaries
public:
  //! Default constructor
  Const_Dictionary();
  //! Manual constructor
  Const_Dictionary(unsigned long a, gsl_vector *b, unsigned long c, unsigned long d, std::string e, gsl_matrix *f, unsigned long g, unsigned long h, unsigned long i, gsl_matrix *j,gsl_matrix *k); // 
  //! Partial Manual constructor
  Const_Dictionary(unsigned long a, gsl_vector *b, unsigned long c, unsigned long d, std::string e, gsl_matrix *f, unsigned long g, unsigned long h, unsigned long i); // 
  //! Default destructor
  ~Const_Dictionary();
  //! Entering parameters of the basis signals
  void SetParams(double param_min1, double param_min2 , double param_increm);
  void SetParams(double param_min1, double param_increm);
  void SetRealisticParams(double param_min1, double param_increm,double decay);
  void ShowParams();
  //! Initializes the shapes
  void SetShapes(std::string normalizationtype);
  //! Creating the dictionary
  void SetDictionary(std::string normalizationtype);
  //! Returns a copy of the dictionary or of a subdictionary accordingly to some columns only
  gsl_matrix *GetSubdictionary_cols(gsl_vector *bool_col);
  //! Returns a copy of the dictionary or of a subdictionary accordingly to specified blocks
  gsl_matrix *GetSubdictionary_blocks(unsigned long nb_blocks);
  gsl_matrix *GetSubdictionary_blocks(gsl_vector *bool_block);
  //! Gets the whole dictionary
  gsl_matrix *GetDictionary();
  //! Get a dictionary reweighted by the weight matrix W
  gsl_matrix *GetReweightedDictionary(gsl_matrix *W, int dimension);
};
#endif
