/*! This program executes the following tasks:
1. open the data files (int16 format)
2. pre-process the data for future uses of lasso
*/

#include <iostream>
#include <fstream>
#include <gsl/gsl_vector.h>
#include "Spectro_Signal.h"
using namespace std;

int main()
{
  string filelist="filelist.txt";
  ifstream fid_in;

  fid_in.open(filelist.c_str(),ifstream::in);

  while(!fid_in.eof())
    {
      string filename;
      getline(fid_in,filename);
      SpectroSignal test(filename,200,(unsigned long)GSL_POSINF,0,0,0,filename,true,32767-8147);
      //SpectroSignal test(filename,100,1000,0,0,0,filename,true,32767-8147);
      cout << "Now processing file" << filename << "..." << flush;
      test.SetDC(8147);
      test.SetRSTThresh(-1000);
      test.SetSigma(10);
      test.ProcessData("replace");
      gsl_vector *parameters=test.GetParameters();
      test.ExtractPiledupEnergies(40);
      cout << " Done: number of chunks extracted = " << gsl_vector_get(parameters,3) << endl;
    }
  
  fid_in.close();
  return EXIT_SUCCESS;
} 
