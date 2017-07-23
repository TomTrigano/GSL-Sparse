/*! This program executes the following tasks:
1. open the data files (int16 format)
2. pre-process the data for future uses of lasso
*/

#include <iostream>
#include <fstream>
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
      SpectroSignal test(filename,500,(unsigned long)GSL_POSINF,0,0,0,filename,true);
      cout << "Now processing file" << filename << "..." << flush;
      test.SetDC(450);
      test.SetRSTThresh(-50);
      test.SetSigma(4);
      test.ProcessData("cut");
      test.ExtractPiledupEnergies(10);
      cout << " Done." << endl;
    }
  
  fid_in.close();
  return EXIT_SUCCESS;
} 
