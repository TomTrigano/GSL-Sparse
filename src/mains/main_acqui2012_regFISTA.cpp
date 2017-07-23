/*! This program executes the following tasks:
1. Computes the dictionary
*/

#include <iostream>
#include <fstream>
#include "Spectro_Signal.h"
#include "Const_Dictionary.h"
#include "FISTA_solver.h"
#include "Post_Processing.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <cstdio>
#include <cstdlib>

//#define SHOW_RESULTS 1
#define SHOW_GRAPHS 1
//#define SHOW_DIC 1

using namespace std;

int main()
{
  size_t m,n;  
  gsl_vector *shape_reset=gsl_vector_calloc(10);
  unsigned long nb_shapes=10;
  unsigned long nb_params=2;
  std::string typeshape="gamma";
  gsl_matrix *params=gsl_matrix_calloc(nb_params,nb_shapes);
  unsigned long signal_size=500;
  unsigned long shape_length=21;
  unsigned long rangesave=20; // used to set the maximal possible start for a shape
  
  Const_Dictionary D((unsigned long)0,shape_reset,nb_shapes,nb_params,typeshape,params,signal_size,shape_length,rangesave);
  //D.SetParams(0.1,0.4,0.1);
  D.SetRealisticParams(0.4,0.1);
  D.SetDictionary("max");

  {
    //    string filename="./americium241/signal_1_1334915434_605408.values.bin";
    //    string sizename="./americium241/signal_1_1334915434_605408.sizes.bin";
    string filename="./pidie7-1e6/signal_2_1334916701_087627.values.bin";
    string sizename="./pidie7-1e6/signal_2_1334916701_087627.sizes.bin";

    unsigned long nb_trials=100;
    unsigned long trace_length=0,total_number=0;

    ifstream sig(filename.c_str(),ifstream::binary);
    ifstream taille(sizename.c_str(),ifstream::binary);

    for(n=0;n<nb_trials;++n)
      {
	unsigned long sizey=0;
	gsl_matrix *A;
	gsl_vector *y;

	taille.read( reinterpret_cast< char* >( &sizey ), sizeof(unsigned long));

	y=gsl_vector_calloc(sizey);

	for(m=0;m<y->size;++m)
	  {
	    double val=0;
	    sig.read( reinterpret_cast< char* >( &val ), sizeof(double));
	    gsl_vector_set(y,m,val);
	  }
	
	if(sizey==signal_size)
	  A=D.GetDictionary();
	else
	  A=D.GetSubdictionary_blocks(sizey);

#ifdef SHOW_DIC
	std::cout << A->size1 << std::endl << A->size2 << std::endl ;
	for(m=0;m<A->size1;++m)
	  {
	    for(n=0;n<A->size2;++n)
	      std::cout << gsl_matrix_get(A,m,n) << std::endl;
	    //	    std::cout << std::endl;
	  }
#endif

	gsl_vector *result=gsl_vector_calloc(A->size2);
	gsl_vector *beta=gsl_vector_calloc(A->size2);
	gsl_vector *groups=gsl_vector_calloc(A->size2);

        FISTA_solver Fs(A,y,"lasso",groups,10000000,1,beta,59,0.000001);
#ifdef SHOW_RESULTS
	std::cout << "Solver initialized..." << std::endl;
#endif
	Fs.Solve(1.5);
	Fs.GetResult(result);
#ifdef SHOW_RESULTS
	std::cout << "beta = [";
	for(m=0;m<result->size;++m)
	  std::cout << gsl_vector_get(result,m) << " " ;
	std::cout << "]^T" << std::endl << std::endl ;
#endif
	
	Post_Processing PP(result,"sum","sigmathreshold");
	gsl_vector *Cn=PP.Performs_PP(nb_shapes,60);
#ifdef SHOW_RESULTS
	std::cout << "Cn = [";
	for(m=0;m<Cn->size;++m)
	  std::cout << gsl_vector_get(Cn,m) << " " ;
	std::cout << "]^T" << std::endl << std::endl ;
#endif
	
	gsl_vector *Tn=PP.EstimateArrivalTimes(Cn);
#ifdef SHOW_RESULTS
	std::cout << "Tn = [";
	for(m=0;m<Tn->size;++m)
	  std::cout << gsl_vector_get(Tn,m) << " " ;
	std::cout << "]^T" << std::endl << std::endl ;
#endif

	if(n!=nb_trials-1)
	  {
	    trace_length+=sizey;
	    total_number+=gsl_vector_get(Tn,0);
	  }
	else
	  {
	    trace_length += gsl_vector_get(Tn,Tn->size-1);
	    total_number+=gsl_vector_get(Tn,0);
	  }

#ifdef SHOW_GRAPHS
	{
	  gsl_vector *estimated_signal=gsl_vector_calloc(y->size);
	  gsl_blas_dgemv(CblasNoTrans,1.,A,beta,0,estimated_signal);
	  FILE *fid=fopen("tempsig.txt","w");
	  FILE *fid2=fopen("tempTn.txt","w");
	  FILE *fid3=fopen("esty.txt","w");
	  FILE *fid4=fopen("tempbeta.txt","w");
	  FILE *fid5=fopen("tempopenco.txt","w");
	  gsl_vector_fprintf(fid,y,"%.6f");
	  gsl_vector_fprintf(fid2,Tn,"%.6f");
	  gsl_vector_fprintf(fid3,estimated_signal,"%.6f");
	  gsl_vector_fprintf(fid4,beta,"%.6f");
	  gsl_vector_fprintf(fid5,Cn,"%.6f");
	  fclose(fid);
	  fclose(fid2);
	  fclose(fid3);
	  fclose(fid4);
	  fclose(fid5);
	  system("octave < script_octave.m");
	  system("rm tempsig.txt");
	  system("rm tempTn.txt");
	  system("rm esty.txt");
	  system("rm tempbeta.txt");
	  system("rm tempopenco.txt");
	  gsl_vector_free(estimated_signal);
	}
#endif	
	gsl_vector_free(result);
	gsl_vector_free(beta);
	gsl_vector_free(y);
	gsl_matrix_free(A);
	gsl_vector_free(Tn);
	gsl_vector_free(Cn);
#ifdef SHOW_GRAPHS
	std::cout << (double)total_number / (double)trace_length << std::endl;
#endif
	//	std::cout << "." << std::flush;
    }
    std::cout << std::endl;
    std::cout << (double)total_number / (double)trace_length;
  }

  
  gsl_vector_free(shape_reset);
  gsl_matrix_free(params);
  
  return EXIT_SUCCESS;
} 
