/*! This program executes the following tasks:
  1. Computes the dictionary
  2. Loads the processed data from the previous program
  3. Makes a iterative pileup correction as in Ilia's project (with a possible one-pass)
  NUMERICAL CONDITIONS TO CHANGE ACCORDINGLY AT LINES: 46,82,192,198,200,202
  TEXT CONDITION: 60,61,62,63
*/

#include <iostream>
#include <fstream>
#include "Spectro_Signal.h"
#include "Const_Dictionary.h"
#include "LASSO_LARS_solver.h"
#include "Post_Processing.h"
#include "Kernel_estimator.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <cstdio>
#include <cstdlib>

//#define SHOW_RESULTS 1
//#define SHOW_GRAPHS 1
//#define DEBUG_MAIN 1
#define DEBUG_LONG_TIME 1

using namespace std;

int main(int argc, char *argv[])
{

  if(argc != 4)
    {
      std::cout << std::endl << "ERROR: wrong number of arguments" << std::endl ;
      std::cout << "Usage: ./Pileup_Correction_ADONIS file.values.bin file.sizes.bin number_of_chunks_to_process" << std::endl ;
      return EXIT_FAILURE;
    }
  size_t m,n;
  
  //gsl_vector *pdf_corrected=gsl_vector_calloc(3701);
  //Kernel_Estimator K(pdf_corrected,pdf_corrected,0,3700,1,1,"gaussian",100,0);
  Kernel_Estimator K(0,3700,1,1,"gaussian",100,0);
  gsl_vector *shape_reset=gsl_vector_calloc(10);
  unsigned long nb_shapes=10;
  unsigned long nb_params=2;
  std::string typeshape="atet";
  gsl_matrix *params=gsl_matrix_calloc(nb_params,nb_shapes);
  unsigned long signal_size=200;
  unsigned long shape_length=21;
  unsigned long rangesave=1 ; //21; // used to set the maximal possible start for a shape
  std::stringstream ss1;
  std::ofstream ofs;
  
  const gsl_rng_type * T;
  gsl_rng * r;

#ifdef DEBUG_MAIN
  std::cout << "Creates the dictionary..." << std::endl ;
#endif
  
  Const_Dictionary D((unsigned long)0,shape_reset,nb_shapes,nb_params,typeshape,params,signal_size,shape_length,rangesave);
  D.SetParams(0.4,0.4,0.1);
  //D.SetRealisticParams(0.4,0.1,0.63);
  D.SetDictionary("max");

#ifdef DEBUG_MAIN
  D.ShowParams();
#endif

  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

#ifdef DEBUG_MAIN
  std::cout << "Dictionary created: It works (checked with Matlab)" << std::endl ;
#endif

  {

    // Change this if needed
    string filename=argv[1];
    string sizename=argv[2];
    ss1 << "depiled_energies_" << filename ;
    ofs.open(ss1.str().c_str(),std::ofstream::binary);
 
    unsigned long nb_trials=strtoul(argv[3],NULL,0); 
    unsigned long trace_length=0,total_number=0;

    ifstream sig(filename.c_str(),ifstream::binary);
    ifstream taille(sizename.c_str(),ifstream::binary);

#ifdef DEBUG_MAIN
    std::cout << "Starting the loop..." << std::endl;
#endif

    for(n=0;n<nb_trials;++n)
      {
	unsigned long sizey=0;
	gsl_matrix *A;
	gsl_vector *y;

#ifdef DEBUG_LONG_TIME
	std::cout << "Now processing chunk " << n+1 << " / " << nb_trials <<  std::endl;
#endif	
	taille.read( reinterpret_cast< char* >( &sizey ), sizeof(unsigned long));
	
	y=gsl_vector_calloc(sizey);
	gsl_vector *pulse_approx=gsl_vector_calloc(y->size);
		
	for(m=0;m<y->size;++m)
	  {
	    double val=0;
	    sig.read( reinterpret_cast< char* >( &val ), sizeof(double));
	    gsl_vector_set(y,m,val);
	  }

#ifdef SHOW_RESULTS
	{
	  FILE *fid=fopen("signal_chunk.dat","wb");
	  gsl_vector_fwrite(fid,y);
	  fclose(fid);
	}
#endif
	  
	if(gsl_vector_max(y)>40 && gsl_vector_max(y)<23000)
	  {
	    if(sizey==signal_size)
	      A=D.GetDictionary();
	    else
	      A=D.GetSubdictionary_blocks(sizey);

#ifdef SHOW_RESULTS
	    {
	      FILE *fid=fopen("dico_test.dat","wb");
	      gsl_matrix_fwrite(fid,A);
	      fclose(fid);
	    }
#endif

	    gsl_vector *beta=gsl_vector_calloc(A->size2);	
	    unsigned long maxIters=10000; // maximal number of interations
	    double lambdaStop=0; // when the Lagrange multiplier <= lambdaStop.
	    double resStop_times=0;double resStop_energies=0; // L2 norm of the residual <= resStop. 

	    for(m=0;m<y->size;++m)
	      {
		resStop_times += pow(gsl_ran_gaussian(r,4),2);
		resStop_energies += pow(gsl_ran_gaussian(r,4),2);
	      }
	    resStop_times=1500;//1000*sqrt(resStop_times);
	    resStop_energies=400;//30*sqrt(resStop_energies);


	    LASSO_LARS_solver Ls_times(A,y,"NNLASSO",maxIters,lambdaStop,resStop_times,0);
#ifdef SHOW_RESULTS
	    std::cout << "Solver initialized..." << std::endl;
#endif
	    Ls_times.SolveLASSO();
	    Ls_times.GetRegressor(beta);
#ifdef SHOW_RESULTS
	    std::cout << "beta = [";
	    for(m=0;m<beta->size;++m)
	      if(gsl_vector_get(beta,m) != 0)
		std::cout << m << " " << gsl_vector_get(beta,m) << std::endl ;
	    std::cout << "]^T" << std::endl << std::endl ;
	    {

	      FILE *fid=fopen("beta_test.dat","wb");
	      gsl_vector_fwrite(fid,beta);
	      fclose(fid);
	    }
#endif
	
	    Post_Processing PP(beta,"L1","sigmathreshold");
	    gsl_vector *Cn=PP.Performs_PP(nb_shapes,4);

	
	    gsl_vector *Tn=PP.EstimateArrivalTimes(Cn);
#ifdef SHOW_RESULTS
	    std::cout << "Tn = [";
	    for(m=0;m<Tn->size;++m)
	      std::cout << gsl_vector_get(Tn,m) << " " ;
	    std::cout << "]^T" << std::endl << std::endl ;
	    std::cout << "Cn = [";
	    for(m=0;m<Cn->size;++m)
	      std::cout << gsl_vector_get(Cn,m) << " " ;
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
	      gsl_vector_fprintf(fid,y,"%.6f");
	      gsl_vector_fprintf(fid2,Tn,"%.6f");
	      gsl_vector_fprintf(fid3,estimated_signal,"%.6f");
	      gsl_vector_fprintf(fid4,beta,"%.6f");
	      fclose(fid);
	      fclose(fid2);
	      fclose(fid3);
	      fclose(fid4);
	      system("octave < script_octave.m");
	      system("rm tempsig.txt");
	      system("rm tempTn.txt");
	      system("rm esty.txt");
	      gsl_vector_free(estimated_signal);
	    }
#endif	


	    // At this stage, we have the arrival times of each individual pulses, we rerun LASSO to derive each energy
	    // index m for a reweighted matrix
	    // Note TOM: I try to make a single run instead of several runs, by putting all the weights inside a matrix
	    gsl_matrix *W=gsl_matrix_calloc(A->size2,A->size2);
	    gsl_matrix *Acopy=gsl_matrix_calloc(A->size1,A->size2);		
		
	    // if you want to reject pile-ups, comment the following and uncomment the line below it
	    gsl_vector* Tn_copy=gsl_vector_calloc(Tn->size);gsl_vector_memcpy(Tn_copy,Tn);
	    // gsl_vector *Tn_copy=PP.ThinningArrivalTimes(Tn,5);

#ifdef DEBUG_MAIN
	    std::cout << "Iterate on arrival times... " << std::endl ;
#endif		
	    for(m=1;m<=gsl_vector_get(Tn_copy,0);++m)
	      {
		double sig_weight=-1;
		double Tval=gsl_vector_get(Tn_copy,m);
#ifdef DEBUG_MAIN
		std::cout << "Tval = " << Tval << " " ;
#endif	
		if(m==1) //CONTINUER ICI
		  {
		    sig_weight = (double)nb_shapes*(1+gsl_vector_get(Tn_copy,1))/8.0;
		    if(gsl_vector_get(Tn_copy,0)>1)
		      {
			sig_weight=GSL_MIN_DBL(nb_shapes*(gsl_vector_get(Tn_copy,2)-gsl_vector_get(Tn_copy,1))/8, nb_shapes*(1+gsl_vector_get(Tn_copy,1))/8);
			if(gsl_vector_get(Tn_copy,1)==0)
			  sig_weight = (gsl_vector_get(Tn_copy,2)-1)*nb_shapes / 8;
		      }
		  }
		else
		  {
		    if(m==gsl_vector_get(Tn_copy,0))
		      sig_weight=GSL_MIN_DBL(nb_shapes*(y->size-(gsl_vector_get(Tn_copy,m)+1))/8, nb_shapes*(gsl_vector_get(Tn_copy,m)-gsl_vector_get(Tn_copy,m-1))/8);
		    else
		      sig_weight=GSL_MIN_DBL(nb_shapes*(gsl_vector_get(Tn_copy,m+1)-gsl_vector_get(Tn_copy,m))/8, nb_shapes*(gsl_vector_get(Tn_copy,m)-gsl_vector_get(Tn_copy,m-1))/8);
		  }

#ifdef SHOW_RESULTS
		std::cout << "sig_weight = " << sig_weight << " , Tn = "<< Tval << std::endl ;
#endif	

		size_t z=0;
		double maxtime=0,findmax=0;
		if(m==gsl_vector_get(Tn_copy,0))
		  {
		    for(z=(size_t)Tval*nb_shapes;z<beta->size;++z)
		      {
			if(gsl_vector_get(beta,z) > findmax)
			  {
			    maxtime = z-(size_t)Tval*nb_shapes;
			    findmax=gsl_vector_get(beta,z);
			  }
		      }
		  }
		else
		  {
		    size_t end_search=Tval;
		    while(gsl_vector_get(Cn,end_search)!=0)
		      ++end_search;

		    for(z=(size_t)Tval*nb_shapes;z<end_search*nb_shapes;++z)
		      {
			if(gsl_vector_get(beta,z) > findmax)
			  {
			    maxtime = z-(size_t)Tval*nb_shapes;
			    findmax=gsl_vector_get(beta,z);
			  }
		      }
		  }


		double acc=0;

#ifdef DEBUG_MAIN
		std::cout << "Variance of the weight matrix computed... " ;
#endif	

		for(z=0;z<W->size2;++z)
		  {
		    double prev_diag=gsl_matrix_get(W,z,z);
		    double val_kernel=exp(-pow((double)z-(double)Tval*(double)nb_shapes-maxtime-1,2) / (2*sig_weight*sig_weight));
					
		    gsl_matrix_set(W,z,z,prev_diag+val_kernel);
		    acc+=gsl_matrix_get(W,z,z);
		  }
				
		for(z=0;z<W->size2;++z)
		  gsl_matrix_set(W,z,z,gsl_matrix_get(W,z,z)*gsl_vector_get(Tn_copy,0)/acc);

#ifdef SHOW_RESULTS
		std::cout << "active_coef = " << maxtime << std::endl;
		std::cout << "weight diag = [";
		for(z=0;z<W->size2;++z)
		  if(gsl_matrix_get(W,z,z) != 0)
		    std::cout << z << " " << gsl_matrix_get(W,z,z) << std::endl ;
		std::cout << "]" << std::endl;

		{
		  FILE *fid=fopen("weight_matrix.dat","wb");
		  gsl_matrix_fwrite(fid,W);
		  fclose(fid);
		}
#endif


		gsl_matrix_set_zero(Acopy);
		gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, A, W, 0, Acopy);



#ifdef SHOW_RESULTS
		{
		  FILE *fid=fopen("weighted_dico.dat","wb");
		  gsl_matrix_fwrite(fid,Acopy);
		  fclose(fid);
		}
#endif
		LASSO_LARS_solver Ls_energies(Acopy,y,"NNLASSO",maxIters,lambdaStop,resStop_energies,0);
		// LASSO for the energies


		Ls_energies.SolveLASSO();
		gsl_vector_set_zero(beta);
		Ls_energies.GetRegressor(beta);

#ifdef DEBUG_MAIN
		std::cout << "Energy computed..." << std::endl;
#endif


#ifdef SHOW_RESULTS
		std::cout << "beta_energies = [";
		for(z=0;z<beta->size;++z)
		  if(gsl_vector_get(beta,z)!=0)
		    std::cout << z << " " << gsl_vector_get(beta,z) << std::endl ;
		std::cout << "]^T" << std::endl << std::endl ;

		{
		  FILE *fid=fopen("beta_energies.dat","wb");
		  gsl_vector_fwrite(fid,beta);
		  fclose(fid);
		}
#endif
			
		// Compute the energy and the residual
			
			
		gsl_blas_dgemv(CblasNoTrans,1,Acopy,beta,0,pulse_approx);
		double energy=gsl_blas_dasum(pulse_approx);
		gsl_vector_sub(y,pulse_approx);
			
		// saves the energy into a binary file
		ofs.write( reinterpret_cast<char*>( &energy ), sizeof(energy));
		// updates the pdf estimate
		K.AddValue2Histogram(energy);

#ifdef DEBUG_MAIN
		std::cout.precision(15);
		std::cout << "Energy value = " << energy << std::endl;
#endif
	
			
			
	      }
		

	    gsl_vector_free(pulse_approx);
	    gsl_vector_free(beta);
	    gsl_vector_free(y);
	    gsl_vector_free(Tn_copy);
	    gsl_matrix_free(A);
	    gsl_matrix_free(Acopy);
	    gsl_matrix_free(W);
	    gsl_vector_free(Tn);
	    gsl_vector_free(Cn);
#ifdef SHOW_RESULTS
	    std::cout << (double)total_number / (double)trace_length << std::endl;
#endif
	  }
      }
    std::cout << std::endl;
    std::cout << (double)total_number / (double)trace_length;
  }

  K.ScaleHistogram();
  gsl_vector *result = K.ReturnKernelEstimate();

  {
    FILE *fid=fopen("depiled_spectrum.dat","wb");
    gsl_vector_fwrite(fid,result);
    fclose(fid);
  }


  gsl_vector_free(result);  
  gsl_vector_free(shape_reset);
  gsl_matrix_free(params);
  ofs.close();
  
  return EXIT_SUCCESS;
} 
