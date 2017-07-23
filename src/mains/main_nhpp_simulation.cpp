/*! This program executes the following tasks:
  1. Computes the dictionary
  2. Generates a synthetic signal
  3. Computes an NHPP estimator
*/

#include <iostream>
#include <fstream>
#include "Spectro_Signal.h"
#include "Poisson_Process.h"
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

#define DEBUG_1 1

using namespace std;

// Global variable - I know, this is evil, but I did not find another alternative
double alpha;

// intensity used for the simulations
double intensity(double x)
{
  return 0.1*exp(-(x-30.)*(x-30.)/80) + alpha*exp(-(x-60.)*(x-60.)/30);
}

// subfunction for signal creation
gsl_vector *CreateSyntheticSignal(gsl_vector *Tn,double sigma, size_t size_signal, double min_x, double step, size_t seed)
{
  gsl_vector * result=gsl_vector_calloc(size_signal);
  size_t m,n;
  gsl_rng *rng;  // random number generator
  rng = gsl_rng_alloc (gsl_rng_rand48);     // pick random number generator
  gsl_rng_set (rng, seed);                  // set seed

  for(n=0;n<Tn->size;++n)
    {
      double Tval=gsl_vector_get(Tn,n);
      double param1=gsl_ran_flat(rng,0.6,0.8);
      double param2=gsl_ran_flat(rng,0.6,0.8);
      double energy=125+gsl_ran_gaussian(rng,10);

      for(m=0;m<size_signal;++m)
	{
	  double subdiv=min_x+(double)m*step;
	  double valm= subdiv-Tval > 0 ? gsl_ran_gamma_pdf(subdiv-Tval,param1,1./param2) : 0 ;
	  double currval=gsl_vector_get(result,m);
	  gsl_vector_set(result,m,currval+energy*valm);
	}
    }

  // Add the noise
  for(m=0;m<size_signal;++m)
    {
      double currval=gsl_vector_get(result,m);
      double noise = gsl_ran_gaussian(rng,sigma);
      gsl_vector_set(result,m,currval+noise);
    }

  gsl_rng_free(rng);
  return result;
}

int main(int argc, char *argv[])
{
  size_t n=0;

  if(argc != 3)
    {
      std::cout << std::endl << "ERROR: wrong number of arguments" << std::endl ;
      std::cout << "Usage: ./NHPP_Simulations Monte_Carlo_nb_exp alpha" << std::endl ;
      return EXIT_FAILURE;
    }
  
  // random number generator
  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  
  // Dictionary used for the simulation
  gsl_vector *shape_reset=gsl_vector_calloc(10);
  unsigned long nb_shapes=10;
  unsigned long nb_params=2;
  std::string typeshape="atet";
  gsl_matrix *params=gsl_matrix_calloc(nb_params,nb_shapes);
  unsigned long signal_size=100;
  unsigned long shape_length=21;
  unsigned long rangesave=1 ; //21; // used to set the maximal possible start for a shape
  Const_Dictionary D((unsigned long)0,shape_reset,nb_shapes,nb_params,typeshape,params,signal_size,shape_length,rangesave);
  D.SetParams(0.4,0.4,0.1);
  D.SetDictionary("max");

  // Get the number of experiments from the command line and the alpha parameter
  alpha=strtod(argv[2],NULL);
  size_t MC_runs = strtoul(argv[1],NULL,0); 

  // Starts the file for saving the results
  std::stringstream ss1,ss2,ss3,ss4,ss5,ss6,ss7,ss8;
  std::ofstream ofs_KEs, ofs_ISE, ofs_KEs_stota, ofs_ISE_stota,ofs_KEs_ideal, ofs_ISE_ideal, ofs_KEs_oracle, ofs_ISE_oracle;

  ss1 << "densities_LASSO_" << alpha << ".bin" ;
  ofs_KEs.open(ss1.str().c_str(),std::ofstream::binary);
  ss2 << "ISE_" << alpha << "_LASSO.bin" ;
  ofs_ISE.open(ss2.str().c_str(),std::ofstream::binary);
  ss3 << "densities_threshold" << alpha << ".bin" ;
  ofs_KEs_stota.open(ss3.str().c_str(),std::ofstream::binary);
  ss4 << "ISE_" << alpha << "_threshold.bin" ;
  ofs_ISE_stota.open(ss4.str().c_str(),std::ofstream::binary);
  ss5 << "densities_ideal_" << alpha << ".bin" ;
  ofs_KEs_ideal.open(ss5.str().c_str(),std::ofstream::binary);
  ss6 << "ISE_" << alpha << "_ideal.bin" ;
  ofs_ISE_ideal.open(ss6.str().c_str(),std::ofstream::binary);
  ss7 << "densities_oracle" << alpha << ".bin" ;
  ofs_KEs_oracle.open(ss7.str().c_str(),std::ofstream::binary);
  ss8 << "ISE_" << alpha << "_oracle.bin" ;
  ofs_ISE_oracle.open(ss8.str().c_str(),std::ofstream::binary);

  // parameters for the poisson process and the signal
  double Tmax=100;
  size_t nb_pts_max=10000;
  gsl_vector *storage=gsl_vector_calloc(10000);
  double DeltaT=1;
  double sigma_noise=0.1;
  double threshold=10;

  // parameters for the kernel density estimate
  double min_x=0;
  double max_x=99;
  double step=1;
  double bandwidth=3;

  // We save the true pdf in a vector
  gsl_vector *true_pdf=gsl_vector_calloc(100);
  for(n=0;n<true_pdf->size;++n)
    gsl_vector_set(true_pdf,n,intensity(min_x+n*step));

  // We now run the experiment

  for(n=0;n<MC_runs;++n)
    {
      gsl_vector *tmp=gsl_vector_calloc(true_pdf->size);
      size_t m=0;

      // Draws a sample path
      Poisson_Process NHPP(Tmax,nb_pts_max,storage,n);
      NHPP.SamplePath(&intensity,2*alpha);
      gsl_vector *ideal_sp=NHPP.GetSamplePath();


      // the rest of the experiment is done only if there is at least one point in the sample path
      if(gsl_vector_get(ideal_sp,0)!=-M_PI)
	{
	  ////////////////////////////////////////////////////////
	  // Computes the ideal estimate and the associated ISE //
	  ////////////////////////////////////////////////////////
	  Kernel_Estimator Kideal(ideal_sp,min_x,max_x,step,bandwidth,"gaussian",1,ideal_sp->size);
	  Kideal.ComputeKernelEstimate();
	  gsl_vector *density_id=Kideal.ReturnKernelEstimate();
	  gsl_vector_scale(density_id,(double)ideal_sp->size);
      
	  gsl_vector_memcpy(tmp,true_pdf);
	  gsl_vector_sub(tmp,density_id);
	  double ISE_id = gsl_blas_dnrm2(tmp);

	  /////////////////////////////////////////////////////////
	  // Computes the oracle estimate and the associated ISE //
	  /////////////////////////////////////////////////////////
	  size_t oracle_size=0;
	  gsl_vector *oracle_sp;
	  double tmpval=0;
	  for(m=0;m<ideal_sp->size;++m)
	    {
	      if(ceil(gsl_vector_get(ideal_sp,m)) / DeltaT > tmpval)
		{
		  tmpval = ceil(gsl_vector_get(ideal_sp,m)) / DeltaT;
		  ++oracle_size;
		}
	    }
	  oracle_sp=gsl_vector_calloc(oracle_size);
	  oracle_size=0;
	  tmpval=0;
	  for(m=0;m<ideal_sp->size;++m)
	    {
	      if(ceil(gsl_vector_get(ideal_sp,m)) / DeltaT > tmpval)
		{
		  tmpval = ceil(gsl_vector_get(ideal_sp,m)) / DeltaT;
		  gsl_vector_set(oracle_sp,oracle_size,tmpval);
		  ++oracle_size;
		}
	    }



	  Kernel_Estimator Koracle(oracle_sp,min_x,max_x,step,bandwidth,"gaussian",1,oracle_sp->size);
	  Koracle.ComputeKernelEstimate();
	  gsl_vector *density_oracle=Koracle.ReturnKernelEstimate();
	  gsl_vector_scale(density_oracle,(double)oracle_sp->size);
      
	  gsl_vector_memcpy(tmp,true_pdf);
	  gsl_vector_sub(tmp,density_oracle);
	  double ISE_oracle = gsl_blas_dnrm2(tmp);

	  /////////////////////////////////////////////////////
	  // We create a synthetic signal which will be used //
	  // for simple thresholding and LASSO methods       //
	  /////////////////////////////////////////////////////
	  gsl_vector *signal=CreateSyntheticSignal(ideal_sp,sigma_noise,signal_size,min_x,step,n);
#ifdef DEBUG_1
	  FILE *fid=fopen("test.dat","wb");
	  gsl_vector_fwrite(fid,signal);
	  fclose(fid);
#endif


	  //////////////////////////////////////////////////
	  // Simple thresholding to get the arrival times //
	  //////////////////////////////////////////////////
	  double t[signal->size];
	  int acc = 0;
	  double nbThresholdedValues=0;
	  for(m=0;m<signal_size;++m)
	    {
	      t[m]=0;
	      if((acc == 0) && (gsl_vector_get(signal,m) > threshold))
		{
		  acc=1;
		  t[m] = m+1;
		  ++nbThresholdedValues;
		}

	      if((acc == 1) && (gsl_vector_get(signal,m) < threshold))
		{
		  acc=0;
		}
	    }

	  
	  gsl_vector *thresh_sp=gsl_vector_calloc(nbThresholdedValues);
	  for(m=0;m<signal_size;++m)
	    {
	      size_t tmpind=0;
	      if(t[m]>0)
		{
		  gsl_vector_set(thresh_sp,tmpind,t[m]);
		  ++tmpind;
		}
	    }

	  Kernel_Estimator Kthreshold(thresh_sp,min_x,max_x,step,bandwidth,"gaussian",1,thresh_sp->size);
	  Kthreshold.ComputeKernelEstimate();
	  gsl_vector *density_threshold=Kthreshold.ReturnKernelEstimate();
	  gsl_vector_scale(density_threshold,(double)thresh_sp->size);
      
	  gsl_vector_memcpy(tmp,true_pdf);
	  gsl_vector_sub(tmp,density_threshold);
	  double ISE_threshold = gsl_blas_dnrm2(tmp);


	  ////////////////////////////////////////////
	  // Post-Processed LASSO for arrival times //
	  ////////////////////////////////////////////




	  /////////////////////////////
	  // Now we save the results //
	  /////////////////////////////
      
	  for(m=0;m<density_id->size;++m)
	    {
	      double val;

    	      val=gsl_vector_get(density_id,m);
	      ofs_KEs_ideal.write(reinterpret_cast<char*>(&val),sizeof(val));
	      val=gsl_vector_get(density_oracle,m);
	      ofs_KEs_oracle.write(reinterpret_cast<char*>(&val),sizeof(val));
	      val=gsl_vector_get(density_threshold,m);
	      ofs_KEs_stota.write(reinterpret_cast<char*>(&val),sizeof(val));
	    }

	  ofs_ISE_ideal.write(reinterpret_cast<char*>(&ISE_id),sizeof(ISE_id));
	  ofs_ISE_oracle.write(reinterpret_cast<char*>(&ISE_oracle),sizeof(ISE_oracle));
	  ofs_ISE_stota.write(reinterpret_cast<char*>(&ISE_threshold),sizeof(ISE_threshold));


	  gsl_vector_free(tmp);
	  gsl_vector_free(signal);
	  gsl_vector_free(thresh_sp);
	  gsl_vector_free(oracle_sp);
	  gsl_vector_free(density_id);
	  gsl_vector_free(density_oracle);
	  gsl_vector_free(density_threshold);
	}
      gsl_vector_free(ideal_sp);
    }



  gsl_vector_free(shape_reset);
  gsl_matrix_free(params);
  gsl_rng_free(r);
  ofs_KEs.close();
  ofs_ISE.close();
  ofs_KEs_stota.close();
  ofs_ISE_stota.close();
  ofs_KEs_ideal.close();
  ofs_ISE_ideal.close();
  ofs_KEs_oracle.close();
  ofs_ISE_oracle.close();

  
  return EXIT_SUCCESS;
} 
