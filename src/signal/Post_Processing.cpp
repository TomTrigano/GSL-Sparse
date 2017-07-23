#include <iostream>
#include <ostream>
#include <iomanip>
#include <string>
#include <cmath>
#include "Post_Processing.h"

//#define DEBUG 1

// Constructors
Post_Processing::Post_Processing()
{
  beta=gsl_vector_calloc(10);
  regroupmethod="sum";
  samplepathmethod="sigmathreshold";
}

Post_Processing::Post_Processing(gsl_vector *a, std::string b, std::string c): beta(a), regroupmethod(b), samplepathmethod(c)
{
}

// Destructor
Post_Processing::~Post_Processing()
{
}

// Estimates the sample path with a method independent of the noise processing
gsl_vector* Post_Processing::Performs_PP(unsigned long block_size)
{
  gsl_vector* Tn=gsl_vector_calloc(1);

  if(beta->size % block_size)
    std::cout << "ERROR: " << beta->size << " must be a multiple of " << block_size << std::endl;
  else
    {
      gsl_vector_free(Tn);
      Tn=gsl_vector_calloc((unsigned long)(double(beta->size)/double(block_size)));
      unsigned long k=0;

      // First pass: regroup the coefficients of each block
      if(regroupmethod.compare("sum")==0)
	for(k=0;k<beta->size;++k)
	  {
#ifdef DEBUG
	    std::cout << "value:" << (double)k/ (double)block_size << std::endl;
#endif
	    double temp=gsl_vector_get(Tn,(unsigned long)((double)k / (double)block_size));
	    temp+=gsl_vector_get(beta,k);
	    gsl_vector_set(Tn,(unsigned long)(double(k)/double(block_size)),temp);
	  }

     if(regroupmethod.compare("mean")==0)
	for(k=0;k<beta->size;++k)
	  {
	    double temp=gsl_vector_get(Tn,(unsigned long)(double(k)/double(block_size)));
	    temp+=gsl_vector_get(beta,k)/double(block_size);
	    gsl_vector_set(Tn,(unsigned long)(double(k)/double(block_size)),temp);
	  }

     if(regroupmethod.compare("hard")==0)
       {
	 gsl_vector *temp=gsl_vector_calloc(beta->size);
	 for(k=0;k<temp->size;++k)
	   gsl_vector_set(temp,k,fabs(gsl_vector_get(beta,k)));

	 for(k=0;k<Tn->size;++k)
	   {
	     gsl_vector_view v=gsl_vector_subvector(temp,k*block_size,block_size);
	     gsl_vector_set(Tn,k,gsl_vector_max(&v.vector));
	   }

	 gsl_vector_free(temp);
       }

     if(regroupmethod.compare("L1")==0)
       for(k=0;k<Tn->size;++k)
	 {
	   gsl_vector_view v=gsl_vector_subvector(beta,k*block_size,block_size);
	   gsl_vector_set(Tn,k,gsl_blas_dasum(&v.vector));
	 }

#ifdef DEBUG
     std::cout << " estimated Tn after first pass = [";
     for(k=0;k<Tn->size;++k)
      std::cout << std::setprecision(16) << gsl_vector_get(Tn,k) << " " ;
    std::cout << "]^T" << std::endl << std::endl ;
#endif


      // Second pass: keeps only the most representative coefficients
     if(samplepathmethod.compare("tentimesthreshold")==0)
       {
	 double maxval=gsl_vector_max(Tn);
	 for(k=0;k<Tn->size;++k)
	   gsl_vector_set(Tn,k,(gsl_vector_get(Tn,k)>0.1*maxval ? 1 : 0));
       }

     if(samplepathmethod.compare("blockselection")==0)
	 for(k=0;k<Tn->size;++k)
	   gsl_vector_set(Tn,k,(gsl_vector_get(Tn,k)>0 ? 1 : 0));

    }

  return Tn;
}



// Estimates the sample path with a method independent of the noise processing
gsl_vector* Post_Processing::Performs_PP(unsigned long block_size,double sigma)
{
  gsl_vector* Tn=gsl_vector_calloc(1);

  if(beta->size % block_size)
    std::cout << "ERROR: " << beta->size << " must be a multiple of " << block_size << std::endl;
  else
    {
      gsl_vector_free(Tn);
      Tn=gsl_vector_calloc((unsigned long)(double(beta->size)/double(block_size)));
      unsigned long k=0;

      // First pass: regroup the coefficients of each block
      if(regroupmethod.compare("sum")==0)
	for(k=0;k<beta->size;++k)
	  {
	    double temp=gsl_vector_get(Tn,(unsigned long)(double(k)/double(block_size)));
	    temp+=gsl_vector_get(beta,k);
	    gsl_vector_set(Tn,(unsigned long)(double(k)/double(block_size)),temp);
	  }

     if(regroupmethod.compare("mean")==0)
	for(k=0;k<beta->size;++k)
	  {
	    double temp=gsl_vector_get(Tn,(unsigned long)(double(k)/double(block_size)));
	    temp+=gsl_vector_get(beta,k)/double(block_size);
	    gsl_vector_set(Tn,(unsigned long)(double(k)/double(block_size)),temp);
	  }

     if(regroupmethod.compare("hard")==0)
       {
	 gsl_vector *temp=gsl_vector_calloc(beta->size);
	 for(k=0;k<temp->size;++k)
	   gsl_vector_set(temp,k,fabs(gsl_vector_get(beta,k)));

	 for(k=0;k<Tn->size;++k)
	   {
	     gsl_vector_view v=gsl_vector_subvector(temp,k*block_size,block_size);
	     gsl_vector_set(Tn,k,gsl_vector_max(&v.vector));
	   }

	 gsl_vector_free(temp);
       }

     if(regroupmethod.compare("L1")==0)
       for(k=0;k<Tn->size;++k)
	 {
	   gsl_vector_view v=gsl_vector_subvector(beta,k*block_size,block_size);
	   gsl_vector_set(Tn,k,gsl_blas_dasum(&v.vector));
	 }


#ifdef DEBUG
     std::cout << " estimated Tn after first pass = [";
     for(k=0;k<Tn->size;++k)
      std::cout << std::setprecision(16) << gsl_vector_get(Tn,k) << " " ;
    std::cout << "]^T" << std::endl << std::endl ;
#endif


      // Second pass: keeps only the most representative coefficients
     if(samplepathmethod.compare("tentimesthreshold")==0)
       {
	 double maxval=gsl_vector_max(Tn);
	 for(k=0;k<Tn->size;++k)
	   gsl_vector_set(Tn,k,(gsl_vector_get(Tn,k)>0.1*maxval ? 1 : 0));
       }

     if(samplepathmethod.compare("blockselection")==0)
	 for(k=0;k<Tn->size;++k)
	   gsl_vector_set(Tn,k,(gsl_vector_get(Tn,k)>0 ? 1 : 0));

     if(samplepathmethod.compare("sigmathreshold")==0)
	 for(k=0;k<Tn->size;++k)
	   gsl_vector_set(Tn,k,(gsl_vector_get(Tn,k)>3*sigma ? 1 : 0));

    }

  return Tn;
}

gsl_vector* Post_Processing::EstimateArrivalTimes(gsl_vector *Tn)
{
  size_t n=0,place=1,nb_arr=0;

  for(n=1;n<Tn->size;++n)
    if(gsl_vector_get(Tn,n)-gsl_vector_get(Tn,n-1) == 1)
      nb_arr+=1;

  if(gsl_vector_get(Tn,0)==1)
	nb_arr++;

  gsl_vector *Arrivals=gsl_vector_calloc(nb_arr+1);
  gsl_vector_set(Arrivals,0,(double)nb_arr);

  n=0;
  
  if(gsl_vector_get(Tn,0)==1)
    {
	gsl_vector_set(Arrivals,place,0);
	++place;
    }

  for(n=1;n<Tn->size;++n)
    if(gsl_vector_get(Tn,n)-gsl_vector_get(Tn,n-1) == 1)
      {
	gsl_vector_set(Arrivals,place,(double)n);
	++place;
      }
  return Arrivals;

}

gsl_vector* Post_Processing::ThinningArrivalTimes(gsl_vector *Tn, unsigned int min_distance)
{
	unsigned long m,n,nb_points=0;
	gsl_vector *diff_Tn=gsl_vector_calloc(Tn->size);
	
	if(Tn->size == 1)
		return diff_Tn;
	else
	{
		nb_points=1;
		gsl_vector_set(diff_Tn,0,gsl_vector_get(Tn,0)-1);
		for(m=1;m<diff_Tn->size;++m)
		{
			double val=gsl_vector_get(Tn,m+1)-gsl_vector_get(Tn,m);
			gsl_vector_set(diff_Tn,m,val);
			if(val > min_distance)
				++nb_points;
		}
		
		gsl_vector *new_points=gsl_vector_calloc(nb_points+1);
		gsl_vector_set(new_points,0,nb_points);
		n=1;
		
		for(m=1;m<diff_Tn->size;++m)
		{
			if(gsl_vector_get(diff_Tn,m)>min_distance)
			{
				gsl_vector_set(new_points,n,gsl_vector_get(Tn,m+1));
				++n;
			}
		}
		
		gsl_vector_free(diff_Tn);
		return new_points;
	}
}	
	
