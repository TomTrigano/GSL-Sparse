#include "Spectro_Signal.h"

//#define DEBUG_SIGMA_ESTIMATE 1
//#define DEBUG_PROCESS_DATA 1

// default class constructor
SpectroSignal::SpectroSignal()
{
  filename="signal_1_1108634597_106629"; // raw signal file to process
  chunk_size=500; // the maximal size of signal chunks to process afterwards
  nb_chunks=(unsigned long)GSL_POSINF; // the number of chunks to be used (set to -1) if all the raw signal is used
  DC=0; // The DC value, set to 0 initially
  sigma=10;
  RST_lower_threshold=0; // the lower threshold for the reset
  savename="block"; // the basic name for saving all the vectors obtained
  binary_save=false;
  saturation_value=32767;
  nb_effective_chunks=0;
}

// manual class constructor
SpectroSignal::SpectroSignal(std::string s, unsigned long a, unsigned long b, double c, double d, double e, std::string f, bool g, double h) : filename(s), chunk_size(a), nb_chunks(b), DC(c), sigma(d), RST_lower_threshold(e), savename(f), binary_save(g), saturation_value(h)
{
  nb_effective_chunks = 0;
}

// class destructor
SpectroSignal::~SpectroSignal()
{}

////////////////////////////////////////////////////////////////////////////////
//////////////////////  SPECTRO SIGNAL INNER FUNCTIONS /////////////////////////
////////////////////////////////////////////////////////////////////////////////

void SpectroSignal::ComputeSigma(unsigned long piece_size,double minval,double maxval, unsigned long nb_bins,double normalize)
{
  unsigned long k;
  gsl_vector *v=gsl_vector_calloc(piece_size);
  std::ifstream ifs;
  gsl_histogram * h = gsl_histogram_alloc (nb_bins);
  size_t i=0;

  ifs.open(filename.c_str(),std::ifstream::binary);
  
  for(k=0;k<piece_size;++k)
    {
      int16_t n = 0;
      n |= (unsigned char)ifs.get();
      n |= ifs.get() << 8;
      gsl_vector_set(v,k,double(n)-DC);
    }

  gsl_histogram_set_ranges_uniform(h,minval,maxval);
  for(k=0;k<v->size;++k)
    gsl_histogram_increment(h,gsl_vector_get(v,k));

#ifdef DEBUG_SIGMA_ESTIMATE
  std::cout << "On a construit l'histograaaaaamme" <<  std::endl;
  gsl_histogram_fprintf (stdout, h, "%g", "%g");
#endif

  // LE BUG IL EST LA :)
  gsl_histogram_find(h,0.0,&i);

#ifdef DEBUG_SIGMA_ESTIMATE
  std::cout << "coin:" <<  std::endl;
  gsl_histogram_fprintf (stdout, h, "%g", "%g");
#endif

  while(gsl_histogram_get(h,i)>gsl_histogram_max(h)/normalize)
    i -= 1;
  sigma=fabs(double(h->range[i+1])/3.);

#ifdef DEBUG_SIGMA_ESTIMATE
    std::cout << "Value of the estimated noise std dev: " << sigma << std::endl;
#endif
    ifs.close();
    gsl_vector_free(v);
    gsl_histogram_free(h);
}

void SpectroSignal::ComputeRSTThreshold(long piece_size,double quantile)
{
  long k;
  gsl_vector *v=gsl_vector_calloc(piece_size);
  std::ifstream ifs;

  ifs.open(filename.c_str(),std::ifstream::binary);
  
  for(k=0;k<piece_size;++k)
    {
      int16_t n = 0;
      n |= (unsigned char)ifs.get();
      n |= ifs.get() << 8;
      gsl_vector_set(v,k,double(n)-DC);
#ifdef DEBUG_RST_ESTIMATE_0
      std::cout << k << ", " << n << std::endl;
#endif
    }
  
#ifdef DEBUG_RST_ESTIMATE
  FILE *fid=fopen("debug.bin","wb");
  gsl_vector_fwrite(fid,v);
  fclose(fid);
#endif

  RST_lower_threshold = quantile*gsl_vector_min(v);

#ifdef DEBUG_RST_ESTIMATE
    std::cout << "Value of the RSTt estimated by a percentage of the mean: " << RST_lower_threshold << std::endl;
#endif
    ifs.close();
    gsl_vector_free(v);
}

void SpectroSignal::ComputeMedianDC(long piece_size)
{
  long k;
  double values[piece_size];
  gsl_vector *v=gsl_vector_calloc(piece_size);
  std::ifstream ifs;

  ifs.open(filename.c_str(),std::ifstream::binary);
  
  for(k=0;k<piece_size;++k)
    {
      int16_t n = 0;
      n |= (unsigned char)ifs.get();
      n |= ifs.get() << 8;
      values[k]=double(n);
      gsl_vector_set(v,k,double(n));
#ifdef DEBUG_DC_ESTIMATE
      std::cout << k << ", " << n << std::endl;
#endif
    }
  
#ifdef DEBUG_DC_ESTIMATE
  FILE *fid=fopen("debug.bin","wb");
  gsl_vector_fwrite(fid,v);
  fclose(fid);
#endif
  
    gsl_sort(values,1,piece_size);
    DC=gsl_stats_median_from_sorted_data(values,1,piece_size);
    
#ifdef DEBUG_DC_ESTIMATE
    std::cout << "Value of the DC estimated by the median: " << DC << std::endl;
#endif
    ifs.close();
    gsl_vector_free(v);
}

 void SpectroSignal::SetSigma(double value)
 {
   sigma = value;
 }

 void SpectroSignal::SetDC(double value)
 {
   DC = value;
 }

 void SpectroSignal::SetRSTThresh(double value)
{
  RST_lower_threshold = value;
}

gsl_vector* SpectroSignal::GetParameters() const
{
	gsl_vector *v=gsl_vector_calloc(4);
	gsl_vector_set(v,0,DC);
	gsl_vector_set(v,1,RST_lower_threshold);
	gsl_vector_set(v,2,sigma);
	gsl_vector_set(v,3,nb_effective_chunks);
	return v;
}

void SpectroSignal::ProcessData(std::string cutno) 
{

  std::ifstream ifs;
  std::ofstream sizes,values;
  std::stringstream ss1,ss2;
  unsigned long k,m=0;
  
  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  

  ifs.open(filename.c_str(),std::ifstream::binary);
  if(binary_save)
    {
      ss1 << savename << ".sizes.bin";
      ss2 << savename << ".values.bin";
      sizes.open(ss1.str().c_str(),std::ofstream::binary);
      values.open(ss2.str().c_str(),std::ofstream::binary);
    }
  else
    {
      ss1 << savename << ".sizes.txt";
      ss2 << savename << ".values.txt";
      sizes.open(ss1.str().c_str());
      values.open(ss2.str().c_str());
    }
  
  while((!ifs.eof()) && (m<nb_chunks) )
    {
      double temp[chunk_size];
      double test[chunk_size];
      unsigned long good_points=0;
      ++nb_effective_chunks;

      for(k=0;k<chunk_size;++k)
	{
	  int16_t n = 0;
	  n |= (unsigned char)ifs.get();
	  n |= ifs.get() << 8;
	  temp[k]=double(n)-DC;
	  test[k]=0;
	}

      // At  this stage, we have a vector which is probably badly cut and may include a reset and saturated pulses

      // Check if there are resets undershoots (more than one are possible)
      for(k=0;k<chunk_size;++k)
	if((temp[k]<RST_lower_threshold) || temp[k]==saturation_value)
	  test[k]=1;

      // Aggregates consecutive points to identify the whole reset
      if(fabs(temp[0])>3*sigma)
	test[0] = 1;
      if(fabs(temp[chunk_size-1])>3*sigma)
	test[chunk_size-1]=1;
      // Forward
      for(k=0;k<chunk_size-2;k++)
	if((test[k] == 1) && (test[k+1]==0) && (fabs(temp[k+1])>3*sigma))
	  test[k+1]=1;
      // Backward
      for(k=chunk_size-1;k>1;--k)
	if((test[k] == 1) && (test[k-1]==0) && (fabs(temp[k-1])>3*sigma))
	  test[k-1]=1;

	  if(cutno.compare("cut")==0)
		{      // Number of elements in the vectors which are related to a reset or a bad cut
			for(k=0;k<chunk_size;++k)
				if(test[k]==0)
					++good_points;
		}
	
	if(cutno.compare("replace")==0)
		{      // Number of elements in the vectors which are related to a reset or a bad cut
			for(k=0;k<chunk_size;++k)
			{	
				if(test[k]==1)
				{	temp[k]=gsl_ran_gaussian(r,sigma); test[k]=0; };
			};
			good_points=chunk_size;
		};


#ifdef DEBUG_PROCESS_DATA
      gsl_vector_view test_v=gsl_vector_view_array(test,chunk_size);
      gsl_vector_view temp_v=gsl_vector_view_array(temp,chunk_size);
      std::cout << "bad points nb for chunk " << m << " = " << chunk_size-good_points << std::endl;
      {
	std::stringstream a,b;
	a << "save_debug_temp_" << m << ".txt";
	b << "save_debug_test_" << m << ".txt";
	FILE *fid=fopen(a.str().c_str(),"w");

	gsl_vector_fprintf(fid,&temp_v.vector,"%.8f");
	fclose(fid);
	fid=fopen(b.str().c_str(),"w");
	gsl_vector_fprintf(fid,&test_v.vector,"%.8f");
	fclose(fid);
      }
#endif
      // Save the useful vector (we unify all chunks in one - Poissonity allows it somehow...)
      if(binary_save)
	{
	  sizes.write(reinterpret_cast<char*>(&good_points),sizeof(good_points));
	  for(k=0;k<chunk_size;++k)
	    if(test[k]==0)
	      values.write(reinterpret_cast<char*>(&temp[k]),sizeof(temp[k]));
	}
      else
	{
	  sizes << std::fixed << good_points << std::endl;
	  for(k=0;k<chunk_size;++k)
	    if(test[k]==0)
	      values << std::fixed << std::setprecision(16) << temp[k] << std::endl;
	}

      ++m;
    }
 
  ifs.close();
  sizes.close();
  values.close();
  gsl_rng_free (r);

}

void SpectroSignal::ExtractPiledupEnergies(double thresh)
{
  std::stringstream ss1,ss2;
  std::ifstream ifs;
  std::ofstream ofs;
  double acc=0;
  int above_threshold=0;
  
  if(binary_save)
    {
      ss1 << savename << ".values.bin";
      ss2 << savename << ".piledupenergies.bin";
      std::ifstream test(ss1.str().c_str(),std::ofstream::binary);
      if(test.fail())
	SpectroSignal::ProcessData("replace");
      test.close();
      ifs.open(ss1.str().c_str(),std::ofstream::binary);
      ofs.open(ss2.str().c_str(),std::ofstream::binary);
      
    }
  else
    {
      ss1 << savename << ".values.txt";
      ss2 << savename << ".piledupenergies.txt";
      std::ifstream test(ss1.str().c_str());
      if(test.fail())
	SpectroSignal::ProcessData("replace");
      test.close();
      ifs.open(ss1.str().c_str());
      ofs.open(ss2.str().c_str());
    }
  
  while(!ifs.eof())
    {
      double value;
      ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
      
      if(value>thresh)
	{
	  acc += value ;
	  above_threshold = 1;
	}
      else
	{
	  if(above_threshold == 1)
	    {
	      if(binary_save)
		ofs.write( reinterpret_cast<char*>( &acc ), sizeof(acc));
	      else
		ofs << std::fixed << std::setprecision(16) << acc << std::endl;
	    }
	  acc=0;
	  above_threshold=0;
	}
      
}
  
  ifs.close();
  ofs.close();

  
}
