#include "TIMIT_Signal.h"

// default class constructor
TIMIT_Signal::TIMIT_Signal()
{
  filename="speech.wav";
  window_size=256;
  overlapping=128;
  nb_frames=1;
  sound=gsl_vector_calloc(window_size);
  frames=gsl_matrix_calloc(nb_frames,window_size);
  current_frame=gsl_vector_calloc(window_size);
  savename="speech_results.dat";
  binary_save=TRUE;
}

// manual class constructor
TIMIT_Signal::TIMIT_Signal(std::string s, unsigned long a, unsigned long b, unsigned long bb,gsl_vector* c, gsl_matrix* d, gsl_vector* e, std::string f, bool g) : filename(s),window_size(a),overlapping(b),nb_frames(bb),savename(f),binary_save(g)
{
  sound=gsl_vector_calloc(c->size);
  frame=gsl_matrix_calloc(d->size1,d->size2);
  current_frame=gsl_matrix_calloc(e->size);
  gsl_vector_memcpy(sound,c);
  gsl_matrix_memcpy(frames,d);
  gsl_vector_memcpy(current_frame,e);
}

// partial manual class constructor
TIMIT_Signal::TIMIT_Signal(std::string s, unsigned long a, unsigned long b, unsigned long bb, std::string f, bool g) : filename(s),window_size(a),overlapping(b),nb_frames(bb),savename(f),binary_save(g)
{
  sound=gsl_vector_calloc(a);
  frame=gsl_matrix_calloc(bb,a);
  current_frame=gsl_matrix_calloc(a);
  gsl_vector_memcpy(sound,c);
  gsl_matrix_memcpy(frames,d);
  gsl_vector_memcpy(current_frame,e);
}

// class destructor
TIMIT_Signal::~TIMIT_Signal()
{
  gsl_vector_free(sound);
  gsl_matrix_free(frames);
  gsl_vector_free(current_frame);
}

////////////////////////////////////////////////////////////////////////////////
/////////////////////////  TIMIT SIGNAL INNER FUNCTIONS /////////////////////////
////////////////////////////////////////////////////////////////////////////////

void TIMIT_Signal::RemoveDC()
{
  unsigned long k;
  double average=0;

  // Computes the mean
  for(k=0;k<sound->size;++k)
    average += gsl_vector_get(sound,k);
  average /= (double)(sound->size);

  // Discards the mean in vectors and matrices
  gsl_vector_add_constant(sound,-average);
  gsl_matrix_add_constant(frames,-average);
  gsl_vector_add_constant(current_frame,-average);
}

 void TIMIT_Signal::SetSigma(double value)
 {
   sigma = value;
 }

 void TIMIT_Signal::SetDC(double value)
 {
   DC = value;
 }

 void TIMIT_Signal::SetRSTThresh(double value)
{
  RST_lower_threshold = value;
}

gsl_vector* TIMIT_Signal::GetParameters() const
{
	gsl_vector *v=gsl_vector_calloc(3);
	gsl_vector_set(v,0,DC);
	gsl_vector_set(v,1,RST_lower_threshold);
	gsl_vector_set(v,2,sigma);
	return v;
}

void TIMIT_Signal::ProcessData() const
{

  std::ifstream ifs;
  std::ofstream sizes,values;
  std::stringstream ss1,ss2;
  unsigned long k,m=0;

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

      for(k=0;k<chunk_size;++k)
	{
	  int16_t n = 0;
	  n |= (unsigned char)ifs.get();
	  n |= ifs.get() << 8;
	  temp[k]=double(n)-DC;
	  test[k]=0;
	}

      // At  this stage, we have a vector which is probably badly cut and may include a reset

      // Check if there are resets undershoots (more than one are possible)
      for(k=0;k<chunk_size;++k)
	if(temp[k]<RST_lower_threshold)
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

      // Number of elements in the vectors which are related to a reset or a bad cut
      for(k=0;k<chunk_size;++k)
	if(test[k]==0)
	  ++good_points;

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
}

void TIMIT_Signal::ExtractPiledupEnergies(double thresh) const
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
	ProcessData();
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
	ProcessData();
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
