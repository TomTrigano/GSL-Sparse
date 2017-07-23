#include "Const_Dictionary.h"

//#define DEBUG_SHAPES

// default class constructor
Const_Dictionary::Const_Dictionary()
{
  IncludeShapeReset=0;
  nb_shapes=4;
  nb_params=2;
  shape_length=15;
  signal_size=200;
  rangesave=5;
  shape_reset= gsl_vector_calloc(15); 
  typeshape="gamma";
  params=gsl_matrix_calloc(nb_params,nb_shapes);
  // declaring and allocating the dictionary
  shapes=gsl_matrix_calloc(shape_length,nb_shapes+IncludeShapeReset);
  BigA=gsl_matrix_calloc(signal_size,(nb_shapes+IncludeShapeReset)*(signal_size-rangesave));
}

// manual class constructor
Const_Dictionary::Const_Dictionary(unsigned long a, gsl_vector *b, unsigned long c, unsigned long d, std::string e, gsl_matrix *f, unsigned long g, unsigned long h, unsigned long i, gsl_matrix *j,gsl_matrix *k) : IncludeShapeReset(a), shape_reset(b), nb_shapes(c), nb_params(d), typeshape(e), params(f), signal_size(g), shape_length(h), rangesave(i), shapes(j), BigA(k)
{}

Const_Dictionary::Const_Dictionary(unsigned long a, gsl_vector *b, unsigned long c, unsigned long d, std::string e, gsl_matrix *f, unsigned long g, unsigned long h, unsigned long i) : IncludeShapeReset(a), shape_reset(b), nb_shapes(c), nb_params(d), typeshape(e), params(f), signal_size(g), shape_length(h), rangesave(i)
{
  shapes=gsl_matrix_calloc(h,c+a);
  BigA=gsl_matrix_calloc(g,(c+a)*(g-i));
}

// class destructor
Const_Dictionary::~Const_Dictionary()
{}

////////////////////////////////////////////////////////////////////////////////
//////////////////////  CONST_DICTIONARY INNER FUNCTIONS /////////////////////////
////////////////////////////////////////////////////////////////////////////////

void Const_Dictionary::ShowParams()
{
  unsigned long i=0,j=0;

  std::cout << std::endl << "params = [" ;

  for(i=0;i<params->size1;++i)
   {
     std::cout << "[" ;
     for(j=0;j<params->size2;++j)
       std::cout << gsl_matrix_get(params,i,j) << " " ;
     std::cout << "]" << std::endl;
   }
  std::cout << "]" << std::endl;
}

void Const_Dictionary::SetRealisticParams(double param_min1, double param_increm,double decay)
{
unsigned long i=0;

 for(i=0;i<nb_shapes;++i)
   {
     gsl_matrix_set(params,0,i,param_min1+param_increm*i);
     gsl_matrix_set(params,1,i,decay);
   }
}

void Const_Dictionary::SetParams(double param_min1, double param_min2 , double param_increm)
{
unsigned long i=0;

 for(i=0;i<nb_shapes;++i)
   {
     gsl_matrix_set(params,0,i,param_min1+param_increm*i);
     gsl_matrix_set(params,1,i,param_min2+param_increm*i);
   }
}

void Const_Dictionary::SetParams(double param_min1, double param_increm)
{
unsigned long i=0;

 for(i=0;i<nb_shapes;++i)
   gsl_matrix_set(params,0,i,param_min1+param_increm*i);
}


void Const_Dictionary::SetShapes(std::string normalizationtype)
{
  unsigned long i,j;
  double norm;
  
  for(j=0;j<nb_shapes;j++)
    {
      for(i=0;i<shape_length;i++)
	{
	  if(typeshape.compare("gamma")==0)
	    gsl_matrix_set(shapes,i,j,pow(double(i),gsl_matrix_get(params,0,j))*exp(-double(i)*gsl_matrix_get(params,1,j)));
	  if(typeshape.compare("atet")==0)
	    gsl_matrix_set(shapes,i,j,(double(i)*gsl_matrix_get(params,0,j))*exp(-double(i)*gsl_matrix_get(params,1,j)));
	  if(typeshape.compare("exponential")==0)
	    gsl_matrix_set(shapes,i,j,exp(-double(i)*gsl_matrix_get(params,0,j)));  
	  if(typeshape.compare("linexp_adonis")==0)
	  {
	    if(i<shape_length/5) // CHANGE THIS PARAMETER IF NEEDED
	      gsl_matrix_set(shapes,i,j,1-exp(-double(i)*gsl_matrix_get(params,0,j)));
	    else
	      gsl_matrix_set(shapes,i,j,(1-exp(-gsl_matrix_get(params,0,j)*shape_length/5))*exp(-double(i-shape_length/5)*gsl_matrix_get(params,1,j)));
		}
	}
      
      gsl_vector_view col=gsl_matrix_column(shapes,j);	

      if(normalizationtype.compare("L1")==0)
	norm=gsl_blas_dasum(&col.vector)/signal_size;
      if(normalizationtype.compare("L2")==0)
	norm=gsl_blas_dnrm2(&col.vector)/sqrt(signal_size);
      if(normalizationtype.compare("max")==0)
	norm=gsl_vector_max(&col.vector);
      if(normalizationtype.compare("none")==0)
	norm=1;

      gsl_blas_dscal(1/norm,&col.vector);	
    }
#ifdef DEBUG_SHAPES
 for(j=0;j<shape_length;++j)
   {
     for(i=0;i<nb_shapes;++i)
       std::cout << gsl_matrix_get(shapes,j,i) << " " ;
     std::cout << std::endl;
   }
#endif

}

void Const_Dictionary::SetDictionary(std::string normalizationtype)
{
  unsigned long j;
  unsigned long a=signal_size-shape_length;
  unsigned long b=signal_size-(shape_reset->size);

  SetShapes(normalizationtype);

  for(j=0;j<signal_size-rangesave;j++)
    {
      unsigned long leg=shape_length;
      if(j>a)
	leg=signal_size-j;

      gsl_matrix_view R=gsl_matrix_submatrix(BigA,j,(nb_shapes+IncludeShapeReset)*j,leg,nb_shapes);
      gsl_matrix_view S=gsl_matrix_submatrix(shapes,0,0,leg,nb_shapes);
      gsl_matrix_memcpy(&R.matrix,&S.matrix);
      if(j>b)
	leg=signal_size-j;
      if(IncludeShapeReset)
	{
	  gsl_vector_view columnreset=gsl_matrix_column(BigA,(nb_shapes+IncludeShapeReset)*(j+1)-1);
	  gsl_vector_view trunccolumnreset=gsl_vector_subvector(&columnreset.vector,j,leg);
	  gsl_vector_view truncreset=gsl_vector_subvector(shape_reset,0,leg);
	  gsl_vector_memcpy(&trunccolumnreset.vector,&truncreset.vector);
	}
    }
}

gsl_matrix* Const_Dictionary::GetSubdictionary_cols(gsl_vector *bool_col)
{
  gsl_matrix *B;
  unsigned long i,c=0;

  for(i=0;i<bool_col->size;i++)
    c += (gsl_vector_get(bool_col,i) > 0 ? 1 : 0) ;

  B=gsl_matrix_calloc(BigA->size1,c);

  c=0;

  for(i=0;i<BigA->size2;i++)
    {
      if(gsl_vector_get(bool_col,i)==1)
	{
	  gsl_vector_view A_column = gsl_matrix_column(BigA,i);
	  gsl_matrix_set_col(B,c,&A_column.vector);
	  ++c;
	}
    }
  return B;
}

gsl_matrix* Const_Dictionary::GetSubdictionary_blocks(unsigned long nb_blocks)
{
  unsigned long totshapes=IncludeShapeReset+nb_shapes;
  gsl_matrix *B = gsl_matrix_calloc(nb_blocks,(nb_blocks-rangesave)*totshapes);
  
  gsl_matrix_view sub_A = gsl_matrix_submatrix(BigA,0,0,nb_blocks,(nb_blocks-rangesave)*totshapes);
  gsl_matrix_memcpy(B,&sub_A.matrix);
  return B;
}

gsl_matrix* Const_Dictionary::GetSubdictionary_blocks(gsl_vector *bool_block)
{
unsigned long i,totshapes=IncludeShapeReset+nb_shapes,c=0;
gsl_matrix *B;
gsl_vector *bool_col = gsl_vector_calloc(BigA->size2);
gsl_vector *y = gsl_vector_calloc(totshapes);
gsl_vector_add_constant(y,1);

for(i=0;i<bool_block->size;i++)
{
if(gsl_vector_get(bool_block,i)>0)
{
c++;
gsl_vector_view x = gsl_vector_subvector(bool_col,totshapes*i,totshapes);
gsl_vector_memcpy(&x.vector,y);
}
}
B=GetSubdictionary_cols(bool_col);
gsl_vector_free(y);
gsl_vector_free(bool_col);
return B;
}

gsl_matrix* Const_Dictionary::GetDictionary()
{
gsl_matrix *Acopy=gsl_matrix_calloc(BigA->size1,BigA->size2);
gsl_matrix_memcpy(Acopy,BigA);
return Acopy;
}

gsl_matrix* Const_Dictionary::GetReweightedDictionary(gsl_matrix *W, int dimension)
{
gsl_matrix *Acopy=gsl_matrix_calloc(BigA->size1,BigA->size2);

if(dimension==1)
 gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, W, BigA, 0, Acopy);
else if(dimension==2)
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, BigA, W, 0, Acopy);
else
std::cout << "ERROR: precise 1 for AW or 2 for WA." << std::endl;
 
return Acopy;
}
