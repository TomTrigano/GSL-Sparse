#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h> 
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*==== Setting relevant dimensions ====*/
int rangesave=2;
int shape_length=4;
unsigned int nb_shapes=4;
int signal_size=20;

/*============ DECLARATIONS ===============*/
gsl_vector* Getparams(unsigned int , double , double , double );

gsl_matrix* Getshapes(gsl_vector* , char* );

gsl_matrix* Create_Dico_NHPP(gsl_vector* , char* );

gsl_matrix *GetDictionary_cols(gsl_vector* , gsl_matrix* );

/*=====================================*/

gsl_vector* Getparams(unsigned int number, double param_min1, double param_min2 , double param_increm)
{
int i=0;
gsl_vector *lambda;
lambda=gsl_vector_calloc(2*number);

while(i<number)
{
gsl_vector_set(lambda,2*i,param_min1+param_increm*i);
gsl_vector_set(lambda,2*i+1,param_min2+param_increm*i);
i++;
}

return(lambda);
}

gsl_matrix* Getshapes(gsl_vector *lambda, char *normalizationtype)
{
int i,j;
double norm;

gsl_matrix *shapes;
shapes=gsl_matrix_calloc(shape_length,nb_shapes);

  for(j=0;j<nb_shapes;j++)
    {
	 for(i=0;i<shape_length;i++)
	    {
		gsl_matrix_set(shapes,i,j,pow(i,gsl_vector_get(lambda,2*j))*exp(-i*gsl_vector_get(lambda,2*j+1)));
	    }
	 gsl_vector_view col=gsl_matrix_column(shapes,j);	
	 if(strcmp(normalizationtype,"L1")==0)
        {
         norm=gsl_blas_dasum(&col.vector)/signal_size;
		}
     else
		{
		 norm=gsl_blas_dnrm2(&col.vector)/sqrt(signal_size);
		}
	 gsl_blas_dscal(1/norm,&col.vector);	
	}

return(shapes);
}

gsl_matrix* Create_Dico_NHPP(gsl_vector *lambda, char *normalizationtype)
{
int i,j,nb_columns=nb_shapes*(signal_size-rangesave);

gsl_matrix *A,*shapes;
A=gsl_matrix_calloc(signal_size,nb_columns);

shapes=Getshapes(lambda, normalizationtype);

for(j=0;j<signal_size-rangesave;j++)
  {
   float leg=shape_length;
   if(j>signal_size-shape_length)
   {
    leg=signal_size-j;
   }
   gsl_matrix_view R=gsl_matrix_submatrix(A,j,nb_shapes*j,leg,nb_shapes);
   gsl_matrix_view S=gsl_matrix_submatrix(shapes,0,0,leg,nb_shapes);
   gsl_matrix_memcpy(&R.matrix,&S.matrix);
  }

return(A);
}

gsl_matrix *GetDictionary_cols(gsl_vector *bool_col, gsl_matrix *A)
{
int i;
gsl_matrix *B;
B=gsl_matrix_calloc(signal_size,bool_col->size);

for(i=0;i<bool_col->size;i++)
{
gsl_vector_view column=gsl_matrix_column(A,gsl_vector_get(bool_col,i));
gsl_vector_view column_B=gsl_matrix_column(B,i);
gsl_vector_memcpy(&column_B.vector,&column.vector);
}

return(B);
}

gsl_matrix* GetDictionary_blocks(unsigned long nb_blocks, gsl_vector *bool_block, char *method, gsl_matrix *A)
{
int i;
gsl_matrix *B;
gsl_vector *x,*y;
gsl_vector *bool_col;

x=gsl_vector_calloc(nb_shapes);
y=gsl_vector_calloc(nb_shapes);

for(i=0;i<nb_shapes;i++)
{
gsl_vector_set(x,i,1);
gsl_vector_set(y,i,i);
}

if(strcmp(method,"first")==0)
{
bool_col=gsl_vector_calloc(nb_shapes*nb_blocks);
B=gsl_matrix_calloc(signal_size,bool_col->size);

for(i=0;i<bool_col->size;i++)
  {
   gsl_vector_set(bool_col,i,i);
  }
 
}
else
{
gsl_vector *z;
z=gsl_vector_calloc(nb_shapes);
bool_col=gsl_vector_calloc(nb_shapes*(bool_block->size));
B=gsl_matrix_calloc(signal_size,bool_col->size);

for(i=0;i<bool_block->size;i++)
  {
   gsl_blas_dcopy(y,z);
   gsl_vector_view indices=gsl_vector_subvector(bool_col,nb_shapes*i,nb_shapes);
   gsl_blas_daxpy(nb_shapes*gsl_vector_get(bool_block,i),x,z);
   gsl_vector_memcpy(&indices.vector,z);
  }

}

B=GetDictionary_cols(bool_col, A); 
return(B);
}

/*==============================================*/
 main()
{
int i,j;

gsl_vector *lambda,*bool_col;
gsl_matrix *dico,*subdico;

lambda=Getparams(nb_shapes, 0.5, 0.7 , 0.2);

dico=Create_Dico_NHPP(lambda, "default");
/*
for(i=0;i<shape_length+2;i++)
{
  for(j=0;j<3*nb_shapes;j++)
   {
    printf("%g;",gsl_matrix_get(dico,i,j));
   }
  printf("\n"); 
}
*/


double blocs[]={3 };
gsl_vector *bool_block;
bool_block=gsl_vector_calloc(1);

gsl_vector_view V=gsl_vector_view_array(blocs,1);
gsl_vector_memcpy(bool_block,&V.vector);

subdico=gsl_matrix_calloc(signal_size,nb_shapes*(bool_block->size));
subdico=GetDictionary_blocks(3, bool_block, "default", dico);

for(i=0;i<signal_size;i++)
{
  for(j=0;j<subdico->size2;j++)
   {
    printf("%g;",gsl_matrix_get(subdico,i,j));
   }
  printf("\n"); 
}

}