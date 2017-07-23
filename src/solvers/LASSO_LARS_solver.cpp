#include "LASSO_LARS_solver.h"

//#define DEBUG_STOPPING_CRITERION 0
//#define DEBUG_MAIN_FUNCTION 1
//#define DEBUG_GAMMA 1
//#define DEBUG_LARS_DIRECTION 1
//#define DEBUG_CHOL 0
//#define DEBUG_REDUCEACTIVESET 1
//#define DEBUG_COLINEARITY
#define ZEROTOL 1e-5
#define Q_PRECISION 1e-5
#define EPSILON 1e-12

// default class constructor
LASSO_LARS_solver::LASSO_LARS_solver()
{
  A=gsl_matrix_calloc(10,30); // dictionary
  y=gsl_vector_calloc(10); // output signal
  algType="LASSO"; // can be lars, lasso, nnlars, nnlasso
  maxIters=1000000; // maximal number of interations
  lambdaStop=0; // when the Lagrange multiplier <= lambdaStop.
  resStop=0; // L2 norm of the residual <= resStop. 
  verbose=1; // 1 to print out detailed progress at each iteration, 0 for no output.
  Aactive=gsl_matrix_calloc(A->size1,A->size2);
  Aactive_size=0;
  newIndices=gsl_vector_calloc(A->size2);
  removeIndices=gsl_vector_calloc(A->size2);;
  activeSet=gsl_vector_calloc(A->size2);;
  inactiveSet=gsl_vector_calloc(A->size2);
  collinearSet=gsl_vector_calloc(A->size2);
  R=gsl_matrix_calloc(A->size2,A->size2);
  Rsize=0;
  residual=gsl_vector_calloc(y->size);
  corr=gsl_vector_calloc(A->size2);
  lambda=0;
  beta=gsl_vector_calloc(A->size2); // solution of the LASSO
  OptTol=1e-5; // error tolerance
	
  gsl_matrix_set_identity(A);
  gsl_vector_memcpy(residual,y);
}

// manual class constructor
LASSO_LARS_solver::LASSO_LARS_solver(gsl_matrix *a,gsl_vector *b, std::string c, unsigned long d, double e, double f, int g) : A(a), y(b), algType(c), maxIters(d), lambdaStop(e), resStop(f), verbose(g)
{
  Aactive=gsl_matrix_calloc(A->size1,A->size2);
  Aactive_size=0;
  newIndices=gsl_vector_calloc(A->size2);
  removeIndices=gsl_vector_calloc(A->size2);;
  activeSet=gsl_vector_calloc(A->size2);;
  inactiveSet=gsl_vector_calloc(A->size2);
  collinearSet=gsl_vector_calloc(A->size2);
  R=gsl_matrix_calloc(A->size2,A->size2);
  Rsize=0;
  residual=gsl_vector_calloc(y->size);
  corr=gsl_vector_calloc(A->size2);
  lambda=0;
  beta=gsl_vector_calloc(A->size2); // solution of the LASSO
  OptTol=1e-5; // error tolerance
	
  gsl_vector_memcpy(residual,y);
}

LASSO_LARS_solver::~LASSO_LARS_solver()
{
  //	gsl_matrix_free(A);
  //	gsl_vector_free(y);
  gsl_matrix_free(Aactive);
  gsl_vector_free(newIndices);
  gsl_vector_free(removeIndices);
  gsl_vector_free(activeSet);
  gsl_vector_free(inactiveSet);
  gsl_vector_free(collinearSet);
  gsl_matrix_free(R);
  gsl_vector_free(residual);
  gsl_vector_free(corr);
  gsl_vector_free(beta);
}

///////////////////// DEBUGGING FUNCTIONS ////////////////////////////////////
void LASSO_LARS_solver::SetLambdaMaxCorr(int nntest)
{
  if(nntest)
    lambda=gsl_vector_get(corr,gsl_vector_max_index(activeSet));
  else
    lambda=fabs(gsl_vector_get(corr,gsl_vector_max_index(activeSet)));
}

void LASSO_LARS_solver::DisplayVector(gsl_vector *v)
{
  std::cout << "Vector is of size " << v->size << std::endl << "[" ;
  size_t k;
  for(k=0;k<v->size ;++k)
    std::cout << gsl_vector_get(v,k) << " ";
  std::cout << "]^T" << std::endl;
}

void LASSO_LARS_solver::DisplayPrivateMembers(std::string input)
{
  unsigned long i,j;

  // A
  // std::cout << "A=[" << std::endl ;

  // for(i=0;i<A->size1;++i)
  //   {
  //     std::cout << "[" ;
  //     for(j=0;j<A->size2;++j)
  // 	std::cout << gsl_matrix_get(A,i,j) << " " ;
  //     std::cout << "]" << std::endl ;
  //   }
  // std::cout << "]" << std::endl << std::endl ;

  // AAactive
  if(input.compare("Aactive")==0)
    {
      std::cout << "Aactive=[" << std::endl ;

      for(i=0;i<Aactive->size1;++i)
	{
	  std::cout << "[" ;
	  for(j=0;j<Aactive_size;++j)
	    std::cout << gsl_matrix_get(Aactive,i,j) << " " ;
	  std::cout << "]" << std::endl ;
	}
      std::cout << "]" << std::endl << std::endl ;

      // Aactive size
      std::cout << "Aactive_size = " << Aactive_size << std::endl << std::endl ;
    }

  // lambda
  //  std::cout << "lambda = " << lambda << std::endl << std::endl ;

  // R
  if(input.compare("R")==0)
    {
      std::cout << "R (" << Rsize << "x" << Rsize <<")=[" << std::endl ;

      for(i=0;i<Rsize;++i)
	{
	  std::cout << "[" ;
	  for(j=0;j<Rsize;++j)
	    std::cout << gsl_matrix_get(R,i,j) << " " ;
	  std::cout << "]" << std::endl ;
	}
      std::cout << "]" << std::endl << std::endl  ;

      // R size
      std::cout << "Rsize = " << Rsize << std::endl << std::endl ;
    }

  // newIndices
  //  std::cout << "newIndices (" << newIndices->size <<")=[";

  // for(i=0;i<newIndices->size;++i)
  //   if(gsl_vector_get(newIndices,i)==1)
  //     std::cout << i << " ";
  // std::cout << "]^T" << std::endl << std::endl  ;

  //  // removeIndices
  // std::cout << "removeIndices (" << removeIndices->size <<")=[" ;
  // for(i=0;i<removeIndices->size;++i)
  //   if(gsl_vector_get(removeIndices,i)==1)
  //     std::cout << i << " ";
  // std::cout << "]^T" << std::endl << std::endl  ;

  //  // activeSet
  //  std::cout << "activeSet (" << activeSet->size <<")=[" ;

  //  {
  //    size_t cntr;
  //    for(cntr=1;cntr<=Rsize;++cntr)
  //      for(i=0;i<activeSet->size;++i)
  // 	 if(gsl_vector_get(activeSet,i)==cntr)
  // 	   std::cout << i << " ";
  //   }
  // std::cout << "]^T" << std::endl << std::endl ;

  //  // inactiveSet
  // std::cout << "inactiveSet=["  << std::endl;

  // for(i=0;i<inactiveSet->size;++i)
  //   std::cout << gsl_vector_get(inactiveSet,i) << " ";
  // std::cout << "]^T" << std::endl << std::endl ;

  // collinearSet
  // std::cout << "collinearSet=[" ;

  // for(i=0;i<collinearSet->size;++i)
  //   std::cout << gsl_vector_get(collinearSet,i) << " ";
  // std::cout << "]^T" << std::endl << std::endl  ;

  // y
  // std::cout << "y=["  ;

  // for(i=0;i<y->size;++i)
  //   std::cout << gsl_vector_get(y,i) << " ";
  // std::cout << "]^T" << std::endl << std::endl ;

  // residual
  std::cout << "residual=["  ;

  for(i=0;i<residual->size;++i)
    std::cout << gsl_vector_get(residual,i) << " ";
  std::cout << "]^T" << std::endl << std::endl ;

  // corr
  // std::cout << "corr=["  ;

  // for(i=0;i<corr->size;++i)
  //   std::cout << gsl_vector_get(corr,i) << " ";
  // std::cout << "]^T" << std::endl << std::endl ;

  // beta
  // std::cout << "beta=["  ;

  // for(i=0;i<beta->size;++i)
  //   std::cout << gsl_vector_get(beta,i) << " ";
  // std::cout << "]^T" << std::endl << std::endl  ;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////// AUXILIARY FUNCTIONS /////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
void LASSO_LARS_solver::GetRegressor(gsl_vector *dest)
{
  if(beta->size == dest->size)
    gsl_vector_memcpy(dest,beta);
  else
    std::cout << "ERROR (GetRegressor): the target vector and beta have not the same size !" << std::endl ;
}
/////////////////////////////////////////////////////////////////////////////////
void LASSO_LARS_solver::updateChol(unsigned long newIndex, int &flag)
{
	
  gsl_vector_view new_vec=gsl_matrix_column(A,newIndex);
  flag=0;
	
  if(Aactive_size==0)	
    {
      gsl_matrix_set(R,0,0,gsl_blas_dnrm2(&new_vec.vector));
      gsl_matrix_set_col(Aactive,0,&new_vec.vector);
    }
  else
    {	
      gsl_matrix_view Rtemp=gsl_matrix_submatrix(R,0,0,Rsize,Rsize);
      gsl_matrix_view Aactivetemp=gsl_matrix_submatrix(Aactive,0,0,Aactive->size1,Aactive_size);
      double q=0,tmp=0;
      //double temp[A->size1],
      double s1[Rsize];
      //gsl_vector_view tempvec=gsl_vector_view_array(temp,A->size1);
      gsl_vector_view sol=gsl_vector_view_array(s1,Rsize);
			
      gsl_vector_set_zero(&sol.vector);
      gsl_blas_dgemv(CblasTrans,1,&Aactivetemp.matrix,&new_vec.vector,0,&sol.vector);
      gsl_blas_dtrsv(CblasUpper,CblasTrans,CblasNonUnit,&Rtemp.matrix,&sol.vector);
					
      // Update of the matrix R
      gsl_blas_ddot(&new_vec.vector,&new_vec.vector,&q);
      gsl_blas_ddot(&sol.vector,&sol.vector,&tmp);

      q = q - tmp;

#ifdef DEBUG_COLINEARITY
      std::cout << "colinearity q = " << q << std::endl;
#endif

      if(q <= Q_PRECISION)
	flag=1;
      else
	{
	  unsigned long i;
										
	  for(i=0;i<Rsize;++i)
	    gsl_matrix_set(R,i,Rsize,s1[i]);
					
	  for(i=0;i<Rsize;++i)
	    gsl_matrix_set(R,Rsize,i,0);
					
	  gsl_matrix_set(R,Rsize,Rsize,sqrt(q));
	  ++Rsize;
	}
				
    }
		
}

//////////////////////////////////////////////////////////////////////////////////
void LASSO_LARS_solver::downdateChol(unsigned long j)
{
  gsl_matrix *Rtemp=gsl_matrix_calloc(Rsize,Rsize-1);
  gsl_vector *temp=gsl_vector_calloc(R->size1);

  unsigned long k,m;
  unsigned long n=Rsize-1;

  // Remove the elements from the Cholesky matrix
  for(m=0;m<Rsize;++m)
    {
      if(j<Rsize-1)
	{
	  for(k=j;k<Rsize-1;++k)
	    {
	      double tmp=gsl_matrix_get(R,m,k+1);
	      gsl_matrix_set(R,m,k,tmp);
	    }
	}
    }

  for(k=0;k<Rsize;++k)
    gsl_matrix_set(R,k,Rsize-1,0);

  // Removes the elements from the Aactive matrix
  for(m=0;m<Aactive->size1;++m)
    {
      if(j<Aactive_size-1)
	{
	  for(k=j;k<Aactive_size-1;++k)
	    {
	      double tmp=gsl_matrix_get(Aactive,m,k+1);
	      gsl_matrix_set(Aactive,m,k,tmp);
	    }
	}
    }

  for(k=0;k<Aactive_size;++k)
    gsl_matrix_set(Aactive,k,Aactive_size-1,0);


	
  // Givens rotations to cancel the violating nonzeros
	
  for(k=j;k<n;++k)
    {
      double r=gsl_hypot(gsl_matrix_get(R,k,k),gsl_matrix_get(R,k+1,k));
      double c=gsl_matrix_get(R,k,k) / r;
      double s=-gsl_matrix_get(R,k+1,k)/r;
		
      double t[4]={c,-s,s,c};
		
      gsl_matrix_set(R,k,k,r);
      gsl_matrix_set(R,k+1,k,0);
		
      if(k<n)
	{
	  gsl_matrix_view G = gsl_matrix_view_array(t,2,2);
	  gsl_matrix_view subR=gsl_matrix_submatrix(R,k,k+1,2,n-k);
	  gsl_matrix *subRcpy=gsl_matrix_calloc(2,n-k);
			
	  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,1.0, &G.matrix, &subR.matrix, 0.0, subRcpy);
	  gsl_matrix_memcpy(&subR.matrix,subRcpy);
			
	  gsl_matrix_free(subRcpy);
	}
    }
	
  for(k=0;k<R->size2;++k)
    gsl_matrix_set(R,n,k,0);
	
  gsl_matrix_free(Rtemp);
  gsl_vector_free(temp);

  --Rsize;
  --Aactive_size;
}

// TESTE - VERIFIER AVEC MATLAB - WORKS////////////////////////////////////////////////////////////////////////////////
int LASSO_LARS_solver::InitializeActiveSet(int nonNegative)
{
  unsigned long k;
  double counter = 1.;
  unsigned long nb_cols=A->size2;
  // First compute the maximum correlation
  gsl_blas_dgemv(CblasTrans,1,A,y,0,corr);
  gsl_matrix_set_zero(Aactive);
  Aactive_size=-1;
	
  // If the correlation is negative and we ask for something positive, we stop
  if(nonNegative==1)
    {
      lambda=gsl_vector_max(corr);
      if(lambda<0)
	{
	  std::cout << "ERROR:y is not expressible as a non-negative linear combination of the columns of A\n";
	  return 0;
	}
		
      // NOTE : ici new_indices est un vecteur binaire
		
      for(k=0;k<nb_cols;++k)
	{
	  double val=fabs(gsl_vector_get(corr,k)-lambda);
	  if(val < ZEROTOL)
	    {
	      double t[A->size1];
	      gsl_vector_view vec=gsl_vector_view_array(t,A->size1); 
	      gsl_vector_set(newIndices,k,counter);
	      counter += 1. ;
	      gsl_matrix_get_col(&vec.vector,A,k);
	      ++Aactive_size;
	      gsl_matrix_set_col(Aactive,Aactive_size,&vec.vector);
	    }
	  else
	    {
	      gsl_vector_set(newIndices,k,0.);
	    }

	}
    }
  else
    {
      double tab[A->size2];
      gsl_vector_view corr_copy=gsl_vector_view_array(tab,A->size2);
		
      for(k=0;k<A->size2;++k)
	tab[k]=fabs(gsl_vector_get(corr,k));
		
      lambda=gsl_vector_max(&corr_copy.vector);
		
      for(k=0;k<nb_cols;++k)
	{
	  double val=fabs(tab[k]-lambda);
	  if(val < ZEROTOL)
	    {
	      double t[A->size1];
	      gsl_vector_view vec=gsl_vector_view_array(t,A->size1); 
	      gsl_vector_set(newIndices,k,counter);
	      counter += 1.;
	      gsl_matrix_get_col(&vec.vector,A,k);
	      ++Aactive_size;
	      gsl_matrix_set_col(Aactive,Aactive_size,&vec.vector);
	    }
	  else
	    {
	      gsl_vector_set(newIndices,k,0.);
	    }

	}
			
    }

  return 1;
}

// TESTE - VERIFIER AVEC MATLAB - WORKS ////////////////////////////////////////////////////////////////////////////////
int LASSO_LARS_solver::CheckStoppingCriterion(double gammamin)
{
  bool cond1=((lambda-gammamin) < OptTol);
  bool cond2=((lambdaStop>0) && (lambda <= lambdaStop));
  bool cond3=((resStop>0) && (gsl_blas_dnrm2(residual) < resStop));
  int stop=0;
	
  if(cond1 || cond2 || cond3)
    {
      stop=1;
      gsl_vector_set_zero(newIndices);
      gsl_matrix_set_zero(Aactive);
      Aactive_size=0;
      gsl_matrix_set_zero(R);
      Rsize=0;
      gsl_vector_set_zero(removeIndices);
    } 
#ifdef DEBUG_STOPPING_CRITERION
  std::cout << "lambda - gammamin = " << lambda-gammamin << std::endl;
  std::cout << "lambda = "<< lambda << ", lambdaStop = " << lambdaStop << std::endl;
  std::cout << "resStop = " << resStop << ", residual norm =" << gsl_blas_dnrm2(residual) << std::endl;
#endif
		
  return stop;
}

// TESTE - VERIFIER AVEC MATLAB - WORKS////////////////////////////////////////////////////////////////////////////////
void LASSO_LARS_solver::InitializeCholeskyFactor(unsigned long &iter)
{
  unsigned long k;
	
  gsl_vector_set_zero(activeSet);
	
  for(k=0;k<newIndices->size;++k)
    {
      int flag=0;
      if(gsl_vector_get(newIndices,k)>0)
	{
	  double t[A->size1];
	  gsl_vector_view x=gsl_vector_view_array(t,A->size1);
	  updateChol(k,flag);
	  gsl_vector_set(activeSet,k,gsl_vector_get(newIndices,k));
					
	  gsl_matrix_get_col(&x.vector,A,k);
	  gsl_matrix_set_col(Aactive,Aactive_size,&x.vector);
	  Aactive_size++;
	  Rsize++;
					
	  if(verbose==1)
	    std::cout << "Iteration " << iter << ": Adding variable " << k << std::endl;
					
	  ++iter;
	}
    }
		
		
}

/////////////////////////////////////////////////////////////////////////////////
void LASSO_LARS_solver::ComputeLARSDirection(gsl_vector *dx,gsl_vector* ATv,gsl_vector *v)
{
  unsigned long k=0,j=0;
	
  gsl_matrix_view tmpR=gsl_matrix_submatrix(R,0,0,Rsize,Rsize);
  //gsl_matrix_view A1=gsl_matrix_submatrix(Aactive,0,0,Aactive->size1,Rsize);

  gsl_matrix *A1m=gsl_matrix_calloc(Aactive->size1,Rsize);
  gsl_vector *sign_corr_tab=gsl_vector_calloc(Rsize);
  gsl_matrix *Rcopy=gsl_matrix_calloc(Rsize,Rsize);
  gsl_vector *z=gsl_vector_calloc(Rsize);
  gsl_vector *z2=gsl_vector_calloc(Rsize);
	
  gsl_vector_set_zero(v);
  gsl_vector_set_zero(dx);
  gsl_vector_set_zero(ATv);
  gsl_matrix_memcpy(Rcopy,&tmpR.matrix);

	
  for(k=0;k<corr->size;++k)
    if(gsl_vector_get(activeSet,k)>0)
      gsl_vector_set(sign_corr_tab,(unsigned long)(gsl_vector_get(activeSet,k)-1),gsl_vector_get(corr,k)/fabs(gsl_vector_get(corr,k)));

#ifdef DEBUG_LARS_DIRECTION
  std::cout << "sign_corr_tab = [";
  for(j=0;j<Rsize;j++)
    std::cout << gsl_vector_get(sign_corr_tab,j) << " ";
  std::cout << "]^T"<< std::endl;
  DisplayPrivateMembers("R");
#endif

  gsl_blas_dtrsv(CblasUpper,CblasTrans,CblasNonUnit,Rcopy,sign_corr_tab);
  gsl_vector_memcpy(z,sign_corr_tab);
  gsl_blas_dtrsv(CblasUpper,CblasNoTrans,CblasNonUnit,Rcopy,sign_corr_tab);
  gsl_vector_memcpy(z2,sign_corr_tab);


  for(k=0;k<activeSet->size;++k)
    if(gsl_vector_get(activeSet,k) > 0)
      gsl_vector_set(dx,k,gsl_vector_get(z2,(unsigned long)gsl_vector_get(activeSet,k)-1));


#ifdef DEBUG_LARS_DIRECTION
  std::cout << "z = [";
  for(j=0;j<Rsize;j++)
    std::cout << gsl_vector_get(z,j) << " ";
  std::cout << "]^T"<< std::endl;
  std::cout << "z2 = [";
  for(j=0;j<Rsize;j++)
    std::cout << gsl_vector_get(z2,j) << " ";
  std::cout << "]^T"<< std::endl;
  std::cout << "sign_corr_tab = [";
  for(j=0;j<Rsize;j++)
    std::cout << gsl_vector_get(sign_corr_tab,j) << " ";
  std::cout << "]^T"<< std::endl;
  std::cout << "size(Aactive) = " << Aactive->size1 << "x" << Aactive_size << std::endl ;
  std::cout << "Rsize =" << Rsize << std::endl ;
  std::cout << "v =" << v->size << std::endl ;
#endif
       

  for(j=0;j<Aactive->size1;++j)
    for(k=0;k<Rsize;++k)
      gsl_matrix_set(A1m,j,k,gsl_matrix_get(Aactive,j,k));

  gsl_blas_dgemv(CblasNoTrans,1,A1m,z2,0,v);

#ifdef DEBUG_LARS_DIRECTION
  std::cout << "Coin 4 " << std::endl;
  DisplayVector(v);
#endif	

  gsl_blas_dgemv(CblasTrans,1,A,v,0,ATv);
#ifdef DEBUG_LARS_DIRECTION
  std::cout << "Coin 4 " << std::endl;
  DisplayVector(dx);
  DisplayVector(v);
  DisplayVector(ATv);
#endif	
	
  gsl_matrix_free(Rcopy);
  gsl_matrix_free(A1m);
  gsl_vector_free(sign_corr_tab);
  gsl_vector_free(z);
  gsl_vector_free(z2);
}
/////////////////////////////////////////////////////////////////////////////////
void LASSO_LARS_solver::FindFirstVector2Activate(double &gammaIc, int nonNegative, gsl_vector *ATv)
{

  unsigned long k;
	
  gsl_vector_set_all(inactiveSet,1);
	
  for(k=0;k<inactiveSet->size;++k)
    {
      if(gsl_vector_get(activeSet,k)>0)
	gsl_vector_set(inactiveSet,k,0);
      if(gsl_vector_get(collinearSet,k)>0)
	gsl_vector_set(inactiveSet,k,0);
    }	

  gsl_vector_set_zero(newIndices);
	
  if(gsl_vector_isnull(inactiveSet))
    {
      gammaIc=1;
    }
  else
    {	
      double gammaArr[inactiveSet->size];
      gsl_vector_view gV=gsl_vector_view_array(gammaArr,inactiveSet->size);
      double gammaArrnN[inactiveSet->size];
		
		
      for(k=0;k<inactiveSet->size;++k)
	{
	  gammaArr[k]=GSL_POSINF;
	  gammaArrnN[k]=GSL_POSINF;
	  if(gsl_vector_get(inactiveSet,k)>0)
	    {
	      gammaArr[k]=(lambda-gsl_vector_get(corr,k)) / (1+EPSILON-gsl_vector_get(ATv,k));
	      if(nonNegative==0)
		gammaArrnN[k]=(lambda+gsl_vector_get(corr,k)) / (1+EPSILON+gsl_vector_get(ATv,k));
				
	      if(gammaArr[k] < ZEROTOL)
		gammaArr[k]=GSL_POSINF;
	      if(gammaArrnN[k] < ZEROTOL)
		gammaArrnN[k]=GSL_POSINF;
				
	      gammaArr[k]=GSL_MIN(gammaArr[k],gammaArrnN[k]);
	    }
	}

#ifdef DEBUG_GAMMA
      std::cout << "gammaArr =\n [" ;
      for(k=0;k<inactiveSet->size;++k)
	std::cout << gammaArr[k] << " ";
      std::cout << "]^T" << std::endl;
#endif
		
      gammaIc=gsl_vector_min(&gV.vector);
				
      for(k=0;k<newIndices->size;++k)
	if(fabs(gammaArr[k]-gammaIc)<ZEROTOL)
	  gsl_vector_set(newIndices,k,gsl_vector_get(inactiveSet,k));
    }
}
/////////////////////////////////////////////////////////////////////////////////
void LASSO_LARS_solver::AugmentActiveSet(unsigned long &iter)
{
  unsigned long j;
	
  for(j=0;j<newIndices->size;++j)
    {
      if(gsl_vector_get(newIndices,j)>0)
	{
	  int flag=0;
	  ++iter;
	  if(verbose)
	    std::cout << "Iteration " << iter << ": Adding variable " << j << std::endl;
			
	  updateChol(j,flag);
#ifdef DEBUG_COLINEARITY
	  std::cout << "Colinearity flag = " << flag << std::endl; 
#endif 
			
	  if(flag)
	    {
	      gsl_vector_set(collinearSet,j,1);
	      if(verbose==1)
		std::cout << "Iteration " << iter << ": Variable " << j << " is collinear" << std::endl;
	    }
	  else
	    {
				
	      double t[Aactive->size1];
	      gsl_vector_view col=gsl_vector_view_array(t,Aactive->size1);
	      gsl_vector_set(activeSet,j,gsl_vector_max(activeSet)+1);
	      gsl_matrix_get_col(&col.vector,A,j);
	      gsl_matrix_set_col(Aactive,Aactive_size,&col.vector);
				
	      Aactive_size++;
				
	      //				for(k=0;k<Aactive->size2;++k)
	      //{
	      //	if(gsl_vector_get(activeSet,k)==1)
	      //	{
	      //		gsl_matrix_get_col(&col.vector,A,k);
	      //		gsl_matrix_set_col(Aactive,Aactive_size,&col.vector);
	      //		Aactive_size++;
	      //	}
	      //}
	    }
	}
    }
}
/////////////////////////////////////////////////////////////////////////////////
void LASSO_LARS_solver::ReduceActiveSet(unsigned long &iter)
{
  unsigned long j,k;
  //double t[A->size1];
	
  for(j=0;j<removeIndices->size;++j)
    {
      if(gsl_vector_get(removeIndices,j)>0)
	{
	  ++iter;


	  if(verbose)
	    std::cout << "Iteration " << iter << ": Dropping variable " << j << std::endl;
			
	  downdateChol((size_t)gsl_vector_get(activeSet,j)-1);


	  double setval=gsl_vector_get(activeSet,j);
	  gsl_vector_set(activeSet,j,0);
	  for(k=0;k<activeSet->size;++k)
	    if(gsl_vector_get(activeSet,k) > setval)
	      gsl_vector_set(activeSet,k,gsl_vector_get(activeSet,k)-1);
		
	  gsl_vector_set(beta,j,0);		 
	}
    }
  gsl_vector_set_zero(collinearSet);


#ifdef DEBUG_REDUCEACTIVESET
  DisplayPrivateMembers("Aactive");
#endif	
}

///////////////////////////////////////////////////////////////////////////////////
/////////////////////////// PRIMARY SOLVER ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

void LASSO_LARS_solver::SolveLASSO()
{
  int isLASSO, nonNegative,res,done;
  unsigned long iter=0;
  //unsigned long k,i;
  //	double lambda=0;
  double gammaI=0, gammaIc=0, gammaMin=0;
  // correlation vector
  gsl_vector *v=gsl_vector_calloc(A->size1);
  gsl_vector *dx = gsl_vector_calloc(A->size2);
  gsl_vector *ATv = gsl_vector_calloc(A->size2);
	
  gsl_vector_set_zero(v);
  // check which algorithm is used
  if(algType.size()==4)
    {
      isLASSO=0;
      nonNegative=0;
    }
  if(algType.size()==5)
    {
      isLASSO=1;
      nonNegative=0;
    }
  if(algType.size()==6)
    {
      isLASSO=0;
      nonNegative=1;
    }
  if(algType.size()==7)
    {
      isLASSO=1;
      nonNegative=1;
    }
	
  res=InitializeActiveSet(nonNegative);
  if(res==0)
    {std::cout << "Error";done=1;}
	
  if(res)
    done=CheckStoppingCriterion(-1000000000);

  if(done)
    std::cout << "Warning: No iterations were done: sparse approximation is already fine\n";
  else
    {
      gsl_vector_set_zero(activeSet);
      gsl_matrix_set_zero(Aactive);
      Aactive_size=0;
      gsl_matrix_set_zero(R);
      Rsize=0;
      InitializeCholeskyFactor(iter);
    }
	
  while(!done)
    {

      size_t tmpcorr=0;
      size_t tmpcorr_it=0;
      for(tmpcorr_it=0;tmpcorr_it<activeSet->size;++tmpcorr_it)
	if(gsl_vector_get(activeSet,tmpcorr_it)==1)
	  tmpcorr=tmpcorr_it;

      if(nonNegative)
      	lambda=gsl_vector_get(corr,tmpcorr);
      else
      	lambda=fabs(gsl_vector_get(corr,tmpcorr));
		
      // if(nonNegative)
      // 	lambda=gsl_vector_get(corr,gsl_vector_max_index(activeSet));
      // else
      // 	lambda=fabs(gsl_vector_get(corr,gsl_vector_max_index(activeSet)));
		
      ComputeLARSDirection(dx,ATv,v);

// #ifdef DEBUG_MAIN_FUNCTION
//       DisplayVector(dx);
//       DisplayVector(v);
//       DisplayVector(ATv);
// #endif


		
      // For LASSO,we find the first active vector to violate sign constraint
      gammaI=GSL_POSINF;
      gsl_vector_set_zero(removeIndices);
      if(isLASSO)
	{
	  double t[activeSet->size];
	  unsigned long k;
			
	  for(k=0;k<activeSet->size;++k)
	    {
	      if(fabs(gsl_vector_get(dx,k))>ZEROTOL)
		{
		  t[k]=-gsl_vector_get(beta,k) / gsl_vector_get(dx,k);
		  if(int(t[k] < gammaI) && (t[k]>ZEROTOL))
		    gammaI = t[k];
		}
	      else
		{
		  t[k]=0;
		}
				
	    }		

	  for(k=0;k<activeSet->size;++k)
	    if(t[k]==gammaI)
	      gsl_vector_set(removeIndices,k,1);
	}
		
      FindFirstVector2Activate(gammaIc,nonNegative,ATv);
#ifdef DEBUG_MAIN_FUNCTION
      std::cout << "GammaI = " << gammaI << std::endl;
      std::cout << "GammaIc = " << gammaIc << std::endl;
#endif
		
      gammaMin=GSL_MIN(gammaIc,gammaI);
	
      // Compute the next LARS step
      gsl_vector_scale(dx,gammaMin);
      gsl_vector_scale(v,-gammaMin);
      gsl_vector_scale(ATv,-gammaMin);
      gsl_vector_add(beta,dx);
      gsl_vector_add(residual,v);
      gsl_vector_add(corr,ATv);
		
      done=CheckStoppingCriterion(gammaMin);
		
      // augment active set with new columns
      if((gammaIc <= gammaI) && (!gsl_vector_isnull(newIndices)))
	AugmentActiveSet(iter);

      // reduce active set by withdrawing collinear columns
      // 14.9.14: BUG IN THE REDUCTION OF THE ACTIVE SET
      if(gammaI<=gammaIc)
	ReduceActiveSet(iter);
		
      if(iter>=maxIters)
	done=1;
		
      if(verbose)
	std::cout << "lambda = " << lambda << ", |I| = " << gsl_vector_max(activeSet) << ", residual norm = " << gsl_blas_dnrm2(residual) << std::endl;

#ifdef DEBUG_MAIN_FUNCTION
      {
	// size_t i;//,iter=0;
	std::cout << "############# ITERATION  " << iter << " ################"<< std::endl;
	// std::cout << "v=[" ;
	// for(i=0;i<v->size;++i)
	//   std::cout << gsl_vector_get(v,i) << " ";
	// std::cout << "]^T" << std::endl << std::endl  ;
	// // dx
	// std::cout << "dx=[" ;
	// for(i=0;i<dx->size;++i)
	//   std::cout << gsl_vector_get(dx,i) << " ";
	// std::cout << "]^T" << std::endl << std::endl  ;
	// // ATv
	// std::cout << "ATv=[" ;
	// for(i=0;i<ATv->size;++i)
	//   std::cout << gsl_vector_get(ATv,i) << " ";
	// std::cout << "]^T" << std::endl << std::endl  ;
	//	DisplayPrivateMembers("Aactive");
	//	DisplayVector(beta);
	std::cout << "#######################################################" << std::endl <<std::endl ;
      }
#endif	



    }


  gsl_vector_free(v);
  gsl_vector_free(dx);
  gsl_vector_free(ATv);

}

void LASSO_LARS_solver::ResetRegressor()
{
  gsl_matrix_set_zero(Aactive);
  Aactive_size=0;
  gsl_vector_set_zero(newIndices);
  gsl_vector_set_zero(removeIndices);
  gsl_vector_set_zero(activeSet);
  gsl_vector_set_zero(inactiveSet);
  gsl_vector_set_zero(collinearSet);
  gsl_matrix_set_zero(R);
  Rsize=0;
  gsl_vector_set_zero(corr);
  lambda=0;
  gsl_vector_set_zero(beta);
  gsl_vector_memcpy(residual,y);
}
