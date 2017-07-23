#include "FISTA_solver.h"

#define HISTORY_SIZE 10
//#define DEBUG_FISTA 1
//#define RUN_FISTA 1

FISTA_solver::FISTA_solver()
{
  A=gsl_matrix_calloc(10,10);
  gsl_matrix_set_identity(A);
  y=gsl_vector_calloc(10);
  regType="lasso";
  groups=gsl_vector_calloc(10);
  maxIters=10000;
  verbose=1;
  beta=gsl_vector_calloc(10);
  sigma0=0;
  tol=0.000001;
}

FISTA_solver::FISTA_solver(gsl_matrix *a,gsl_vector *b, std::string c, gsl_vector *d, unsigned long f, int g) : A(a),y(b),regType(c),groups(d),maxIters(f),verbose(g)
{
  beta=gsl_vector_calloc(A->size2);
  sigma0=0;
  tol=0.000001;
}

FISTA_solver::FISTA_solver(gsl_matrix *a,gsl_vector *b, std::string c, gsl_vector *d, unsigned long f, int g, gsl_vector *i, double j, double k):  A(a),y(b),regType(c),groups(d),maxIters(f),verbose(g),beta(i),sigma0(j),tol(k)
{
}

FISTA_solver::~FISTA_solver()
{
  //gsl_matrix_free(A);
  //gsl_vector_free(y);
  //gsl_vector_free(beta);
  //gsl_vector_free(groups);
}


void FISTA_solver::EstimateSigma0(double tolest)
{
  gsl_vector *Ax = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size2);
  size_t m=0,n=0;  

  double e=0., cnt=0., e0 =0., tmp=0. ;

  for(m=0;m<A->size2;m++)
    {
      for(n=0;n<A->size1;++n)
	{
	  tmp = 0.;
	  tmp += fabs(gsl_matrix_get(A,n,m));
	}
      gsl_vector_set(x,m,tmp);      
    }
  
  e = gsl_blas_dnrm2(x);
  gsl_vector_scale(x,1./e);
  
  while ( (fabs(e-e0)) > (tolest * e))
    {
      e0 = e;
      // Sx = S*x
      gsl_blas_dgemv(CblasNoTrans, 1., A, x, 0., Ax);
      // e = norm(Sx)
      e = gsl_blas_dnrm2(Ax);
      // x = S'*Sx
      gsl_blas_dgemv(CblasTrans, 1., A, Ax, 0., x);

      double tmp=gsl_blas_dnrm2(x);      
      gsl_vector_scale(x,(double)1./tmp);
      
      cnt+=1;
    }
  
  gsl_vector_free(x);  
  gsl_vector_free(Ax);

  sigma0 = e*e / (double)A->size1;
}


void FISTA_solver::GetResult(gsl_vector *dest) const
{
  if(dest->size != beta->size)
    std::cout << "ERROR IN FISTA_solver::GetResult : dest has not the same size as beta" << std::endl;
  else
    gsl_vector_memcpy(dest,beta);
}

void FISTA_solver::Solve(double tau)
{
  unsigned long n_iter=0,stop=0,n;
  double step,tau_s,t;
  gsl_matrix *XT=gsl_matrix_calloc(A->size2,A->size1);
  gsl_vector *h=gsl_vector_calloc(beta->size);
  gsl_vector *E_prevs=gsl_vector_calloc(HISTORY_SIZE);
  gsl_vector *Xb=gsl_vector_calloc(A->size1);
  gsl_vector *Xh=gsl_vector_calloc(A->size1);

#ifdef RUN_FISTA
      std::cout << "Computes normest(AAT)..." << std::endl;
#endif

  if(sigma0==0)
    EstimateSigma0(tol);

  step=sigma0;
  tau_s=tau/step;
  t=1;
#ifdef DEBUG_FISTA
  std::cout << std::setprecision(16) << "step = " << step << std::endl;
  std::cout << std::setprecision(16) << "tau_s = " << tau_s << std::endl;
#endif

  gsl_matrix_transpose_memcpy(XT,A);
  gsl_matrix_scale(XT,1./(step*(double)A->size1));
  gsl_vector_memcpy(h,beta);
  gsl_vector_set_all(E_prevs,GSL_POSINF);
  gsl_blas_dgemv(CblasNoTrans,1.,A,beta,1,Xb);
  gsl_vector_memcpy(Xh,Xb);

#ifdef DEBUG_FISTA
  std::cout << "Xb = [";
  for(n=0;n<Xb->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(Xb,n) << " " ;
  std::cout << "]^T" << std::endl;
  std::cout << "beta = [";
  for(n=0;n<beta->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(beta,n) << " " ;
  std::cout << "]^T" << std::endl;
  std::cout << "Xh = [";
  for(n=0;n<Xh->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(Xh,n) << " " ;
  std::cout << "]^T" << std::endl;
  std::cout << "h = [";
  for(n=0;n<h->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(h,n) << " " ;
  std::cout << "]^T" << std::endl;
#endif

#ifdef RUN_FISTA
      std::cout << "Enters the while..." << std::endl;
#endif

  while((n_iter<maxIters) && !stop)
    {
      gsl_vector *beta_prev=gsl_vector_calloc(beta->size);
      gsl_vector *beta_noproj=gsl_vector_calloc(beta->size);
      gsl_vector *Xb_prev=gsl_vector_calloc(Xb->size);
      double t_new=0.5*(1+sqrt(1+4*t*t));
      double E=0;

      gsl_vector_memcpy(Xb_prev,Xb);
      gsl_vector_memcpy(beta_prev,beta);
      ++n_iter;

#ifdef RUN_FISTA
      std::cout << "Iteration " << n_iter << " in the while..." << std::endl;
#endif

      // gradient step
      gsl_vector_sub(Xh,y);
      gsl_blas_dgemv(CblasNoTrans,-1,XT,Xh,1,h); // resulting gradient in h
      // soft-thresholding
      for(n=0;n<beta->size;++n)
	if(tau_s > fabs(gsl_vector_get(h,n)))
	  gsl_vector_set(beta,n,0);
	else
	  gsl_vector_set(beta,n,gsl_vector_get(h,n)*(1-tau_s/fabs(gsl_vector_get(h,n))));

      // updates
      gsl_blas_dgemv(CblasNoTrans,1,A,beta,0,Xb);

#ifdef DEBUG_FISTA
      std::cout << "Xb = [";
      for(n=0;n<Xb->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(Xb,n) << " " ;
      std::cout << "]^T" << std::endl;
      std::cout << "beta = [";
      for(n=0;n<beta->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(beta,n) << " " ;
      std::cout << "]^T" << std::endl;
#endif

      gsl_vector_sub(beta_prev,beta);
      gsl_vector_scale(beta_prev,(1-t)/t_new);
      gsl_vector_add(beta_prev,beta);
      gsl_vector_memcpy(h,beta_prev);
      gsl_vector_sub(Xb_prev,Xb);
      gsl_vector_scale(Xb_prev,(1-t)/t_new);
      gsl_vector_memcpy(Xh,Xb);
      gsl_vector_add(Xb_prev,Xh);

      //      gsl_vector_memcpy(Xb_prev,Xb);
      gsl_vector_memcpy(Xh,Xb_prev);
      t=t_new;

#ifdef DEBUG_FISTA
      std::cout << "Xh = [";
      for(n=0;n<Xh->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(Xh,n) << " " ;
      std::cout << "]^T" << std::endl;
      std::cout << "h = [";
      for(n=0;n<h->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(h,n) << " " ;
      std::cout << "]^T" << std::endl;
      std::cout << std::setprecision(16) << "new t = " << t << std::endl;
#endif

      // empirical error
      gsl_vector_memcpy(Xb_prev,Xb);
      gsl_vector_sub(Xb_prev,y);
      E=2*tau*gsl_blas_dasum(beta)+pow(gsl_blas_dnrm2(Xb_prev),2)/((double)A->size1);

      // Stopping Criterion
      gsl_vector_set(E_prevs,n_iter%HISTORY_SIZE,E);
      if(n_iter>10)
	{
	  double mean=0;
	  for(n=0;n<E_prevs->size;++n)
	    mean += gsl_vector_get(E_prevs,n)/HISTORY_SIZE;
	  if(mean-E < mean*tol)
	    stop=1;
	}

#ifdef DEBUG_FISTA
      std::cout << "E_prevs = [";
      for(n=0;n<E_prevs->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(E_prevs,n) << " " ;
      std::cout << "]^T" << std::endl;
#endif

      gsl_vector_free(beta_prev);
      gsl_vector_free(beta_noproj);
      gsl_vector_free(Xb_prev);
    }

  // Deallocation
  gsl_matrix_free(XT);
  gsl_vector_free(h);
  gsl_vector_free(E_prevs);
  gsl_vector_free(Xb);
  gsl_vector_free(Xh);
}

void FISTA_solver::ReinitializeValues()
{
  gsl_vector_set_zero(beta);
  sigma0=0;
}

void FISTA_solver::Solve(double tau, double smooth_par)
{
  unsigned long n_iter=0,stop=0,n;
  double step,tau_s,mu,mu_s,t;
  gsl_matrix *XT=gsl_matrix_calloc(A->size2,A->size1);
  gsl_vector *h=gsl_vector_calloc(beta->size);
  gsl_vector *E_prevs=gsl_vector_calloc(HISTORY_SIZE);
  gsl_vector *Xb=gsl_vector_calloc(A->size1);
  gsl_vector *Xh=gsl_vector_calloc(A->size1);


  if(sigma0==0)
    EstimateSigma0(tol);
  
  mu=smooth_par*sigma0;
  step=sigma0+mu;
  tau_s=tau/step;
  mu_s=mu/step;
  t=1;
#ifdef DEBUG_FISTA
  std::cout << std::setprecision(16) << "step = " << step << std::endl;
  std::cout << std::setprecision(16) << "tau_s = " << tau_s << std::endl;
  std::cout << std::setprecision(16) << "mu_s = " << mu_s << std::endl;
#endif

  gsl_matrix_transpose_memcpy(XT,A);
  gsl_matrix_scale(XT,1./(step*(double)A->size1));
  gsl_vector_memcpy(h,beta);
  gsl_vector_set_all(E_prevs,GSL_POSINF);
  gsl_blas_dgemv(CblasNoTrans,1.,A,beta,1,Xb);
  gsl_vector_memcpy(Xh,Xb);

#ifdef DEBUG_FISTA
  std::cout << "Xb = [";
  for(n=0;n<Xb->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(Xb,n) << " " ;
  std::cout << "]^T" << std::endl;
  std::cout << "beta = [";
  for(n=0;n<beta->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(beta,n) << " " ;
  std::cout << "]^T" << std::endl;
  std::cout << "Xh = [";
  for(n=0;n<Xh->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(Xh,n) << " " ;
  std::cout << "]^T" << std::endl;
  std::cout << "h = [";
  for(n=0;n<h->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(h,n) << " " ;
  std::cout << "]^T" << std::endl;
#endif

  while((n_iter<maxIters) && !stop)
    {
      gsl_vector *beta_prev=gsl_vector_calloc(beta->size);
      gsl_vector *beta_noproj=gsl_vector_calloc(beta->size);
      gsl_vector *Xb_prev=gsl_vector_calloc(Xb->size);
      double t_new=0.5*(1+sqrt(1+4*t*t));
      double E=0;

      gsl_vector_memcpy(Xb_prev,Xb);
      gsl_vector_memcpy(beta_prev,beta);
      ++n_iter;

      // gradient step
      gsl_vector_sub(Xh,y);
      gsl_blas_dgemv(CblasNoTrans,-1,XT,Xh,1-mu_s,h); // resulting gradient in h
      // soft-thresholding
      for(n=0;n<beta->size;++n)
	if(tau_s > fabs(gsl_vector_get(h,n)))
	  gsl_vector_set(beta,n,0);
	else
	  gsl_vector_set(beta,n,gsl_vector_get(h,n)*(1-tau_s/fabs(gsl_vector_get(h,n))));

      // updates
      gsl_blas_dgemv(CblasNoTrans,1,A,beta,0,Xb);

#ifdef DEBUG_FISTA
      std::cout << "Xb = [";
      for(n=0;n<Xb->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(Xb,n) << " " ;
      std::cout << "]^T" << std::endl;
      std::cout << "beta = [";
      for(n=0;n<beta->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(beta,n) << " " ;
      std::cout << "]^T" << std::endl;
#endif

      gsl_vector_sub(beta_prev,beta);
      gsl_vector_scale(beta_prev,(1-t)/t_new);
      gsl_vector_add(beta_prev,beta);
      gsl_vector_memcpy(h,beta_prev);
      gsl_vector_sub(Xb_prev,Xb);
      gsl_vector_scale(Xb_prev,(1-t)/t_new);
      gsl_vector_memcpy(Xh,Xb);
      gsl_vector_add(Xb_prev,Xh);

      //      gsl_vector_memcpy(Xb_prev,Xb);
      gsl_vector_memcpy(Xh,Xb_prev);
      t=t_new;

#ifdef DEBUG_FISTA
      std::cout << "Xh = [";
      for(n=0;n<Xh->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(Xh,n) << " " ;
      std::cout << "]^T" << std::endl;
      std::cout << "h = [";
      for(n=0;n<h->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(h,n) << " " ;
      std::cout << "]^T" << std::endl;
      std::cout << std::setprecision(16) << "new t = " << t << std::endl;
#endif

      // empirical error
      gsl_vector_memcpy(Xb_prev,Xb);
      gsl_vector_sub(Xb_prev,y);
      E=2*tau*gsl_blas_dasum(beta)+pow(gsl_blas_dnrm2(Xb_prev),2)/((double)A->size1)+mu*pow(gsl_blas_dnrm2(beta),2);

      // Stopping Criterion
      gsl_vector_set(E_prevs,n_iter%HISTORY_SIZE,E);
      if(n_iter>10)
	{
	  double mean=0;
	  for(n=0;n<E_prevs->size;++n)
	    mean += gsl_vector_get(E_prevs,n)/HISTORY_SIZE;
	  if(mean-E < mean*tol)
	    stop=1;
	}

#ifdef DEBUG_FISTA
      std::cout << "E_prevs = [";
      for(n=0;n<E_prevs->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(E_prevs,n) << " " ;
      std::cout << "]^T" << std::endl;
#endif

      gsl_vector_free(beta_prev);
      gsl_vector_free(beta_noproj);
      gsl_vector_free(Xb_prev);
    }

  // Deallocation
  gsl_matrix_free(XT);
  gsl_vector_free(h);
  gsl_vector_free(E_prevs);
  gsl_vector_free(Xb);
  gsl_vector_free(Xh);
}

void FISTA_solver::GroupSolve(double tau)
{
  unsigned long n_iter=0,stop=0,m,n;
  double step,tau_s,t;
  gsl_matrix *XT=gsl_matrix_calloc(A->size2,A->size1);
  gsl_vector *h=gsl_vector_calloc(beta->size);
  gsl_vector *E_prevs=gsl_vector_calloc(HISTORY_SIZE);
  gsl_vector *Xb=gsl_vector_calloc(A->size1);
  gsl_vector *Xh=gsl_vector_calloc(A->size1);
  size_t nb_groups = (size_t)gsl_vector_max(groups); // nb of groups ;
  gsl_vector *group_norms = gsl_vector_calloc(nb_groups);


  if(sigma0==0)
    EstimateSigma0(tol);
  
  step=sigma0;
  tau_s=tau/step;
  t=1;
#ifdef DEBUG_FISTA
  std::cout << std::setprecision(16) << "step = " << step << std::endl;
  std::cout << std::setprecision(16) << "tau_s = " << tau_s << std::endl;
#endif

  gsl_matrix_transpose_memcpy(XT,A);
  gsl_matrix_scale(XT,1./(step*(double)A->size1));
  gsl_vector_memcpy(h,beta);
  gsl_vector_set_all(E_prevs,GSL_POSINF);
  gsl_blas_dgemv(CblasNoTrans,1.,A,beta,1,Xb);
  gsl_vector_memcpy(Xh,Xb);

#ifdef DEBUG_FISTA
  std::cout << "Xb = [";
  for(n=0;n<Xb->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(Xb,n) << " " ;
  std::cout << "]^T" << std::endl;
  std::cout << "beta = [";
  for(n=0;n<beta->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(beta,n) << " " ;
  std::cout << "]^T" << std::endl;
  std::cout << "Xh = [";
  for(n=0;n<Xh->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(Xh,n) << " " ;
  std::cout << "]^T" << std::endl;
  std::cout << "h = [";
  for(n=0;n<h->size;++n)
    std::cout << std::setprecision(16) << gsl_vector_get(h,n) << " " ;
  std::cout << "]^T" << std::endl;
#endif

  while((n_iter<maxIters) && !stop)
    {
      gsl_vector *beta_prev=gsl_vector_calloc(beta->size);
      gsl_vector *beta_noproj=gsl_vector_calloc(beta->size);
      gsl_vector *Xb_prev=gsl_vector_calloc(Xb->size);
      double t_new=0.5*(1+sqrt(1+4*t*t));
      double E=0;

      gsl_vector_memcpy(Xb_prev,Xb);
      gsl_vector_memcpy(beta_prev,beta);
      ++n_iter;

      // gradient step
      gsl_vector_sub(Xh,y);
      gsl_blas_dgemv(CblasNoTrans,-1,XT,Xh,1,h); // resulting gradient in h
      // soft-thresholding for each group
      for(m=1;m<=nb_groups;++m)
	{
	  double nrm=0;
	  // computes the block norm
	  for(n=0;n<beta->size;++n)
	    if((size_t)gsl_vector_get(groups,n)==m)
	      nrm += gsl_vector_get(h,n)*gsl_vector_get(h,n);

	  gsl_vector_set(group_norms,m-1,sqrt(nrm));

	  // updates the group
	  for(n=0;n<beta->size;++n)
	    if((size_t)gsl_vector_get(groups,n)==m)
	      gsl_vector_set(beta,n,GSL_MAX(1-tau_s/sqrt(nrm),0)*gsl_vector_get(h,n));
	}

      // updates
      gsl_blas_dgemv(CblasNoTrans,1,A,beta,0,Xb);

#ifdef DEBUG_FISTA
      std::cout << "Xb = [";
      for(n=0;n<Xb->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(Xb,n) << " " ;
      std::cout << "]^T" << std::endl;
      std::cout << "beta = [";
      for(n=0;n<beta->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(beta,n) << " " ;
      std::cout << "]^T" << std::endl;
#endif

      gsl_vector_sub(beta_prev,beta);
      gsl_vector_scale(beta_prev,(1-t)/t_new);
      gsl_vector_add(beta_prev,beta);
      gsl_vector_memcpy(h,beta_prev);
      gsl_vector_sub(Xb_prev,Xb);
      gsl_vector_scale(Xb_prev,(1-t)/t_new);
      gsl_vector_memcpy(Xh,Xb);
      gsl_vector_add(Xb_prev,Xh);

      //      gsl_vector_memcpy(Xb_prev,Xb);
      gsl_vector_memcpy(Xh,Xb_prev);
      t=t_new;

#ifdef DEBUG_FISTA
      std::cout << "Xh = [";
      for(n=0;n<Xh->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(Xh,n) << " " ;
      std::cout << "]^T" << std::endl;
      std::cout << "h = [";
      for(n=0;n<h->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(h,n) << " " ;
      std::cout << "]^T" << std::endl;
      std::cout << std::setprecision(16) << "new t = " << t << std::endl;
#endif

      // empirical error
      gsl_vector_memcpy(Xb_prev,Xb);
      gsl_vector_sub(Xb_prev,y);
      E=2*tau*gsl_blas_dasum(group_norms)+pow(gsl_blas_dnrm2(Xb_prev),2)/((double)A->size1);

      // Stopping Criterion
      gsl_vector_set(E_prevs,n_iter%HISTORY_SIZE,E);
      if(n_iter>10)
	{
	  double mean=0;
	  for(n=0;n<E_prevs->size;++n)
	    mean += gsl_vector_get(E_prevs,n)/HISTORY_SIZE;
	  if(mean-E < mean*tol)
	    stop=1;
	}

#ifdef DEBUG_FISTA
      std::cout << "E_prevs = [";
      for(n=0;n<E_prevs->size;++n)
	std::cout << std::setprecision(16) << gsl_vector_get(E_prevs,n) << " " ;
      std::cout << "]^T" << std::endl;
#endif

      gsl_vector_free(beta_prev);
      gsl_vector_free(beta_noproj);
      gsl_vector_free(Xb_prev);
    }

  // Deallocation
  gsl_matrix_free(XT);
  gsl_vector_free(h);
  gsl_vector_free(E_prevs);
  gsl_vector_free(Xb);
  gsl_vector_free(Xh);
  gsl_vector_free(group_norms);
}
