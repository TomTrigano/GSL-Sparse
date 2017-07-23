#include "Bayesian_LASSO.h"

Bayesian_LASSO::Bayesian_LASSO() : Bayesian_Regressor::Bayesian_Regressor()
{
}

Bayesian_LASSO::Bayesian_LASSO(gsl_vector* b,gsl_matrix* a, size_t c, size_t d, size_t e, double f): Bayesian_Regressor(b,a,c,d,e,f)
{
    SparsityParameter=0;
}

Bayesian_LASSO::~Bayesian_LASSO()
{
    //dtor
}

Bayesian_LASSO::Bayesian_LASSO(const Bayesian_LASSO& other)
{
    //copy ctor
}

Bayesian_LASSO& Bayesian_LASSO::operator=(const Bayesian_LASSO& rhs)
{
    if (this == &rhs) return *this; // handle self assignment
    //assignment operator
    return *this;
}

void Bayesian_LASSO::ComputeAuxiliary(const gsl_vector* Invtau)
{
#ifdef DEBUG_BAYESIAN_LASSO_VALS
for(size_t n=0;n<Invtau->size;++n)
	std::cout << gsl_vector_get(Invtau,n) << " ";
std::cout << std::endl;
#endif
    // Creates the diagonal matrix
    gsl_matrix *tmp = gsl_matrix_alloc(Invtau->size, Invtau->size);
    gsl_vector_view diag = gsl_matrix_diagonal(tmp);
    gsl_matrix_set_all(tmp, 0.0); //or whatever number you like
    gsl_vector_memcpy(&diag.vector, Invtau);
    gsl_blas_dgemm (CblasTrans,CblasNoTrans,1,A,A,1,tmp);
    gsl_linalg_cholesky_decomp(tmp);
    gsl_linalg_cholesky_invert(tmp);
    gsl_matrix_memcpy(Auxiliary,tmp);
    gsl_matrix_free(tmp);
}

void Bayesian_LASSO::SolveWithFixedSparsityParameter(double valSparsityParameter)
{

    SparsityParameter=valSparsityParameter;
    Auxiliary=gsl_matrix_calloc(A->size2,A->size2);
    sigma2_estimate=gsl_vector_calloc(NbIterations+BurnIn);

    // Initialization of the random number generator
    const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_mt19937;
    r = gsl_rng_alloc (T);
    gsl_rng_set (r,seed);

    // Initialization of the class parameters and auxiliary vectors and matrices
    gsl_matrix_set_zero(Auxiliary);
    gsl_matrix_set_zero(ResultsBeta);
    gsl_vector_set_zero(beta);
    gsl_vector_set_zero(sigma2_estimate);
    gsl_vector* tmpbeta=gsl_vector_calloc(beta->size);
    gsl_vector* ATy=gsl_vector_calloc(beta->size);
    gsl_vector* tmpinvtau=gsl_vector_calloc(beta->size);
    gsl_vector* residue=gsl_vector_calloc(y->size);
    gsl_matrix* tmpmat=gsl_matrix_calloc(A->size2,A->size2);

    // Initialization before iterations
    //! \todo Check if this initialization is actually needed

    // beta = 0.01*(XTX+diag(p))^{-1}XTy
    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1,A,A,0,tmpmat);
    gsl_blas_dgemv(CblasTrans,1,A,y,0,tmpbeta);
    gsl_vector_memcpy(ATy,tmpbeta);
    for(size_t n=0;n<tmpmat->size2;++n)
        gsl_matrix_set(tmpmat,n,n,gsl_matrix_get(tmpmat,n,n)+(double)(A->size2));
    gsl_linalg_cholesky_decomp(tmpmat);
    gsl_linalg_cholesky_svx(tmpmat,tmpbeta);
    gsl_vector_scale(tmpbeta,0.01);

    // residue = y-A.beta
    gsl_vector_memcpy(residue,y);
    gsl_blas_dgemv(CblasNoTrans,-1,A,tmpbeta,1,residue);

    // sigma2 = norm(residue)/n
    double tmpsigma2;
    gsl_blas_ddot(residue,residue,&tmpsigma2);
    tmpsigma2 = tmpsigma2/((double)(y->size));

    // tau =1/ beta^2
    for(size_t n=0;n<tmpinvtau->size;++n)
        gsl_vector_set(tmpinvtau,n,1./gsl_pow_2(gsl_vector_get(tmpbeta,n)));

    std::cout << ">>>>> Gibbs Sampler iterations, please wait <<<<<" << std::endl;
    gsl_vector* MultinormalMean=gsl_vector_calloc(tmpbeta->size);
    gsl_matrix* MultinormalCov=gsl_matrix_calloc(tmpbeta->size,tmpbeta->size);
    for(size_t n=0;n<ResultsBeta->size2;++n)
    {
#ifdef DEBUG_BAYESIAN_LASSO
        std::cout << "Iteration " << n << "/" << NbIterations+BurnIn << std::endl;
#endif // DEBUG_BAYESIAN_LASSO

        // Resample from beta
        ComputeAuxiliary(tmpinvtau); // Calculate A^{-1}, where A=XTX+tmpinvtau
        gsl_blas_dgemv(CblasNoTrans,1,Auxiliary,ATy,0,MultinormalMean);
        gsl_matrix_memcpy(MultinormalCov,Auxiliary);
        gsl_matrix_scale(MultinormalCov,tmpsigma2);
        DrawMultivariateNormal(r,MultinormalMean,MultinormalCov,tmpbeta);
        gsl_matrix_set_col(ResultsBeta,n,tmpbeta);
#ifdef DEBUG_BAYESIAN_LASSO_VALS
std::cout << "beta= [" ;
for(size_t nd=0;nd<tmpbeta->size;++nd)
	std::cout << gsl_vector_get(tmpbeta,nd) << " ";
std::cout << "]^T"<< std::endl;
#endif


        // Resample from sigma2
        double shape=0.5*((double)(y->size)-1+(double)(A->size2));
        // residue = y-A.beta
        gsl_vector_memcpy(residue,y);
        gsl_blas_dgemv(CblasNoTrans,-1,A,tmpbeta,1,residue);
        // scale
        double scale;
        for(size_t m=0;m<tmpbeta->size;++m)
            gsl_vector_set(tmpbeta,m,gsl_vector_get(tmpbeta,m)*sqrt(gsl_vector_get(tmpinvtau,m)));
        gsl_blas_ddot(tmpbeta,tmpbeta,&scale);
        scale = 0.5*gsl_pow_2(gsl_blas_dnrm2(residue))+0.5*scale;
        gsl_vector_set(sigma2_estimate,n,DrawInverseGamma(r,shape,scale));
#ifdef DEBUG_BAYESIAN_LASSO_VALS
std::cout << "sigma2_estimate= " << gsl_vector_get(sigma2_estimate,n) << std::endl;
#endif


        // Resample 1/tau2
        gsl_matrix_get_col(tmpbeta,ResultsBeta,n);
#ifdef DEBUG_BAYESIAN_LASSO_VALS
std::cout << "beta= [" ;
for(size_t nd=0;nd<ResultsBeta->size1;++nd)
	std::cout << gsl_vector_get(tmpbeta,nd) << " ";
std::cout << "]^T"<< std::endl;
#endif
        for(size_t m=0;m<tmpinvtau->size;++m)
        {
            double mu=sqrt(SparsityParameter*SparsityParameter+gsl_vector_get(sigma2_estimate,n)) / fabs(gsl_vector_get(tmpbeta,m)) ;
	    double val=DrawInverseNormal(r,mu,SparsityParameter*SparsityParameter);
#ifdef DEBUG_BAYESIAN_LASSO_VALS
std::cout << "mu_num = " << sqrt(SparsityParameter*SparsityParameter+gsl_vector_get(sigma2_estimate,n)) << ", mu_den = " << fabs(gsl_vector_get(tmpbeta,m)) << std::endl;
std::cout << "mu = " << mu << ", qval_" << m << " = " << val << std::endl;
#endif
            gsl_vector_set(tmpinvtau,m,val);
        }

#ifdef DEBUG_BAYESIAN_LASSO_VALS
std::cout << "1/tau= [" ;
for(size_t nd=0;nd<tmpinvtau->size;++nd)
	std::cout << gsl_vector_get(tmpinvtau,nd) << " ";
std::cout << "]^T"<< std::endl;
#endif

#ifdef DEBUG_BAYESIAN_LASSO
std::cout << "Value of beta stored in column "<< n << "/" << ResultsBeta->size2 << std::endl;
#endif
    }
    std::cout << ">>>>> Done. <<<<<" << std::endl;
    gsl_vector_free(MultinormalMean);
    gsl_matrix_free(MultinormalCov);

    gsl_vector_free(ATy);
    gsl_matrix_free(tmpmat);
    gsl_vector_free(tmpbeta);
    gsl_vector_free(tmpinvtau);
    gsl_vector_free(residue);
    gsl_rng_free(r);
}
