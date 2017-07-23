#include "Bayesian_Regressor.h"

Bayesian_Regressor::Bayesian_Regressor()
{
    A=gsl_matrix_calloc(10,100);
    y=gsl_vector_calloc(10);
    seed=1;
    NbIterations=10000;
    BurnIn=1000;
    sigma2=0.05;
    ResultsBeta=gsl_matrix_calloc(A->size2,NbIterations+BurnIn);
    beta=gsl_vector_calloc(A->size2);
}

Bayesian_Regressor::Bayesian_Regressor(gsl_vector* b, gsl_matrix* a, size_t c, size_t d, size_t e, double f) : y(b),A(a),seed(c),NbIterations(d),BurnIn(e),sigma2(f)
{
    ResultsBeta=gsl_matrix_calloc(A->size2,NbIterations+BurnIn);
    beta=gsl_vector_calloc(A->size2);
}

Bayesian_Regressor::~Bayesian_Regressor()
{
    //dtor
}

Bayesian_Regressor::Bayesian_Regressor(const Bayesian_Regressor& other)
{
    //copy ctor
}

Bayesian_Regressor& Bayesian_Regressor::operator=(const Bayesian_Regressor& rhs)
{
    if (this == &rhs) return *this; // handle self assignment
    //assignment operator
    return *this;
}

void Bayesian_Regressor::DrawMultivariateNormal(const gsl_rng *r, const gsl_vector *mean, const gsl_matrix *var, gsl_vector *result)
{
size_t k;
size_t n=mean->size;
gsl_matrix *work = gsl_matrix_alloc(n,n);

gsl_matrix_memcpy(work,var);
gsl_linalg_cholesky_decomp(work);

for(k=0; k<n; k++)
	gsl_vector_set( result, k, gsl_ran_ugaussian(r) );

gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, work, result);
gsl_vector_add(result,mean);

gsl_matrix_free(work);
}

double Bayesian_Regressor::DrawInverseGamma(const gsl_rng *r, const double shape, const double scale)
{
return 1./  gsl_ran_gamma (r,shape,1./scale);
}

double Bayesian_Regressor::DrawInverseNormal(const gsl_rng *r, const double mu, const double lambda)
{
    double nu=gsl_ran_gaussian(r,1);
    double yig=nu*nu;
    double x= mu + (mu*mu*yig - mu*sqrt(4*mu*lambda*yig+mu*mu*yig*yig))/(2*lambda);
    double z=gsl_ran_flat(r,0,1);
    if(z*(x+mu) < mu)
        return x;
    else
        return mu*mu/x;
}

void Bayesian_Regressor::MeanEstimate()
{
    gsl_vector_set_zero(beta);
    for(size_t n=0;n<beta->size;++n)
    {
        double tmp=0;
        for(size_t m=0;m<NbIterations;++m)
            tmp += gsl_matrix_get(ResultsBeta,n,m) / (double)(NbIterations) ;

        gsl_vector_set(beta,n,tmp);
    }
}
