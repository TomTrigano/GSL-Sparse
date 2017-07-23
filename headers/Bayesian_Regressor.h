#ifndef BAYESIAN_REGRESSOR_H
#define BAYESIAN_REGRESSOR_H

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <string>
#include <cmath>

class Bayesian_Regressor
{
    public:
        /** Default constructor */
        Bayesian_Regressor();
        Bayesian_Regressor(gsl_vector* b, gsl_matrix* a, size_t c, size_t d, size_t e, double f);
        /** Default destructor */
        virtual ~Bayesian_Regressor();
        /** Copy constructor
         *  \param other Object to copy from
         */
        Bayesian_Regressor(const Bayesian_Regressor& other);
        /** Assignment operator
         *  \param other Object to assign from
         *  \return A reference to this
         */
        Bayesian_Regressor& operator=(const Bayesian_Regressor& other);
        /** Access y
         * \return The current value of y
         */
        void Gety(gsl_vector* cpy) { gsl_vector_memcpy(cpy,y); }
        /** Set y
         * \param val New value to set
         */
        void Sety(gsl_vector* val) { gsl_vector_memcpy(y,val); }
        /** Access A
         * \return The current value of A
         */
        void GetA(gsl_matrix* cpy) { gsl_matrix_memcpy(cpy,A); }
        /** Set A
         * \param val New value to set
         */
        void SetA(gsl_matrix* val) { gsl_matrix_memcpy(A,val); }
        /** Access seed
         * \return The current value of seed
         */
        size_t Getseed() { return seed; }
        /** Set seed
         * \param val New value to set
         */
        void Setseed(size_t val) { seed = val; }
        /** Access NbIterations
         * \return The current value of NbIterations
         */
        size_t GetNbIterations() { return NbIterations; }
        /** Set NbIterations
         * \param val New value to set
         */
        void SetNbIterations(size_t val) { NbIterations = val; }
        /** Access BurnIn
         * \return The current value of BurnIn
         */
        size_t GetBurnIn() { return BurnIn; }
        /** Set BurnIn
         * \param val New value to set
         */
        void SetBurnIn(size_t val) { BurnIn = val; }
        /** Access sigma2
         * \return The current value of sigma2
         */
        double Getsigma2() { return sigma2; }
        /** Set sigma2
         * \param val New value to set
         */
        void Setsigma2(double val) { sigma2 = val; }
        /** Access ResultsBeta
         * \return The current value of ResultsBeta
         */
        void GetResultsBeta(gsl_matrix* cpy) { gsl_matrix_memcpy(cpy,ResultsBeta); }
        /** Set ResultsBeta
         * \param val New value to set
         */
        void SetResultsBeta(gsl_matrix* val) { gsl_matrix_memcpy(ResultsBeta,val); }
        /** Access beta
         * \return The current value of beta
         */
        void Getbeta(gsl_vector* cpy) { gsl_vector_memcpy(cpy,beta); }
        /** Set beta
         * \param val New value to set
         */
        void Setbeta(gsl_vector* val) {gsl_vector_memcpy(beta,val); }
        /** multivariate normal distribution random number generator
         *	\param mean vector of means of size n
         *	\param var variance matrix of dimension n x n
         *	\param result output variable with a sigle random vector normal distribution generation
         */
        void DrawMultivariateNormal(const gsl_rng *r, const gsl_vector *mean, const gsl_matrix *var, gsl_vector *result);
        /** inverse gamma distribution random number generator
         *	\param shape shape parameter
         *	\param scale scale parameter
         */
        double DrawInverseGamma(const gsl_rng *r, const double shape, const double scale);
        /** inverse gaussian distribution random number generator
         *	\param mu shape parameter
         *	\param lambda scale parameter
         *	\param result output variable with a sigle random vector normal distribution generation
         */
        double DrawInverseNormal(const gsl_rng *r, const double mu, const double lambda);
        /** Compute the Mean Estimate from the samples of the Gibbs Sampler - only the samples after the burn-in
         *  are taken into account.
         *  \param None
         */
        void MeanEstimate();

    protected:
        gsl_vector* y; //!< The observed data
        gsl_matrix* A; //!< The dictionary used in the regression
        size_t seed; //!< The seed used in the random number generators
        size_t NbIterations; //!< The number of iterations taken into account in the Giibs sampler
        size_t BurnIn; //!< The burn-in of the Gibbs sampler
        double sigma2; //!< The variance of the additive noise
        gsl_matrix* ResultsBeta; //!< The returned values of the Gibbs sampler
        gsl_vector* beta; //!< The MAP estimator taken from ResultsBeta
};

#endif // BAYESIAN_REGRESSOR_H
