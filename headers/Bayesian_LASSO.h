#ifndef BAYESIAN_LASSO_H
#define BAYESIAN_LASSO_H

#include "Bayesian_Regressor.h"
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



class Bayesian_LASSO : public Bayesian_Regressor
{
    public:
        /** Default constructor */
        Bayesian_LASSO();
        //! Manual constructor
        Bayesian_LASSO(gsl_vector *b,gsl_matrix *a, size_t c, size_t d, size_t e, double f);
        /** Default destructor */
        virtual ~Bayesian_LASSO();
        /** Copy constructor
         *  \param other Object to copy from
         */
        Bayesian_LASSO(const Bayesian_LASSO& other);
        /** Assignment operator
         *  \param other Object to assign from
         *  \return A reference to this
         */
        Bayesian_LASSO& operator=(const Bayesian_LASSO& other);
        /** Access SparsityParameter
         * \return The current value of SparsityParameter
         */
        double GetSparsityParameter() { return SparsityParameter; }
        /** Set SparsityParameter
         * \param val New value to set
         */
        void SetSparsityParameter(double val) { SparsityParameter = val; }
        /** Access Auxiliary
         * \return The current value of Auxiliary
         */
        void GetAuxiliary(gsl_matrix* cpy) { gsl_matrix_memcpy(cpy,Auxiliary); }
        /** Set Auxiliary
         * \param val New value to set
         */
        void SetAuxiliary(gsl_matrix* val) { gsl_matrix_memcpy(Auxiliary,val); }
        /** Access sigma2_estimate
         * \return The current value of sigma2_estimate
         */
        void Getsigma2_estimate(gsl_vector* cpy) { gsl_vector_memcpy(cpy,sigma2_estimate); }
        /** Set sigma2_estimate
         * \param val New value to set
         */
        void Setsigma2_estimate(gsl_vector* val) { gsl_vector_memcpy(sigma2_estimate,val); }
        /** Computes the auxiliary matrix ATA+D-1
         * \param None
         */
        void ComputeAuxiliary(const gsl_vector* tau);
        /** Solves the Bayesian LASSO by means of a Gibbs sampler
         *  \todo Gives the MAP estimate of the posterior distribution
         */
        void SolveWithFixedSparsityParameter(double valSparsityParameter);


    protected:
        double SparsityParameter; //!< The sparsity parameter of the Bayesian LASSO
        gsl_vector* sigma2_estimate; //!< Estimate of the variance noise given by the Bayesian LASSO
    private:
        gsl_matrix* Auxiliary; //!< Auxiliary matrix used for the computation

};

#endif // BAYESIAN_LASSO_H
