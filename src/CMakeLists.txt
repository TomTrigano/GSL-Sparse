CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(Spectro_Tools CXX)


ADD_LIBRARY(sparse STATIC ./dictionaries/Const_Dictionary.cpp ./signal/Pileup_Correction.cpp ./signal/Post_Processing.cpp ./signal/Spectro_Signal.cpp ./solvers/FISTA_solver.cpp ./solvers/LASSO_LARS_solver.cpp  ./solvers/Bayesian_Regressor.cpp ./solvers/Bayesian_LASSO.cpp)

ADD_LIBRARY(stats STATIC ./stats/Kernel_estimator.cpp ./stats/Poisson_Process.cpp)

INSTALL(TARGETS sparse stats DESTINATION  lib)
