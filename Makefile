CC=g++
ARCHFLAGS=-arch i386 -arch x86_64
LDFLAGS=  -lgsl -lgslcblas -lm
DEBUG_DIR=./src/debug/
SOLVER_DIR=./src/solvers/
SIGNAL_DIR=./src/signal/
MAIN_DIR=./src/mains/
DICO_DIR=./src/dictionaries/
EXEC_DIR=./bin/
O_DIR=./o_files/
CC_OPTS= -Wall -g -I./headers/ 

all: $(EXEC)

pileupcorrection: Const_Dictionary.o LASSO_LARS_solver.o Post_Processing.o Spectro_Signal.o
	$(CC) $(CC_OPTS) -c $(MAIN_DIR)main_pileup_correction_adonis.cpp
	mv main_pileup_correction_adonis.o $(O_DIR)
	$(CC)  -arch x86_64 -v -o $(EXEC_DIR)PILEUP_CORRECTION_ILIA $(O_DIR)main_pileup_correction_adonis.o $(O_DIR)Spectro_Signal.o $(O_DIR)Post_Processing.o $(O_DIR)Const_Dictionary.o $(O_DIR)LASSO_LARS_solver.o  $(LDFLAGS)

process_adonis: Spectro_Signal.o
	$(CC) $(CC_OPTS) -c $(MAIN_DIR)main_adonis_process.cpp
	mv main_adonis_process.o $(O_DIR)
	$(CC) -o $(EXEC_DIR)PROCESS_ADONIS_ILIA $(O_DIR)main_adonis_process.o $(O_DIR)Spectro_Signal.o $(LDFLAGS)

reg_acqui_FISTA2012: Const_Dictionary.o FISTA_solver.o Spectro_Signal.o Post_Processing.o
	$(CC) $(CC_OPTS) -c $(MAIN_DIR)main_acqui2012_regFISTA.cpp
	mv main_acqui2012_regFISTA.o $(O_DIR)
	$(CC) -o $(EXEC_DIR)REG_ACQUI_FISTA2012 $(O_DIR)main_acqui2012_regFISTA.o $(O_DIR)Spectro_Signal.o $(O_DIR)Post_Processing.o $(O_DIR)Const_Dictionary.o $(O_DIR)FISTA_solver.o  $(LDFLAGS)

reg_acqui2012: Const_Dictionary.o Spectro_Signal.o LASSO_LARS_solver.o Post_Processing.o
	$(CC) $(CC_OPTS) -c $(MAIN_DIR)main_acqui2012_reg.cpp
	mv main_acqui2012_reg.o $(O_DIR)
	$(CC) -o $(EXEC_DIR)REG_ACQUI2012 $(O_DIR)main_acqui2012_reg.o $(O_DIR)Spectro_Signal.o $(O_DIR)Post_Processing.o $(O_DIR)Const_Dictionary.o $(O_DIR)LASSO_LARS_solver.o  $(LDFLAGS)

process_acqui2012: Spectro_Signal.o
	$(CC) $(CC_OPTS) -c $(MAIN_DIR)main_acqui2012_process.cpp
	mv main_acqui2012_process.o $(O_DIR)
	$(CC) -o $(EXEC_DIR)PROCESS_ACQUI2012 $(O_DIR)main_acqui2012_process.o $(O_DIR)Spectro_Signal.o $(LDFLAGS)

test_regrouping: Post_Processing.o
	$(CC) $(CC_OPTS) -c $(DEBUG_DIR)test_regrouping.cpp
	mv test_regrouping.o $(O_DIR)
	$(CC) -o $(EXEC_DIR)Test_Regrouping $(O_DIR)test_regrouping.o $(O_DIR)Post_Processing.o $(LDFLAGS)

signal: test_signal.o Spectro_Signal.o
	$(CC) -o $(EXEC_DIR)TestDC $(O_DIR)test_signal.o $(O_DIR)Spectro_Signal.o $(LDFLAGS)

basic_fista: FISTA_solver.o basic_fista.o
	$(CC) -o $(EXEC_DIR)BasicFistaTestDebug $(O_DIR)basic_fista.o $(O_DIR)FISTA_solver.o $(LDFLAGS)

basic_test: LASSO_LARS_solver.o basic_test.o
	$(CC) -o $(EXEC_DIR)BasicTestDebug $(O_DIR)basic_test.o $(O_DIR)LASSO_LARS_solver.o $(LDFLAGS)

lasso_dir: LASSO_LARS_solver.o test_compute_direction.o
	$(CC) -o $(EXEC_DIR)TestLASSODirection $(O_DIR)test_compute_direction.o $(O_DIR)LASSO_LARS_Solver.o $(LDFLAGS)

init_cholesky: LASSO_LARS_solver.o test_init_chol.o
	$(CC) -o $(EXEC_DIR)InitCholesky $(O_DIR)test_init_chol.o $(O_DIR)LASSO_LARS_Solver.o $(LDFLAGS)

stopping: LASSO_LARS_solver.o test_stopping_criterion.o
	$(CC) -o $(EXEC_DIR)Stopping $(O_DIR)test_stopping_criterion.o $(O_DIR)LASSO_LARS_Solver.o $(LDFLAGS)

initialization: LASSO_LARS_solver.o test_initialize_matrices.o
	$(CC) -o $(EXEC_DIR)Initialize $(O_DIR)test_initialize_matrices.o $(O_DIR)LASSO_LARS_Solver.o $(LDFLAGS)

test_new_dict : test_new_dict.o Spectro_Signal.o
	$(CC) -o $(EXEC_DIR)test_new_dict $(O_DIR)test_new_dict.o $(O_DIR)LASSO_LARS_Solver.o $(LDFLAGS)

Post_Processing.o:
	$(CC) $(CC_OPTS) -c $(SIGNAL_DIR)Post_Processing.cpp
	mv Post_Processing.o $(O_DIR)

Spectro_Signal.o:
	$(CC) $(CC_OPTS) -c $(SIGNAL_DIR)Spectro_Signal.cpp
	mv Spectro_Signal.o $(O_DIR)

Kernel.o:
	$(CC) $(CC_OPTS) -c $(SIGNAL_DIR)Kernel_estimator.cpp
	mv Kernel_estimator.o $(O_DIR)

Const_Dictionary.o:
	$(CC) $(CC_OPTS) -c $(DICO_DIR)Const_Dictionary.cpp
	mv Const_Dictionary.o $(O_DIR)

LASSO_LARS_solver.o:
	$(CC) $(CC_OPTS) -c $(SOLVER_DIR)LASSO_LARS_solver.cpp
	mv LASSO_LARS_solver.o $(O_DIR)

FISTA_solver.o:
	$(CC) $(CC_OPTS) -c $(SOLVER_DIR)FISTA_solver.cpp
	mv FISTA_solver.o $(O_DIR)

test_compute_direction.o:
	$(CC) $(CC_OPTS) -c $(DEBUG_DIR)test_compute_direction.cpp
	mv test_compute_direction.o $(O_DIR)

test_init_chol.o:
	$(CC) $(CC_OPTS) -c $(DEBUG_DIR)test_init_chol.cpp
	mv test_init_chol.o $(O_DIR)

test_stopping_criterion.o:
	$(CC) $(CC_OPTS) -c $(DEBUG_DIR)test_stopping_criterion.cpp
	mv test_stopping_criterion.o $(O_DIR)

test_initialize_matrices.o:
	$(CC) $(CC_OPTS) -c $(DEBUG_DIR)test_initialize_matrices.cpp
	mv test_initialize_matrices.o $(O_DIR)

basic_test.o:
	$(CC) $(CC_OPTS) -c $(DEBUG_DIR)basic_test.cpp
	mv basic_test.o $(O_DIR)

basic_fista.o:
	$(CC) $(CC_OPTS) -c $(DEBUG_DIR)basic_fista.cpp
	mv basic_fista.o $(O_DIR)

test_signal.o:
	$(CC) $(CC_OPTS) -c $(DEBUG_DIR)test_signal.cpp
	mv test_signal.o $(O_DIR)

test_new_dict.o:
	$(CC) $(CC_OPTS) -c $(DEBUG_DIR)test_new_dict.cpp
	mv test_new_dict.o $(O_DIR)

clean:
	rm $(O_DIR)*.o

dist_clean:
	rm $(EXEC_DIR)*
	rm $(O_DIR)*.o
