#include "lib.h"
#include <string.h>
#include <math.h>


int sundials_residual(realtype t, N_Vector y, N_Vector ydot, N_Vector rr, void *user_data) {
    Sundials *sundials = (Sundials *)user_data;
    realtype *yy = N_VGetArrayPointer(y);
    realtype *yp = N_VGetArrayPointer(ydot);
    realtype *rrr = N_VGetArrayPointer(rr);
    realtype *data = sundials->model->data;
    int *indices = sundials->model->indices;
    residual(t, yy, yp, data, indices, rrr);
    return(0);
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

int check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  /* Check if retval < 0 */
  else if (opt == 1) {
    retval = (int *) returnvalue;
    if (*retval < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  return(0);
}

int Sundials_init(Sundials *sundials, const Options *options) {
    int number_of_states = get_number_of_states();
    int number_of_parameters = get_number_of_parameters();
    int retval;


    retval = SUNContext_Create(NULL, &sundials->sunctx);
    if (check_retval(&retval, "SUNContext_Create", 0)) return(1);

    void *ida_mem = IDACreate(sundials->sunctx);
    if (check_retval((void *)ida_mem, "IDACreate", 0)) return(1);

    N_Vector yy = N_VNew_Serial(number_of_states, sundials->sunctx);
    if (check_retval((void *)yy, "N_VNew_Serial", 0)) return(1);
    N_Vector yp = N_VNew_Serial(number_of_states, sundials->sunctx);
    if (check_retval((void *)yp, "N_VNew_Serial", 0)) return(1);
    N_Vector avtol = N_VNew_Serial(number_of_states, sundials->sunctx);
    if (check_retval((void *)avtol, "N_VNew_Serial", 0)) return(1);
    N_Vector id = N_VNew_Serial(number_of_states, sundials->sunctx);
    if (check_retval((void *)id, "N_VNew_Serial", 0)) return(1);


    // set tolerances
    N_VConst(options->atol, avtol);
            
    // initialise solver
    retval = IDAInit(ida_mem, sundials_residual, 0.0, yy, yp);
    if (check_retval(&retval, "IDAInit", 0)) return(1);

    // set tolerances
    retval = IDASVtolerances(ida_mem, options->rtol, avtol);
    if (check_retval(&retval, "IDASVtolerances", 0)) return(1);

    // set events
    //IDARootInit(ida_mem, number_of_events, events_casadi);


    // set matrix
    SUNMatrix jacobian;
    if (strcmp(options->jacobian, "sparse")) {
        printf("sparse jacobian not implemented");
        return(1);
    }
    else if (strcmp(options->jacobian, "dense") || strcmp(options->jacobian, "none")) {
        jacobian = SUNDenseMatrix(number_of_states, number_of_states, sundials->sunctx);
    }
    else if (strcmp(options->jacobian, "matrix-free")) {
    } else {
        printf("unknown jacobian %s", options->jacobian);
        return(1);
    };

    int precon_type = SUN_PREC_LEFT;
    if (strcmp(options->preconditioner, "none")) {
        precon_type = SUN_PREC_NONE;
    } else {
        precon_type = SUN_PREC_LEFT;
    };

    // set linear solver
    SUNLinearSolver linear_solver;
    if (strcmp(options->linear_solver, "SUNLinSol_Dense")) {
        linear_solver = SUNLinSol_Dense(yy, jacobian, sundials->sunctx);
    }
    else if (strcmp(options->linear_solver, "SUNLinSol_KLU")) {
        printf("KLU linear solver not implemented");
        return(1);
    }
    else if (strcmp(options->linear_solver, "SUNLinSol_SPBCGS")) {
       linear_solver = SUNLinSol_SPBCGS(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    }
    else if (strcmp(options->linear_solver, "SUNLinSol_SPFGMR")) {
       linear_solver = SUNLinSol_SPFGMR(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    }
    else if (strcmp(options->linear_solver, "SUNLinSol_SPGMR")) {
       linear_solver = SUNLinSol_SPGMR(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    }
    else if (strcmp(options->linear_solver, "SUNLinSol_SPTFQMR")) {
       linear_solver = SUNLinSol_SPTFQMR(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    } else {
        printf("unknown linear solver %s", options->linear_solver);
        return(1);
    };

    retval = IDASetLinearSolver(ida_mem, linear_solver, jacobian);
    if (check_retval(&retval, "IDASetLinearSolver", 0)) return(1);

    if (!strcmp(options->preconditioner, "none")) {
        printf("preconditioner not implemented");
        return(1);
    }

    if (strcmp(options->jacobian, "matrix-free")) {
        //IDASetJacTimes(ida_mem, null, jtimes);
        printf("matrix-free jacobian not implemented");
        return(1);
    }
    else if (!strcmp(options->jacobian, "none")) {
        //IDASetJacFn(ida_mem, jacobian_casadi);
        printf("jacobian not implemented");
        return(1);
    }

    if (number_of_parameters > 0) {
        //IDASensInit(ida_mem, number_of_parameters, IDA_SIMULTANEOUS,
        //            sensitivities, yyS, ypS);
        //IDASensEEtolerances(ida_mem);
    }

    retval = SUNLinSolInitialize(linear_solver);
    if (check_retval(&retval, "SUNLinSolInitialize", 0)) return(1);

    set_id(N_VGetArrayPointer(id));
    retval = IDASetId(ida_mem, id);
    if (check_retval(&retval, "IDASetId", 0)) return(1);

    retval = IDASetUserData(ida_mem, (void *)sundials);
    if (check_retval(&retval, "IDASetUserData", 0)) return(1);

    // allocate and zero model data
    sundials->model->data = malloc(get_data_size() * sizeof(realtype));
    sundials->model->indices = malloc(get_indices_size() * sizeof(int));

    // setup sundials fields
    sundials->ida_mem = ida_mem;
    sundials->data->yy = yy;
    sundials->data->yp = yp;
    sundials->data->avtol = avtol;
    sundials->data->id = id;
    sundials->data->jacobian = jacobian;
    sundials->data->linear_solver = linear_solver;
    sundials->data->options = options;
    return(0);
}

int Sundials_solve(Sundials *sundials, const realtype *times, const size_t number_of_times, const realtype *inputs, realtype *outputs) {
    int retval;
    int number_of_outputs = get_number_of_outputs();
    set_inputs(inputs);
    set_u0(sundials->model->data, sundials->model->indices, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yp));
    const realtype *output = get_output();

    realtype t0 = times[0];

    retval = IDAReInit(sundials->ida_mem, t0, sundials->data->yy, sundials->data->yp);
    if (check_retval(&retval, "IDAReInit", 0)) return(1);

    retval = IDACalcIC(sundials->ida_mem, IDA_YA_YDP_INIT, times[1]);
    if (check_retval(&retval, "IDACalcIC", 0)) return(1);

    retval = IDAGetConsistentIC(sundials->ida_mem, sundials->data->yy, sundials->data->yp);
    if (check_retval(&retval, "IDAGetConsistentIC", 0)) return(1);
            
    calc_out(t0, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yp), sundials->model->data, sundials->model->indices);
    for (int j = 0; j < number_of_outputs; j++) {
        outputs[j] = output[j];
    }

    realtype t_final = times[number_of_times - 1];
    for (int i = 0; i < number_of_times; i++) {
        realtype t_next = times[i];
        retval = IDASetStopTime(sundials->ida_mem, t_next);
        if (check_retval(&retval, "IDASetStopTime", 0)) return(1);

        realtype tret;
        retval = IDASolve(sundials->ida_mem, t_final, &tret, sundials->data->yy, sundials->data->yp, IDA_NORMAL);
        if (check_retval(&retval, "IDASolve", 0)) return(1);

        calc_out(t_next, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yp), sundials->model->data, sundials->model->indices);
        for (int j = 0; j < number_of_outputs; j++) {
            outputs[i * number_of_outputs + j] = output[j];
        }
        if (retval == IDA_SUCCESS || retval == IDA_ROOT_RETURN) {
            break;
        }
    }

    if (sundials->data->options->print_stats) {
        int klast, kcur;
        long int netfails, nlinsetups, nfevals, nsteps;
        realtype hinused, hlast, hcur, tcur;
        retval = IDAGetIntegratorStats(sundials->ida_mem, &nsteps, &nfevals, &nlinsetups, &netfails, &klast, &kcur, &hinused, &hlast, &hcur, &tcur);
        if (check_retval(&retval, "IDAGetIntegratorStats", 0)) return(1);

        long int nniters, nncfails;
        retval = IDAGetNonlinSolvStats(sundials->ida_mem, &nniters, &nncfails);
        if (check_retval(&retval, "IDAGetNonlinSolvStats", 0)) return(1);

        printf("Solver Stats:\n");
        printf("\tNumber of steps = %ld\n", nsteps);
        printf("\tNumber of calls to residual function = %ld\n", nfevals);
        printf("\tNumber of linear solver setup calls = %ld\n", nlinsetups);
        printf("\tNumber of error test failures = %ld\n", netfails);
        printf("\tMethod order used on last step = %d\n", klast);
        printf("\tMethod order used on next step = %d\n", kcur);
        printf("\tInitial step size = %f\n", hinused);
        printf("\tStep size on last step = %f\n", hlast);
        printf("\tStep size on next step = %f\n", hcur);
        printf("\tCurrent internal time reached = %f\n", tcur);
        printf("\tNumber of nonlinear iterations performed = %ld\n", nniters);
        printf("\tNumber of nonlinear convergence failures = %ld\n", nncfails);
    }
    return(0);
}

Sundials *Sundials_create() {
    Sundials *sundials = malloc(sizeof(Sundials));
    sundials->data = malloc(sizeof(SundialsData));
    sundials->model = malloc(sizeof(ModelData));
    return sundials;
}

void Sundials_destroy(Sundials *sundials) {
    SUNLinSolFree(sundials->data->linear_solver);
    SUNMatDestroy(sundials->data->jacobian);
    N_VDestroy(sundials->data->yy);
    N_VDestroy(sundials->data->yp);
    N_VDestroy(sundials->data->avtol);
    N_VDestroy(sundials->data->id);
    IDAFree(&(sundials->ida_mem));
    SUNContext_Free(&sundials->sunctx);
    free(sundials->data);
    free(sundials->model);
    free(sundials);
}

Options *Options_create() {
    Options *options = malloc(sizeof(Options));
    options->print_stats = 0;
    options->jacobian = "dense";
    options->linear_solver = "SUNLinSol_Dense";
    options->preconditioner = "none";
    options->linsol_max_iterations = 0;
    options->rtol = 1e-6;
    options->atol = 1e-6;
    return options;
}

void Options_destroy(Options *options) {
    free(options);
}

Vector *Vector_linspace_create(realtype start, realtype stop, int len) {
    Vector *vector = malloc(sizeof(Vector));
    vector->data = (realtype *)malloc(len * sizeof(realtype));
    vector->len = len;
    realtype step = (stop - start) / (len - 1);
    for (int i = 0; i < len; i++) {
        vector->data[i] = start + i * step;
    }
    return vector;
}

void Vector_destroy(Vector *vector) {
    free(vector->data);
    free(vector);
} 

Vector *Vector_create(int len) {
    Vector *vector = malloc(sizeof(Vector));
    vector->data = (realtype *)malloc(len * sizeof(realtype));
    vector->len = len;
    return vector;
}

