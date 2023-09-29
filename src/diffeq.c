#include "diffeq.h"
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
    int number_of_inputs = 0;
    int number_of_outputs = 0;
    int number_of_states = 0;
    int data_len = 0;
    int indices_len = 0;
    get_dims(&number_of_states, &number_of_inputs, &number_of_outputs, &data_len, &indices_len);

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
    if (strcmp(options->jacobian, "sparse") == 0) {
        printf("sparse jacobian not implemented");
        return(1);
    }
    else if (strcmp(options->jacobian, "dense") == 0 || strcmp(options->jacobian, "none") == 0) {
        jacobian = SUNDenseMatrix(number_of_states, number_of_states, sundials->sunctx);
    }
    else if (strcmp(options->jacobian, "matrix-free") == 0) {
    } else {
        printf("unknown jacobian %s", options->jacobian);
        return(1);
    };

    int precon_type = SUN_PREC_LEFT;
    if (strcmp(options->preconditioner, "none") == 0) {
        precon_type = SUN_PREC_NONE;
    } else {
        precon_type = SUN_PREC_LEFT;
    };

    // set linear solver
    SUNLinearSolver linear_solver;
    if (strcmp(options->linear_solver, "SUNLinSol_Dense") == 0) {
        linear_solver = SUNLinSol_Dense(yy, jacobian, sundials->sunctx);
    }
    else if (strcmp(options->linear_solver, "SUNLinSol_KLU") == 0) {
        printf("KLU linear solver not implemented");
        return(1);
    }
    else if (strcmp(options->linear_solver, "SUNLinSol_SPBCGS") == 0) {
       linear_solver = SUNLinSol_SPBCGS(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    }
    else if (strcmp(options->linear_solver, "SUNLinSol_SPFGMR") == 0) {
       linear_solver = SUNLinSol_SPFGMR(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    }
    else if (strcmp(options->linear_solver, "SUNLinSol_SPGMR") == 0) {
       linear_solver = SUNLinSol_SPGMR(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    }
    else if (strcmp(options->linear_solver, "SUNLinSol_SPTFQMR") == 0) {
       linear_solver = SUNLinSol_SPTFQMR(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    } else {
        printf("unknown linear solver %s", options->linear_solver);
        return(1);
    };

    retval = IDASetLinearSolver(ida_mem, linear_solver, jacobian);
    if (check_retval(&retval, "IDASetLinearSolver", 0)) return(1);

    if (strcmp(options->preconditioner, "none") != 0) {
        printf("preconditioner not implemented");
        return(1);
    }

    if (strcmp(options->jacobian, "matrix-free") == 0) {
        //IDASetJacTimes(ida_mem, null, jtimes);
        printf("matrix-free jacobian not implemented");
        return(1);
    }
    else if (strcmp(options->jacobian, "none") == 0) {
        //IDASetJacFn(ida_mem, jacobian_casadi);
        printf("jacobian not implemented");
        return(1);
    }

    if (number_of_inputs > 0) {
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


int Sundials_number_of_inputs(Sundials *sundials) {
    return sundials->data->number_of_inputs;
}

int Sundials_number_of_outputs(Sundials *sundials) {
    return sundials->data->number_of_outputs;
}

int Sundials_number_of_states(Sundials *sundials) {
    return sundials->data->number_of_states;
}

int Sundials_solve(Sundials *sundials, Vector *times_vec, const Vector *inputs_vec, Vector *outputs_vec) {
    if (sundials->data->options->fixed_times) {
        if (times_vec->len < 2) {
            printf("fixed_times option is set, but times vector has length %d < 2", times_vec->len);
            return(1);
        }
    } else {
        if (times_vec->len != 2) {
            printf("fixed_times option is not set, but times vector has length %d instead of 2", times_vec->len);
            return(1);
        }
    }

    const realtype *inputs = inputs_vec->data;

    int number_of_inputs = 0;
    int number_of_outputs = 0;
    int number_of_states = 0;
    int data_len = 0;
    int indices_len = 0;
    get_dims(&number_of_states, &number_of_inputs, &number_of_outputs, &data_len, &indices_len);

    int retval;
    set_inputs(inputs, sundials->model->data);
    set_u0(sundials->model->data, sundials->model->indices, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yp));


    realtype *output;
    int tensor_size;
    get_out(sundials->model->data , &output, &tensor_size);

    realtype t0 = Vector_get(times_vec, 0);

    retval = IDAReInit(sundials->ida_mem, t0, sundials->data->yy, sundials->data->yp);
    if (check_retval(&retval, "IDAReInit", 0)) return(1);

    retval = IDACalcIC(sundials->ida_mem, IDA_YA_YDP_INIT, Vector_get(times_vec, 1));
    if (check_retval(&retval, "IDACalcIC", 0)) return(1);

    retval = IDAGetConsistentIC(sundials->ida_mem, sundials->data->yy, sundials->data->yp);
    if (check_retval(&retval, "IDAGetConsistentIC", 0)) return(1);
            
    calc_out(t0, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yp), sundials->model->data, sundials->model->indices);

    // init output and save initial state
    Vector_resize(outputs_vec, 0);
    for (int j = 0; j < number_of_outputs; j++) {
        Vector_push(outputs_vec, output[j]);
    }

    int itask = IDA_ONE_STEP;
    if (sundials->data->options->fixed_times) {
        itask = IDA_NORMAL;
    }
    realtype t_final = Vector_get(times_vec, times_vec->len - 1);

    if (!sundials->data->options->fixed_times) {
        // if using solver times save initial time point and get rid of the rest
        Vector_resize(times_vec, 1);
    }

    // set stop time as final time point
    retval = IDASetStopTime(sundials->ida_mem, t_final);
    if (check_retval(&retval, "IDASetStopTime", 0)) return(1);
    int i = 0;
    realtype t_next = t_final;
    while(1) {
        // advance to next time point
        i++;

        // if using fixed times set next time point
        if (sundials->data->options->fixed_times) {
            t_next = Vector_get(times_vec, i);
        }

        // solve up to next/final time point
        realtype tret;
        retval = IDASolve(sundials->ida_mem, t_next, &tret, sundials->data->yy, sundials->data->yp, itask);
        if (check_retval(&retval, "IDASolve", 0)) return(1);

        // get output (calculated into output array)
        calc_out(tret, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yp), sundials->model->data, sundials->model->indices);

        // save output
        for (int j = 0; j < number_of_outputs; j++) {
            Vector_push(outputs_vec, output[j]);
        }

        // if using solver times save time point
        if (!sundials->data->options->fixed_times) {
            Vector_push(times_vec, tret);
        }

        // if finished or errored break
        if (retval == IDA_TSTOP_RETURN || retval < 0) {
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

    int number_of_inputs = 0;
    int number_of_outputs = 0;
    int number_of_states = 0;
    int data_len = 0;
    int indices_len = 0;
    get_dims(&number_of_states, &number_of_inputs, &number_of_outputs, &data_len, &indices_len);
    
    sundials->data->number_of_inputs = number_of_inputs;
    sundials->data->number_of_outputs = number_of_outputs;
    sundials->data->number_of_states = number_of_states;

    sundials->model->data = malloc(data_len * sizeof(realtype));
    sundials->model->indices = malloc(indices_len * sizeof(int));
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
    free(sundials->model->indices);
    free(sundials->model->data);
    free(sundials->data);
    free(sundials->model);
    free(sundials);
}

Options *Options_create() {
    Options *options = malloc(sizeof(Options));
    options->print_stats = 0;
    options->fixed_times = 0;
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

void Options_set_print_stats(Options *options, const int print_stats) {
    options->print_stats = print_stats;
}

void Options_set_fixed_times(Options *options, const int fixed_times) {
    options->fixed_times = fixed_times;
}

int Options_get_fixed_times(Options *options) {
    return options->fixed_times;
}

int Options_get_print_stats(Options *options) {
    return options->print_stats;
}

Vector *Vector_linspace_create(realtype start, realtype stop, int len) {
    Vector *vector = Vector_create(len);
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

Vector *Vector_create_with_capacity(int len, int capacity) {
    if (capacity < len) {
        capacity = len;
    }
    if (capacity < 1) {
        capacity = 1;
    }
    Vector *vector = malloc(sizeof(Vector));
    vector->data = (realtype *)malloc(capacity * sizeof(realtype));
    vector->len = len;
    vector->capacity = capacity;
    return vector;
}

Vector *Vector_create(int len) {
    return Vector_create_with_capacity(len, len);
}

realtype Vector_get(Vector *vector, const int index) {
    return vector->data[index];
}

realtype *Vector_get_data(Vector *vector) {
    return vector->data;
}

void Vector_push(Vector *vector, realtype value) {
    if (vector->len == vector->capacity) {
        vector->capacity *= 2;
        vector->data = realloc(vector->data, vector->capacity * sizeof(realtype));
    }
    vector->data[vector->len++] = value;
}

void Vector_resize(Vector *vector, int len) {
    if (len > vector->capacity) {
        vector->capacity = len;
        vector->data = realloc(vector->data, vector->capacity * sizeof(realtype));
    }
    vector->len = len;
}

int Vector_get_length(Vector *vector) {
    return vector->len;
}

