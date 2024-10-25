#include "diffeq.h"
#include <string.h>
#include <math.h>
#include <assert.h>

int sundials_residual(realtype t, N_Vector y, N_Vector ydot, N_Vector rr, void *user_data) {
    Sundials *sundials = (Sundials *)user_data;
    realtype *yy = N_VGetArrayPointer(y);
    realtype *yp = N_VGetArrayPointer(ydot);
    realtype *rrr = N_VGetArrayPointer(rr);
    realtype *data = sundials->model->data;
    residual(t, yy, yp, data, rrr);
    return(0);
}

int sundials_root(realtype t, N_Vector y, N_Vector ydot, realtype *gout, void *user_data) {
    Sundials *sundials = (Sundials *)user_data;
    realtype *yy = N_VGetArrayPointer(y);
    realtype *yp = N_VGetArrayPointer(ydot);
    realtype *data = sundials->model->data;
    calc_stop(t, yy, yp, data, gout);
    return(0);
}

int sundials_jtime(realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, N_Vector v, N_Vector Jv, realtype cj, void *user_data, N_Vector tmp1, N_Vector tmp2) {
    Sundials *sundials = (Sundials *)user_data;
    realtype *yy_ = N_VGetArrayPointer(yy);
    realtype *yp_ = N_VGetArrayPointer(yp);
    realtype *ypS_ = N_VGetArrayPointer(tmp1);
    realtype *rr_ = N_VGetArrayPointer(rr);
    realtype *v_ = N_VGetArrayPointer(v);
    realtype *Jv_ = N_VGetArrayPointer(Jv);
    realtype *data = sundials->model->data;
    realtype *data_jacobian = sundials->model->data_jacobian;
    
    // J = df/dy + cj * df/dy'
    for (int i = 0; i < sundials->data->number_of_states; i++) {
        ypS_[i] = cj * v_[i];
    }
    residual_grad(tt, yy_, v_, yp_, ypS_, data, data_jacobian, rr_, Jv_);
    return(0);
}

int sundials_jacobian(realtype t, realtype cj, N_Vector y, N_Vector yp, N_Vector r, SUNMatrix Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    Sundials *sundials = (Sundials *)user_data;
    realtype *yy_ = N_VGetArrayPointer(y);
    realtype *yyS_ = N_VGetArrayPointer(tmp1);
    realtype *yp_ = N_VGetArrayPointer(yp);
    realtype *ypS_ = N_VGetArrayPointer(tmp2);
    realtype *rr_ = N_VGetArrayPointer(r);
    realtype *drr_ = N_VGetArrayPointer(tmp3);
    realtype *data = sundials->model->data;
    realtype *data_jacobian = sundials->model->data_jacobian;
    const int is_dense = sundials->data->options->jacobian == DENSE_JACOBIAN;

    int nnz = 0;
    int alloc_nnz = 0;
    if (!is_dense) {
        // zero matrix
        SM_INDEXPTRS_S(Jac)[0] = 0;
        alloc_nnz = SM_NNZ_S(Jac);
    }

    for (int i = 0; i < sundials->data->number_of_states; i++) {
        // J = df/dy + cj * df/dy'
        for (int j = 0; j < sundials->data->number_of_states; j++) {
            if (j == i) {
                yyS_[j] = 1;
                ypS_[j] = cj;
            } else {
                yyS_[j] = 0;
                ypS_[j] = 0;
            }
        }

        residual_grad(t, yy_, yyS_, yp_, ypS_, data, data_jacobian, rr_, drr_);

        
        // copy into jacobian
        if (is_dense) {
            // if dense copy the whole row
            realtype *col = SM_COLUMN_D(Jac, i);
            for (int j = 0; j < sundials->data->number_of_states; j++) {
                col[j] = drr_[j];
            }
        } else {
            sunindextype* row_vals = SM_INDEXVALS_S(Jac) + nnz;
            realtype *data = SM_DATA_S(Jac) + nnz;
            int col_nnz = 0;
            for (int j = 0; j < sundials->data->number_of_states; j++) {
                if (drr_[j] != 0) {
                    row_vals[col_nnz] = j;
                    data[col_nnz] = drr_[j];
                    col_nnz++;
                    if (nnz + col_nnz > alloc_nnz) {
                        alloc_nnz = 2 * (nnz + col_nnz);
                        SM_INDEXPTRS_S(Jac) = (sunindextype *) realloc(SM_INDEXPTRS_S(Jac), alloc_nnz);
                        SM_DATA_S(Jac) = (realtype *) realloc(SM_DATA_S(Jac), alloc_nnz);
                    }
                }
            }
            nnz += col_nnz;
            SM_INDEXPTRS_S(Jac)[i + 1] = nnz;
        }
    }
    return(0);
}

int sundials_sensitivities(int Ns, realtype t, N_Vector yy, N_Vector yp, N_Vector resval, N_Vector *yS, N_Vector *ypS, N_Vector *resvalS, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    // resvalS is df/dy * dy + df/dy' * dy' + df/dp * dp
    Sundials *sundials = (Sundials *)user_data;
    realtype *yy_ = N_VGetArrayPointer(yy);
    realtype *yp_ = N_VGetArrayPointer(yp);
    realtype *rr_ = N_VGetArrayPointer(resval);
    realtype *yyS_ = N_VGetArrayPointer(yS[0]);
    realtype *ypS_ = N_VGetArrayPointer(ypS[0]);
    realtype *rrS_ = N_VGetArrayPointer(resvalS[0]);
    realtype *data = sundials->model->data;
    realtype *data_sens = sundials->model->data_sens;
    residual_grad(t, yy_, yyS_, yp_, ypS_, data, data_sens, rr_, rrS_);
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

// random number between epsilon and 1
realtype my_rand(void) {
    return (realtype)rand() / (realtype)((unsigned)RAND_MAX + 1) + FLT_EPSILON;
}

MatrixCSC *MatrixCSC_create(int nrows, int ncols) {
    MatrixCSC *matrix = malloc(sizeof(MatrixCSC));
    matrix->nrow = nrows;
    matrix->ncol = ncols;
    int nnz = 0;
    int capacity = ncols;
    matrix->nnz = nnz;
    matrix->row_indices = VectorInt_create_with_capacity(nnz, capacity);
    matrix->col_ptrs = VectorInt_create_and_fill(ncols + 1, 0);
    matrix->data = Vector_create_with_capacity(nnz, capacity);
    return matrix;
}

void MatrixCSC_destroy(MatrixCSC *matrix) {
    VectorInt_destroy(matrix->row_indices);
    VectorInt_destroy(matrix->col_ptrs);
    Vector_destroy(matrix->data);
    free(matrix);
}

void MatrixCSC_add_col(MatrixCSC *matrix, Vector *data) {
    // find last col
    int col = -1;
    for (int i = 0; i < matrix->col_ptrs->len; i++) {
        if (matrix->col_ptrs->data[i] == matrix->nnz) {
            col = i;
            break;
        }
    }
    assert(col >= 0);

    // data len should be nrows
    assert(data->len == matrix->nrow);

    // only copy in non-zero elements
    for (int i = 0; i < data->len; i++) {
        if (data->data[i] != 0) {
            VectorInt_push(matrix->row_indices, i);
            Vector_push(matrix->data, data->data[i]);
            matrix->nnz++;
        }
    }

    // update colptrs
    matrix->col_ptrs->data[col + 1] = matrix->nnz;
}

MatrixCSC *Sundials_create_jacobian(Sundials *sundials) {
    Vector *yy = Vector_create_and_fill(sundials->data->number_of_states, my_rand());
    Vector *yp = Vector_create_and_fill(sundials->data->number_of_states, my_rand());
    Vector *yyS = Vector_create(sundials->data->number_of_states);
    Vector *ypS = Vector_create(sundials->data->number_of_states);
    Vector *rr = Vector_create(sundials->data->number_of_states);
    Vector *drr = Vector_create(sundials->data->number_of_states);
    
    MatrixCSC *jacobian = MatrixCSC_create(sundials->data->number_of_states, sundials->data->number_of_states);

    realtype *data = sundials->model->data;
    realtype *data_jacobian = sundials->model->data_jacobian;
    for (int i = 0; i < sundials->data->number_of_states; i++) {
        // J = df/dy + cj * df/dy'
        for (int j = 0; j < sundials->data->number_of_states; j++) {
            drr->data[i] = 0.;
            if (j == i) {
                yyS->data[j] = my_rand();
                ypS->data[i] = my_rand();
            } else {
                yyS->data[j] = 0;
                ypS->data[i] = 0;
            }
        }
        residual_grad(my_rand(), yy->data, yyS->data, yp->data, ypS->data, data, data_jacobian, rr->data, drr->data);
        MatrixCSC_add_col(jacobian, drr);
    }
    return(jacobian);
}

int Sundials_init(Sundials *sundials, const Options *options) {
    int number_of_inputs = 0;
    int number_of_outputs = 0;
    int number_of_states = 0;
    int data_len = 0;
    int number_of_stop = 0;
    get_dims(&number_of_states, &number_of_inputs, &number_of_outputs, &data_len, &number_of_stop);
    
    if (options->debug) {
        printf("number_of_states = %d\n", number_of_states);
        printf("number_of_inputs = %d\n", number_of_inputs);
        printf("number_of_outputs = %d\n", number_of_outputs);
        printf("number_of_stop = %d\n", number_of_stop);
        printf("data_len = %d\n", data_len);
        
        printf("options.print_stats = %d\n", options->print_stats);
        printf("options.fixed_times = %d\n", options->fixed_times);
        printf("options.jacobian = %d\n", options->jacobian);
        printf("options.linear_solver = %d\n", options->linear_solver);
        printf("options.preconditioner = %d\n", options->preconditioner);
        printf("options.linsol_max_iterations = %d\n", options->linsol_max_iterations);
        printf("options.rtol = %f\n", options->rtol);
        printf("options.atol = %f\n", options->atol);
        printf("options.debug = %d\n", options->debug);
        printf("options.fwd_sens = %d\n", options->fwd_sens);
        printf("options.mxsteps = %d\n", options->mxsteps);
        printf("options.min_step = %f\n", options->min_step);
        printf("options.max_step = %f\n", options->max_step);
    }

    int retval;

    retval = SUNContext_Create(NULL, &sundials->sunctx);
    if (check_retval(&retval, "SUNContext_Create", 1)) return(1);

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

    N_Vector tmp = N_VNew_Serial(number_of_states, sundials->sunctx);
    if (check_retval((void *)tmp, "N_VNew_Serial", 0)) return(1);

    N_Vector yyS = NULL;
    N_Vector ypS =  NULL;
    if (options->fwd_sens) {
        yyS = N_VNew_Serial(number_of_states, sundials->sunctx);
        if (check_retval((void *)yyS, "N_VNew_Serial", 0)) return(1);
        ypS = N_VNew_Serial(number_of_states, sundials->sunctx);
        if (check_retval((void *)ypS, "N_VNew_Serial", 0)) return(1);
        N_VConst(RCONST(0.0), yyS);
        N_VConst(RCONST(0.0), ypS);
    }
    
    // set tolerances
    N_VConst(options->atol, avtol);
            
    // initialise solver
    retval = IDAInit(ida_mem, sundials_residual, 0.0, yy, yp);
    if (check_retval(&retval, "IDAInit", 1)) return(1);

    // set tolerances
    retval = IDASVtolerances(ida_mem, options->rtol, avtol);
    if (check_retval(&retval, "IDASVtolerances", 1)) return(1);

    // set user data
    retval = IDASetUserData(ida_mem, (void *)sundials);
    if (check_retval(&retval, "IDASetUserData", 1)) return(1);

    // set events
    if (number_of_stop > 0) {
        retval = IDARootInit(ida_mem, number_of_stop, sundials_root);
        if (check_retval(&retval, "IDARootInit", 1)) return(1);
    }

    // set matrix
    SUNMatrix jacobian;
    if (options->jacobian == SPARSE_JACOBIAN) {
        MatrixCSC *jac = Sundials_create_jacobian(sundials);
        // allocate enough for a 3-banded jacobian
        int nnz = 3 * number_of_states;
        int matrix_size = number_of_states * number_of_states;
        if (nnz > matrix_size) {
            nnz = matrix_size;
        }
        jacobian = SUNSparseMatrix(number_of_states, number_of_states, nnz, CSC_MAT, sundials->sunctx);
        MatrixCSC_destroy(jac);
    }
    else if (options->jacobian == DENSE_JACOBIAN || options->jacobian == NO_JACOBIAN) {
        jacobian = SUNDenseMatrix(number_of_states, number_of_states, sundials->sunctx);
    }
    else if (options->jacobian == MATRIX_FREE_JACOBIAN) {
        // do nothing
    }

    int precon_type;
    if (options->preconditioner == PRECON_LEFT) {
        precon_type = SUN_PREC_LEFT;
    } else if (options->preconditioner == PRECON_RIGHT) {
        precon_type = SUN_PREC_RIGHT;
    } else if (options->preconditioner == PRECON_NONE) {
        precon_type = SUN_PREC_NONE;
    } else {
        printf("unknown preconditioner %d", options->preconditioner);
        return(1);
    }

    // set linear solver
    SUNLinearSolver linear_solver;
    if (options->linear_solver == LINEAR_SOLVER_DENSE) {
        linear_solver = SUNLinSol_Dense(yy, jacobian, sundials->sunctx);
    }
    else if (options->linear_solver == LINEAR_SOLVER_KLU) {
       linear_solver = SUNLinSol_KLU(yy, jacobian, sundials->sunctx);
    }
    else if (options->linear_solver == LINEAR_SOLVER_SPBCGS) {
       linear_solver = SUNLinSol_SPBCGS(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    }
    else if (options->linear_solver == LINEAR_SOLVER_SPFGMR) {
       linear_solver = SUNLinSol_SPFGMR(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    }
    else if (options->linear_solver == LINEAR_SOLVER_SPGMR) {
       linear_solver = SUNLinSol_SPGMR(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    }
    else if (options->linear_solver == LINEAR_SOLVER_SPTFQMR) {
       linear_solver = SUNLinSol_SPTFQMR(yy, precon_type, options->linsol_max_iterations, sundials->sunctx);
    } else {
        printf("unknown linear solver %d", options->linear_solver);
        return(1);
    }

    retval = IDASetLinearSolver(ida_mem, linear_solver, jacobian);
    if (check_retval(&retval, "IDASetLinearSolver", 1)) return(1);

    if (options->preconditioner != PRECON_NONE) {
        printf("preconditioner not implemented");
        return(1);
    }

    if (options->jacobian == MATRIX_FREE_JACOBIAN) {
        IDASetJacTimes(ida_mem, NULL, sundials_jtime);
    }
    
    if (options->jacobian == DENSE_JACOBIAN || options->jacobian == SPARSE_JACOBIAN) {
        IDASetJacFn(ida_mem, sundials_jacobian);
    }

    if (options->fwd_sens) {
        if (number_of_inputs > 0) {
            IDASensInit(ida_mem, 1, IDA_SIMULTANEOUS, sundials_sensitivities, &yyS, &ypS);
            IDASensEEtolerances(ida_mem);
        }
    }

    retval = SUNLinSolInitialize(linear_solver);
    if (check_retval(&retval, "SUNLinSolInitialize", 1)) return(1);

    realtype *id_data = N_VGetArrayPointer(id);
    set_id(id_data);
    bool is_ode = true;
    if (options->debug)
        printf("id_data = [");
    for (int i = 0; i < number_of_states; i++) {
        if (options->debug)
            printf("%f, ", id_data[i]);
        if (id_data[i] == 0.0) {
            is_ode = false;
            break;
        }
    }
    if (options->debug) {
        printf("]\n");
        printf("is_ode = %d\n", is_ode);
    }
    retval = IDASetId(ida_mem, id);
    if (check_retval(&retval, "IDASetId", 1)) return(1);
    
    // Set max number of steps
    retval = IDASetMaxNumSteps(ida_mem, options->mxsteps);
    if (check_retval(&retval, "IDASetMaxNumSteps", 1)) return(1);
    
    // Set min step
    retval = IDASetMinStep(ida_mem, options->min_step);
    if (check_retval(&retval, "IDASetMinStep", 1)) return(1);

    // Set max step
    retval = IDASetMaxStep(ida_mem, options->max_step);
    if (check_retval(&retval, "IDASetMaxStep", 1)) return(1);

    // setup sundials fields
    sundials->ida_mem = ida_mem;
    sundials->data->yy = yy;
    sundials->data->yp = yp;
    sundials->data->yyS = yyS;
    sundials->data->ypS = ypS;
    sundials->data->tmp = tmp;
    sundials->data->avtol = avtol;
    sundials->data->id = id;
    sundials->data->sundials_jacobian = jacobian;
    sundials->data->sundials_linear_solver = linear_solver;
    sundials->data->options = options;
    sundials->data->is_ode = is_ode;
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

int Sundials_solve(Sundials *sundials, Vector *times_vec, const Vector *inputs_vec, const Vector *dinputs_vec, Vector *outputs_vec, Vector *doutputs_vec) {
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
    
    int number_of_inputs = 0;
    int number_of_outputs = 0;
    int number_of_states = 0;
    int number_of_stop = 0;
    int data_len = 0;
    get_dims(&number_of_states, &number_of_inputs, &number_of_outputs, &data_len, &number_of_stop);
    
    const int fwd_sens = sundials->data->options->fwd_sens && number_of_inputs > 0;
    
    const realtype *inputs = inputs_vec->data;
    const realtype *dinputs = NULL;
    if (fwd_sens) {
        if (!dinputs_vec) {
            printf("fwd_sens option is set, but dinputs vector is NULL");
            return(1);
        }
        dinputs = dinputs_vec->data;
    }

    int retval;
    
    if (fwd_sens) {
        set_inputs_grad(inputs, dinputs, sundials->model->data, sundials->model->data_sens);
        set_u0_grad(sundials->model->data, sundials->model->data_sens, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yyS), N_VGetArrayPointer(sundials->data->yp), N_VGetArrayPointer(sundials->data->ypS));
    } else {
        set_inputs(inputs, sundials->model->data);
        set_u0(sundials->model->data, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yp));
    }

    realtype t0 = Vector_get(times_vec, 0);

    if (sundials->data->is_ode) {
        // set tmp to zero
        N_VConst(0.0, sundials->data->tmp);
        // assume residual is M * y' - g(t, y)
        // so if y' = 0, then residual is -g(t, y)
        residual(t0, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->tmp), sundials->model->data, N_VGetArrayPointer(sundials->data->yp));
        // minus sign is because residual is -g(t, y)
        N_VScale(-1.0, sundials->data->yp, sundials->data->yp);
    }

    realtype *output;
    realtype *doutput;
    int tensor_size;
    get_out(sundials->model->data , &output, &tensor_size);
    get_out(sundials->model->data_sens, &doutput, &tensor_size);

    // reinit solve
    retval = IDAReInit(sundials->ida_mem, t0, sundials->data->yy, sundials->data->yp);
    if (check_retval(&retval, "IDAReInit", 1)) return(1);

    // reinit sens
    if (fwd_sens) {
        retval = IDASensReInit(sundials->ida_mem, IDA_SIMULTANEOUS, &sundials->data->yyS, &sundials->data->ypS);
        if (check_retval(&retval, "IDASensReInit", 1)) return(1);
    }

    if (!sundials->data->is_ode) {
        // calculate consistent initial conditions
        retval = IDACalcIC(sundials->ida_mem, IDA_YA_YDP_INIT, Vector_get(times_vec, 1));
        if (check_retval(&retval, "IDACalcIC", 1)) return(1);

        retval = IDAGetConsistentIC(sundials->ida_mem, sundials->data->yy, sundials->data->yp);
        if (check_retval(&retval, "IDAGetConsistentIC", 1)) return(1);
    }

    // if debug output y and yp
    if (sundials->data->options->debug) {
        printf("y0 = [");
        for (int j = 0; j < number_of_states; j++) {
            printf("%f ", N_VGetArrayPointer(sundials->data->yy)[j]);
        }
        printf("]\n");
        printf("yp0 = [");
        for (int j = 0; j < number_of_states; j++) {
            printf("%f ", N_VGetArrayPointer(sundials->data->yp)[j]);
        }
        printf("]\n");
    }

    if (fwd_sens) {
        calc_out_grad(t0, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yyS), N_VGetArrayPointer(sundials->data->yp), N_VGetArrayPointer(sundials->data->ypS), sundials->model->data, sundials->model->data_sens);
    } else {
        calc_out(t0, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yp), sundials->model->data);
    }

    // init output and save initial state
    Vector_resize(outputs_vec, 0);
    for (int j = 0; j < number_of_outputs; j++) {
        Vector_push(outputs_vec, output[j]);
    }
    if (fwd_sens) {
        Vector_resize(doutputs_vec, 0);
        for (int j = 0; j < number_of_outputs; j++) {
            Vector_push(doutputs_vec, doutput[j]);
        }
    }

    realtype t_final = Vector_get(times_vec, times_vec->len - 1);
    int itask = IDA_ONE_STEP;
    if (sundials->data->options->fixed_times) {
        itask = IDA_NORMAL;
    } else {
        retval = IDASetStopTime(sundials->ida_mem, t_final);
    }

    if (!sundials->data->options->fixed_times) {
        // if using solver times save initial time point and get rid of the rest
        Vector_resize(times_vec, 1);
    }

    int i = 0;

    // if not using fixed times we need to request the final time point
    realtype t_next = t_final;
    while(1) {
        // advance to next time point
        i++;

        // check that we don't exceed the maximum number of output steps
        if (sundials->data->options->mxoutsteps > 0 && i > sundials->data->options->mxoutsteps) {
            fprintf(stderr, "Exceeded maximum number of output steps %d", sundials->data->options->mxoutsteps);
            return(1);
        }

        // if using fixed times set next time point
        if (sundials->data->options->fixed_times) {
            t_next = Vector_get(times_vec, i);
        }

        // solve up to next/final time point
        realtype tret;
        retval = IDASolve(sundials->ida_mem, t_next, &tret, sundials->data->yy, sundials->data->yp, itask);
        if (check_retval(&retval, "IDASolve", 1)) return(1);

        // if no residual evaluations were performed, or if the last one was not at the returned solution we need to call residual manually
        // to get the correct output
        residual(tret, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yp), sundials->model->data, N_VGetArrayPointer(sundials->data->tmp));

        // if debug output y and yp
        if (sundials->data->options->debug) {
            printf("tret = %f\n", tret);
            printf("y = [");
            for (int j = 0; j < number_of_states; j++) {
                printf("%f ", N_VGetArrayPointer(sundials->data->yy)[j]);
            }
            printf("]\n");
            printf("yp = [");
            for (int j = 0; j < number_of_states; j++) {
                printf("%f ", N_VGetArrayPointer(sundials->data->yp)[j]);
            }
            printf("]\n");
        }

        // get output (calculated into output and doutput array)
        if (fwd_sens) {
            int retval_fwd_sens = IDAGetSensDky1(sundials->ida_mem, tret, 0, 0, sundials->data->yyS);
            if (check_retval(&retval_fwd_sens, "IDAGetSens1", 1)) return(1);

            calc_out_grad(tret, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yyS), N_VGetArrayPointer(sundials->data->yp), N_VGetArrayPointer(sundials->data->ypS), sundials->model->data, sundials->model->data_sens);
        } else {
            calc_out(tret, N_VGetArrayPointer(sundials->data->yy), N_VGetArrayPointer(sundials->data->yp), sundials->model->data);
        }

        // save output
        for (int j = 0; j < number_of_outputs; j++) {
            Vector_push(outputs_vec, output[j]);
        }
        if (fwd_sens) {
            for (int j = 0; j < number_of_outputs; j++) {
                Vector_push(doutputs_vec, doutput[j]);
            }
        }

        // if using solver times save time point
        if (!sundials->data->options->fixed_times) {
            Vector_push(times_vec, tret);
        }
        
        // if finished or errored break
        if (tret >= t_final || retval == IDA_ROOT_RETURN) {
            break;
        }
    }
    
    if (sundials->data->options->print_stats) {
        int klast, kcur;
        long int netfails, nlinsetups, nfevals, nsteps;
        realtype hinused, hlast, hcur, tcur;
        retval = IDAGetIntegratorStats(sundials->ida_mem, &nsteps, &nfevals, &nlinsetups, &netfails, &klast, &kcur, &hinused, &hlast, &hcur, &tcur);
        if (check_retval(&retval, "IDAGetIntegratorStats", 1)) return(1);

        long int nniters, nncfails;
        retval = IDAGetNonlinSolvStats(sundials->ida_mem, &nniters, &nncfails);
        if (check_retval(&retval, "IDAGetNonlinSolvStats", 1)) return(1);

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

Sundials *Sundials_create(void) {
    Sundials *sundials = malloc(sizeof(Sundials));
    sundials->data = malloc(sizeof(SundialsData));
    sundials->model = malloc(sizeof(ModelData));

    int number_of_inputs = 0;
    int number_of_outputs = 0;
    int number_of_states = 0;
    int number_of_data = 0;
    int number_of_stop = 0;
    get_dims(&number_of_states, &number_of_inputs, &number_of_outputs, &number_of_data, &number_of_stop);
    
    sundials->data->number_of_inputs = number_of_inputs;
    sundials->data->number_of_outputs = number_of_outputs;
    sundials->data->number_of_states = number_of_states;
    sundials->data->number_of_data = number_of_data;
    sundials->data->number_of_stop = number_of_stop;
    sundials->data->is_ode = false;

    sundials->model->data = malloc(number_of_data * sizeof(realtype));
    sundials->model->data_jacobian = malloc(number_of_data * sizeof(realtype));
    sundials->model->data_sens = malloc(number_of_data * sizeof(realtype));
    // initialise data to zero
    for (int i = 0; i < number_of_data; i++) {
        sundials->model->data[i] = 0;
        sundials->model->data_jacobian[i] = 0;
        sundials->model->data_sens[i] = 0;
    }
    return sundials;
}

void Sundials_destroy(Sundials *sundials) {
    SUNLinSolFree(sundials->data->sundials_linear_solver);
    SUNMatDestroy(sundials->data->sundials_jacobian);
    N_VDestroy(sundials->data->yy);
    N_VDestroy(sundials->data->yp);
    N_VDestroy(sundials->data->tmp);
    N_VDestroy(sundials->data->avtol);
    N_VDestroy(sundials->data->id);
    IDAFree(&(sundials->ida_mem));
    SUNContext_Free(&sundials->sunctx);
    free(sundials->model->data);
    free(sundials->data);
    free(sundials->model);
    free(sundials);
}

Options *Options_create(void) {
    Options *options = malloc(sizeof(Options));
    options->print_stats = 0;
    options->fixed_times = 0;
    options->jacobian = DENSE_JACOBIAN;
    options->linear_solver = LINEAR_SOLVER_DENSE;
    options->preconditioner = PRECON_NONE;
    options->linsol_max_iterations = 0;
    options->rtol = 1e-6;
    options->atol = 1e-6;
    options->mxsteps = 500;
    options->min_step = 0;
    options->max_step = FLT_MAX;
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

void Options_set_mxsteps(Options *options, const int mxsteps) {
    options->mxsteps = mxsteps;
}

int Options_get_mxsteps(Options *options) {
    return options->mxsteps;
}

void Options_set_min_step(Options *options, const realtype min_step) {
    options->min_step = min_step;
}

realtype Options_get_min_step(Options *options) {
    return options->min_step;
}

void Options_set_max_step(Options *options, const realtype max_step) {
    options->max_step = max_step;
}

realtype Options_get_max_step(Options *options) {
    return options->max_step;
}

int Options_get_fixed_times(Options *options) {
    return options->fixed_times;
}

int Options_get_print_stats(Options *options) {
    return options->print_stats;
}

int Options_get_fwd_sens(Options *options) {
    return options->fwd_sens;
}

void Options_set_fwd_sens(Options *options, const int fwd_sens) {
    options->fwd_sens = fwd_sens;
}

void Options_set_jacobian(Options *options, const int jacobian) {
    options->jacobian = jacobian;
}

int Options_get_jacobian(Options *options) {
    return options->jacobian;
}

void Options_set_linear_solver(Options *options, const int linear_solver) {
    options->linear_solver = linear_solver;
}

int Options_get_linear_solver(Options *options) {
    return options->linear_solver;
}

void Options_set_preconditioner(Options *options, const int preconditioner) {
    options->preconditioner = preconditioner;
}

int Options_get_preconditioner(Options *options) {
    return options->preconditioner;
}

void Options_set_linsol_max_iterations(Options *options, const int linsol_max_iterations) {
    options->linsol_max_iterations = linsol_max_iterations;
}

int Options_get_linsol_max_iterations(Options *options) {
    return options->linsol_max_iterations;
}

void Options_set_rtol(Options *options, const realtype rtol) {
    options->rtol = rtol;
}

realtype Options_get_rtol(Options *options) {
    return options->rtol;
}

void Options_set_atol(Options *options, const realtype atol) {
    options->atol = atol;
}

realtype Options_get_atol(Options *options) {
    return options->atol;
}

int Options_get_debug(Options *options) {
    return options->debug;
}

void Options_set_debug(Options *options, const int debug) {
    options->debug = debug;
}

int Options_get_max_out_steps(Options *options) {
    return options->mxoutsteps;
}

void Options_set_max_out_steps(Options *options, const int max_out_steps) {
    options->mxoutsteps = max_out_steps;
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

Vector *Vector_create_and_fill(int len, realtype value) {
    Vector *vector = Vector_create(len);
    for (int i = 0; i < len; i++) {
        vector->data[i] = value;
    }
    return vector;
}

VectorInt *VectorInt_create_and_fill(int len, int value) {
    VectorInt *vector = VectorInt_create(len);
    for (int i = 0; i < len; i++) {
        vector->data[i] = value;
    }
    return vector;
}

VectorInt *VectorInt_create_with_capacity(int len, int capacity) {
    if (capacity < len) {
        capacity = len;
    }
    if (capacity < 1) {
        capacity = 1;
    }
    VectorInt *vector = malloc(sizeof(VectorInt));
    vector->data = (int *)malloc(capacity * sizeof(int));
    vector->len = len;
    vector->capacity = capacity;
    return vector;
}

VectorInt *VectorInt_create(int len) {
    return VectorInt_create_with_capacity(len, len);
}

void VectorInt_destroy(VectorInt *vector) {
    free(vector->data);
    free(vector);
}

void VectorInt_push(VectorInt *vector, int value) {
    if (vector->len == vector->capacity) {
        vector->capacity *= 2;
        vector->data = realloc(vector->data, vector->capacity * sizeof(int));
    }
    vector->data[vector->len++] = value;
}

void VectorInt_resize(VectorInt *vector, int len) {
    if (len > vector->capacity) {
        vector->capacity = len;
        vector->data = realloc(vector->data, vector->capacity * sizeof(int));
    }
    vector->len = len;
}

void Vector_printf(Vector *vector) {
    printf("[");
    for (int i = 0; i < vector->len; i++) {
        printf("%f", vector->data[i]);
        if (i < vector->len - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}


