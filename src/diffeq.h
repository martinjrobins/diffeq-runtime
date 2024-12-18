#pragma once

#include <stdio.h>

#include <idas/idas.h>               /* prototypes for IDAS fcts., consts.  */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector            */

#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunmatrix/sunmatrix_sparse.h> /* access to sparse SUNMatrix           */

#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunlinsol/sunlinsol_band.h> /* access to dense linear solver          */
#include <sunlinsol/sunlinsol_spbcgs.h> /* access to spbcgs iterative linear solver          */
#include <sunlinsol/sunlinsol_spfgmr.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_sptfqmr.h>
#include <sunlinsol/sunlinsol_klu.h> /* access to sparse linear solver         */

#ifdef EMSCRIPTEN
  #include <emscripten.h>
#else
  #define EMSCRIPTEN_KEEPALIVE
#endif

#ifdef NDEBUG
  #define DEBUG(x)
#else
  #define DEBUG(x) do { printf("%s:%d ", __FILE__, __LINE__); printf(x); putchar('\n'); } while (0)
#endif

#ifdef NDEBUG
  #define DEBUG_VECTOR(vector)
  #define DEBUG_VECTORn(vector)
#else

  #define DEBUG_VECTORn(vector, N) {\
    printf("%s[n=%d] = [", #vector, N); \
    realtype* array_ptr = N_VGetArrayPointer(vector); \
    for (int i = 0; i < N; i++) { \
      printf("%f", array_ptr[i]); \
      if (i < N-1) { \
        printf(", "); \
      } \
    } \
    printf("]\n"); \
  }

  #define DEBUG_VECTOR(vector) {\
    printf("%s = [", #vector); \
    realtype* array_ptr = N_VGetArrayPointer(vector); \
    int N = N_VGetLength(vector); \
    for (int i = 0; i < N; i++) { \
      printf("%f", array_ptr[i]); \
      if (i < N-1) { \
        printf(", "); \
      } \
    } \
    printf("]\n"); \
  }

#endif

typedef struct Vector {
  realtype *data;
  int len;
  int capacity;
} Vector;

typedef struct VectorInt {
  int *data;
  int len;
  int capacity;
} VectorInt;

EMSCRIPTEN_KEEPALIVE void Vector_destroy(Vector *vector);
EMSCRIPTEN_KEEPALIVE void VectorInt_destroy(VectorInt *vector);
EMSCRIPTEN_KEEPALIVE Vector *Vector_linspace_create(const realtype start, const realtype stop, const int len);
EMSCRIPTEN_KEEPALIVE Vector *Vector_create(const int len);
EMSCRIPTEN_KEEPALIVE VectorInt *VectorInt_create(const int len);
EMSCRIPTEN_KEEPALIVE Vector *Vector_create_and_fill(const int len, const realtype value);
EMSCRIPTEN_KEEPALIVE VectorInt *VectorInt_create_and_fill(const int len, const int value);
EMSCRIPTEN_KEEPALIVE realtype Vector_get(Vector *vector, const int index);
EMSCRIPTEN_KEEPALIVE realtype *Vector_get_data(Vector *vector);
EMSCRIPTEN_KEEPALIVE Vector *Vector_create_with_capacity(const int len, const int capacity);
EMSCRIPTEN_KEEPALIVE VectorInt *VectorInt_create_with_capacity(const int len, const int capacity);
EMSCRIPTEN_KEEPALIVE void Vector_push(Vector *vector, const realtype value);
EMSCRIPTEN_KEEPALIVE void VectorInt_push(VectorInt *vector, const int value);
EMSCRIPTEN_KEEPALIVE void Vector_resize(Vector *vector, const int len);
EMSCRIPTEN_KEEPALIVE void VectorInt_resize(VectorInt *vector, const int len);
EMSCRIPTEN_KEEPALIVE int Vector_get_length(Vector *vector);
void Vector_printf(const Vector *vector);

#define VECTOR_GET(vector, index) vector->data[index]
#define VECTOR_LEN(vector) vector->len

typedef struct MatrixCSC {
  Vector *data;
  VectorInt *row_indices;
  VectorInt *col_ptrs;
  int nnz;
  int nrow;
  int ncol;
} MatrixCSC;

MatrixCSC *MatrixCSC_create(const int nrow, const int ncol);
void MatrixCSC_destroy(MatrixCSC *matrix);
void MatrixCSC_add_col(MatrixCSC *matrix, Vector *col_data);

enum LinearSolver {
  LINEAR_SOLVER_DENSE,
  LINEAR_SOLVER_KLU,
  LINEAR_SOLVER_SPBCGS,
  LINEAR_SOLVER_SPFGMR,
  LINEAR_SOLVER_SPGMR,
  LINEAR_SOLVER_SPTFQMR,
};

enum Preconditioner {
  PRECON_NONE,
  PRECON_LEFT,
  PRECON_RIGHT,
};

enum Jacobian {
  DENSE_JACOBIAN,
  SPARSE_JACOBIAN,
  MATRIX_FREE_JACOBIAN,
  NO_JACOBIAN
};

typedef struct Options {
    realtype atol;
    realtype rtol;
    int print_stats;       // 0 for false, 1 for true
    int fixed_times;      // 0 for false, 1 for true
    int fwd_sens;        // 0 for false, 1 for true
    int mxsteps;
    int mxoutsteps;
    realtype min_step;
    realtype max_step;
    enum LinearSolver linear_solver;
    enum Preconditioner preconditioner;
    enum Jacobian jacobian;
    int linsol_max_iterations;
    int debug;
} Options;

/* Options functions */

EMSCRIPTEN_KEEPALIVE Options* Options_create(void);
EMSCRIPTEN_KEEPALIVE void Options_set_fixed_times(Options *options, const int fixed_times);
EMSCRIPTEN_KEEPALIVE int Options_get_fixed_times(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_print_stats(Options *options, const int print_stats);
EMSCRIPTEN_KEEPALIVE int Options_get_print_stats(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_fwd_sens(Options *options, const int print_stats);
EMSCRIPTEN_KEEPALIVE int Options_get_fwd_sens(Options *options);
EMSCRIPTEN_KEEPALIVE int Options_get_linear_solver(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_linear_solver(Options *options, const int linear_solver);
EMSCRIPTEN_KEEPALIVE void Options_set_mxsteps(Options *options, const int mxsteps);
EMSCRIPTEN_KEEPALIVE int Options_get_mxsteps(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_min_step(Options *options, const realtype min_step);
EMSCRIPTEN_KEEPALIVE realtype Options_get_min_step(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_max_step(Options *options, const realtype max_step);
EMSCRIPTEN_KEEPALIVE realtype Options_get_max_step(Options *options);
EMSCRIPTEN_KEEPALIVE int Options_get_preconditioner(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_preconditioner(Options *options, const int preconditioner);
EMSCRIPTEN_KEEPALIVE int Options_get_jacobian(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_jacobian(Options *options, const int jacobian);
EMSCRIPTEN_KEEPALIVE void Options_set_atol(Options *options, const realtype atol);
EMSCRIPTEN_KEEPALIVE realtype Options_get_atol(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_rtol(Options *options, const realtype rtol);
EMSCRIPTEN_KEEPALIVE realtype Options_get_rtol(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_linsol_max_iterations(Options *options, const int linsol_max_iterations);
EMSCRIPTEN_KEEPALIVE int Options_get_linsol_max_iterations(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_debug(Options *options, const int debug);
EMSCRIPTEN_KEEPALIVE int Options_get_debug(Options *options);
EMSCRIPTEN_KEEPALIVE int Options_get_max_out_steps(Options *options);
EMSCRIPTEN_KEEPALIVE void Options_set_max_out_steps(Options *options, const int max_out_steps);

EMSCRIPTEN_KEEPALIVE void Options_destroy(Options *options);

typedef struct SundialsData {
    size_t number_of_states;
    size_t number_of_inputs;
    size_t number_of_stop;
    size_t number_of_outputs;
    size_t number_of_data;
    bool has_mass;
    N_Vector yy;
    N_Vector yyS;
    N_Vector yp;
    N_Vector ypS;
    N_Vector tmp;
    N_Vector avtol;
    N_Vector id;
    SUNMatrix sundials_jacobian;
    SUNLinearSolver sundials_linear_solver;
    const Options *options;
} SundialsData;

typedef struct ModelData {
  realtype* data;
  realtype* data_sens;
  realtype* data_jacobian;
} ModelData;

typedef struct Sundials {
    void* ida_mem;
    SUNContext sunctx;
    SundialsData *data;
    ModelData *model;
} Sundials;

/*  Sundials functions */

EMSCRIPTEN_KEEPALIVE Sundials *Sundials_create(void);
EMSCRIPTEN_KEEPALIVE void Sundials_destroy(Sundials *sundials);
EMSCRIPTEN_KEEPALIVE int Sundials_init(Sundials *sundials, const Options *options);
EMSCRIPTEN_KEEPALIVE int Sundials_solve(Sundials *sundials, Vector *times, const Vector *inputs, const Vector *dinputs, Vector *outputs, Vector *doutputs);
EMSCRIPTEN_KEEPALIVE int Sundials_number_of_inputs(Sundials *sundials);
EMSCRIPTEN_KEEPALIVE int Sundials_number_of_outputs(Sundials *sundials);
EMSCRIPTEN_KEEPALIVE int Sundials_number_of_states(Sundials *sundials);
MatrixCSC *Sundials_create_jacobian(Sundials *sundials);


/*
* model functions (linked in later)
*/
void rhs(const realtype t, const realtype* u, realtype* data, realtype* rr);
void rhs_grad(const realtype t, const realtype* u, const realtype* du, realtype* data, realtype* ddata, realtype* rr, realtype* drr);

void mass(const realtype t, const realtype* du, realtype* data, realtype* mm);

void set_u0(realtype* u, realtype* data);
void set_u0_grad(realtype* u, realtype* du, realtype* data, realtype* ddata);

void calc_out(const realtype t, const realtype* u, realtype* data);
void calc_out_grad(const realtype t, const realtype* u, const realtype* du, realtype* data, realtype* ddata);

void calc_stop(const realtype t, const realtype* u, realtype* data, realtype* stop);

void set_inputs(const realtype* inputs, realtype* data);
void set_inputs_grad(const realtype* inputs, const realtype* dinputs, realtype* data, realtype* ddata);

void get_dims(int* states, int* inputs, int* outputs, int* data, int* stop, int* has_mass);
void set_id(realtype* id);
void get_out(const realtype* data, realtype** tensor_data, int* tensor_size);


/* function to check function return values */

int check_retval(void *returnvalue, const char *funcname, int opt);
