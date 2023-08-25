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

#

typedef struct Vector {
  realtype *data;
  int len;
  int capacity;
} Vector;

void Vector_destroy(Vector *vector);
Vector *Vector_linspace_create(const realtype start, const realtype stop, const int len);
Vector *Vector_create(const int len);
Vector *Vector_create_with_capacity(const int len, const int capacity);
void Vector_push(Vector *vector, const realtype value);

typedef struct Options {
    realtype atol;
    realtype rtol;
    int print_stats;       // 0 for false, 1 for true
    char* jacobian;        // C-style string
    char* linear_solver;   // C-style string
    char* preconditioner;  // C-style string
    uint32_t linsol_max_iterations;
} Options;

/* Options functions */

Options* Options_create(void);
void Options_destroy(Options *options);

typedef struct SundialsData {
    size_t number_of_states;
    size_t number_of_parameters;
    N_Vector yy;
    N_Vector yp;
    N_Vector avtol;
    N_Vector id;
    SUNMatrix jacobian;
    SUNLinearSolver linear_solver;
    const Options *options;
} SundialsData;

typedef struct ModelData {
  realtype* data;
  int* indices;
} ModelData;

typedef struct Sundials {
    void* ida_mem;
    SUNContext sunctx;
    SundialsData *data;
    ModelData *model;
} Sundials;

/*  Sundials functions */

Sundials *Sundials_create(void);
int Sundials_init(Sundials *sundials, const Options *options);
void Sundials_destroy(Sundials *sundials);
int Sundials_solve(Sundials *sundials, const realtype *times, const size_t number_of_times, const realtype *inputs, realtype *outputs);


/*
* model functions (linked in later)
*/
void residual(const realtype t, const realtype* u, const realtype* up, realtype* data, const int* indices, realtype* rr);
void set_u0(realtype* data, const int* indices, realtype* u, realtype* up);
void calc_out(const realtype t, const realtype* u, const realtype* up, realtype* data, const int* indices);
void get_dims(int* states, int* inputs, int* outputs, int* data, const int* indices);
void set_inputs(const realtype* inputs, realtype* data);
void set_id(realtype* id);
void get_out(const realtype* data, realtype** tensor_data, int* tensor_size);

/* Functions Called by the Solver */




/* function to check function return values */

int check_retval(void *returnvalue, const char *funcname, int opt);
