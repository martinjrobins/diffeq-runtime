#include "diffeq.h"

//in = [r, k]
//r { 1 }
//k { 1 }
//u_i {
//    y = 1,
//}
//dudt_i {
//    dydt = 0,
//}
//F_i {
//    dydt,
//}
//G_i {
//    () * (1 - (y / k)),
//}
//out_i {
//    y,
//}
//
// data = [r, k, F_0, G_0]
#define Y u[0]
#define DYDT up[0]
#define R data[0]
#define K data[1]
#define F_0 data[2]
#define G_0 data[3]

#define DY du[0]
#define DDYDT dup[0]
#define DR ddata[0]
#define DK ddata[1]
#define DF_0 ddata[2]
#define DG_0 ddata[3]


void rhs(const realtype t, const realtype* u, realtype* data, realtype* rr, uint32_t thread_id, uint32_t num_threads) {
    rr[0] = (R * Y) * (1 - (Y / K));
}

void rhs_grad(const realtype t, const realtype* u, const realtype* du, realtype* data, realtype* ddata, realtype* rr, realtype* drr, uint32_t thread_id, uint32_t num_threads) {
    drr[0] = (DR * Y) * (1 - (Y / K)) + (R * DY) * (1 - (Y / K)) + (R * Y) * (0 - (DY / K)) + (R * Y) * (0 + (Y * DK / (K * K)));
}

void mass(const realtype t, const realtype* up, realtype* data, realtype* rr, uint32_t thread_id, uint32_t num_threads) {
    rr[0] = DYDT;
}

void mass_grad(const realtype t, const realtype* up, const realtype* dup, realtype* data, realtype* ddata, realtype* rr, realtype* drr, uint32_t thread_id, uint32_t num_threads) {
    drr[0] = DDYDT;
}

void set_u0(realtype* u, realtype* data, uint32_t thread_id, uint32_t num_threads) {
    Y = 1;
}
void set_u0_grad(realtype* u, realtype* du, realtype* data, realtype* ddata, uint32_t thread_id, uint32_t num_threads) {
    Y = 1;
    DY = 0;
}

void calc_out(const realtype t, const realtype* u, realtype* data, realtype *out, uint32_t thread_id, uint32_t num_threads) {
    out[0] = Y;
}

void calc_out_grad(const realtype t, const realtype* u, const realtype* du, realtype* data, realtype* ddata, realtype *out, realtype *dout, uint32_t thread_id, uint32_t num_threads) {
    out[0] = Y;
    dout[0] = DY;
}

void calc_stop(const realtype t, const realtype* u, realtype* data, realtype* stop, uint32_t thread_id, uint32_t num_threads) {
    stop[0] = Y - 1.2;
}

void set_inputs(const realtype* inputs, realtype* data) {
    R = inputs[0];
    K = inputs[1];
}
void set_inputs_grad(const realtype* inputs, const realtype* dinputs, realtype* data, realtype* ddata) {
    R = inputs[0];
    K = inputs[1];
    DR = dinputs[0];
    DK = dinputs[1];
}

void get_dims(int* states, int* inputs, int* outputs, int* data, int* stop, int* has_mass) {
    *states = 1;
    *inputs = 2;
    *outputs = 1;
    *data = 5;
    *stop = 1;
    *has_mass = 0;
}
void set_id(realtype* id) {
    id[0] = 1;
}

