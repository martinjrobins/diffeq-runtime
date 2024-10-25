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
// data = [r, k, F_0, G_0, out_0]
#define Y u[0]
#define DYDT up[0]
#define R data[0]
#define K data[1]
#define F_0 data[2]
#define G_0 data[3]
#define OUT_0 data[4]

#define DY du[0]
#define DDYDT dup[0]
#define DR ddata[0]
#define DK ddata[1]
#define DF_0 ddata[2]
#define DG_0 ddata[3]
#define DOUT_0 ddata[4]


void residual(const realtype t, const realtype* u, const realtype* up, realtype* data, realtype* rr) {
    F_0 = DYDT;
    G_0 = (R * Y) * (1 - (Y / K));
    rr[0] = F_0 - G_0;
}

void residual_grad(const realtype t, const realtype* u, const realtype* du, const realtype* up, const realtype* dup, realtype* data, realtype* ddata, realtype* rr, realtype* drr) {
    DF_0 = DDYDT;
    DG_0 = (DR * Y) * (1 - (Y / K)) + (R * DY) * (1 - (Y / K)) + (R * Y) * (0 - (DY / K)) + (R * Y) * (0 + (Y * DK / (K * K)));
    drr[0] = DF_0 - DG_0;
}

void set_u0(realtype* data, realtype* u, realtype* up) {
    Y = 1;
    DYDT = 0;
}
void set_u0_grad(realtype* data, realtype* ddata, realtype* u, realtype* du, realtype* up, realtype* dup) {
    Y = 1;
    DYDT = 0;
    DY = 0;
    DDYDT = 0;
}

void calc_out(const realtype t, const realtype* u, const realtype* up, realtype* data) {
    OUT_0 = Y;
}

void calc_out_grad(const realtype t, const realtype* u, const realtype* du, const realtype* up, const realtype* dup, realtype* data, realtype* ddata) {
    OUT_0 = Y;
    DOUT_0 = DY;
}

void calc_stop(const realtype t, const realtype* u, const realtype* up, realtype* data, realtype* stop) {
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

void get_dims(int* states, int* inputs, int* outputs, int* data, int* stop) {
    *states = 1;
    *inputs = 2;
    *outputs = 1;
    *data = 5;
    *stop = 1;
}
void set_id(realtype* id) {
    id[0] = 1;
}
void get_out(const realtype* data, realtype** tensor_data, int* tensor_size) {
    *tensor_data = (realtype*) &OUT_0;
    *tensor_size = 1;
}

