#include "diffeq.h"

//in = [r, k]
//r { 1 }
//k { 1 }
//u_i {
//    y = 1,
//    z = 0,
//}
//dudt_i {
//    dydt = 0,
//    dzdt = 0,
//}
//F_i {
//    dydt,
//    0,
//}
//G_i {
//    () * (1 - (y / k)),
//    (2 * y) - z,
//}
//out_i {
//    y,
//    z,
//}
//
// data = [r, k, F_0, F_1, G_0, G_1, out_0, out_1]
#define Y u[0]
#define Z u[1]
#define DYDT up[0]
#define DZDT up[1]
#define R data[0]
#define K data[1]
#define F_0 data[2]
#define F_1 data[3]
#define G_0 data[4]
#define G_1 data[5]
#define OUT_0 data[6]
#define OUT_1 data[7]

#define DY du[0]
#define DZ du[1]
#define DDYDT dup[0]
#define DDZDT dup[1]
#define DR ddata[0]
#define DK ddata[1]
#define DF_0 ddata[2]
#define DF_1 ddata[3]
#define DG_0 ddata[4]
#define DG_1 ddata[5]
#define DOUT_0 ddata[6]
#define DOUT_1 ddata[7]


void rhs(const realtype t, const realtype* u, realtype* data, realtype* rr) {
    rr[0] = (R * Y) * (1 - (Y / K));
    rr[1] = (2 * Y) - Z;
}

void rhs_grad(const realtype t, const realtype* u, const realtype* du, realtype* data, realtype* ddata, realtype* rr, realtype* drr) {
    drr[0] = (DR * Y) * (1 - (Y / K)) + (R * DY) * (1 - (Y / K)) + (R * Y) * (0 - (DY / K)) + (R * Y) * (0 + (Y * DK / (K * K)));
    drr[1]= (2 * DY) - DZ;
}

void mass(const realtype t, const realtype* up, realtype* data, realtype* rr) {
    rr[0] = DYDT;
    rr[1] = 0;
}

void mass_grad(const realtype t, const realtype* up, const realtype* dup, realtype* data, realtype* ddata, realtype* rr, realtype* drr) {
    drr[0] = DDYDT;
    drr[1] = 0;
}

void set_u0(realtype* u, realtype* data) {
    Y = 1;
    Z = 0;
}
void set_u0_grad(realtype* u, realtype* du, realtype* data, realtype* ddata) {
    Y = 1;
    Z = 0;
    DY = 0;
    DZ = 0;
}

void calc_out(const realtype t, const realtype* u, realtype* data) {
    OUT_0 = Y;
    OUT_1 = Z;
}

void calc_out_grad(const realtype t, const realtype* u, const realtype* du, realtype* data, realtype* ddata) {
    OUT_0 = Y;
    OUT_1 = Z;
    DOUT_0 = DY;
    DOUT_1 = DZ;
}

void calc_stop(const realtype t, const realtype* u, realtype* data, realtype* stop) {
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
    *states = 2;
    *inputs = 2;
    *outputs = 2;
    *data = 8;
    *stop = 1;
    *has_mass = 1;
}
void set_id(realtype* id) {
    id[0] = 1;
    id[1] = 0;
}
void get_out(const realtype* data, realtype** tensor_data, int* tensor_size) {
    *tensor_data = (realtype*) &OUT_0;
    *tensor_size = 2;
}

