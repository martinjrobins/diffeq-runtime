typedef double realtype;

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
#define R data[0]
#define K data[1]
#define Y u[0]
#define Z u[1]
#define DYDT up[0]
#define DZDT up[1]
#define F_0 data[2]
#define F_1 data[3]
#define G_0 data[4]
#define G_1 data[5]
#define OUT_0 data[6]
#define OUT_1 data[7]

#define DR ddata[0]
#define DK ddata[1]
#define DY du[0]
#define DZ du[1]
#define DDYDT dup[0]
#define DDZDT dup[1]
#define DF_0 ddata[2]
#define DF_1 ddata[3]
#define DG_0 ddata[4]
#define DG_1 ddata[5]
#define DOUT_0 ddata[6]
#define DOUT_1 ddata[7]


void residual(const realtype t, const realtype* u, const realtype* up, realtype* data, const int* indices, realtype* rr) {
    F_0 = DYDT;
    F_1 = DZDT;
    G_0 = (R * Y) * (1 - (Y / K));
    G_1 = (2 * Y) - Z;
    rr[0] = F_0 - G_0;
    rr[1] = F_1 - G_1;
}

void residual_grad(const realtype t, const realtype* u, const realtype* du, const realtype* up, const realtype* dup, realtype* data, realtype* ddata, const int* indices, realtype* rr, realtype* drr) {
    DF_0 = DDYDT;
    DF_1 = DDZDT;
    DG_0 = (DR * Y) * (1 - (Y / K)) + (R * DY) * (1 - (Y / K)) + (R * Y) * (0 - (DY / K));
    DG_1 = (2 * DY) - DZ;
    drr[0] = DF_0 - DG_0;
    drr[1] = DF_1 - DG_1;
}

void set_u0(realtype* data, const int* indices, realtype* u, realtype* up) {
    Y = 1;
    Z = 0;
}
void set_u0_grad(realtype* data, realtype* ddata, const int* indices, realtype* u, realtype* du, realtype* up, realtype* dup) {
    DY = 0;
    DZ = 0;
}

void calc_out(const realtype t, const realtype* u, const realtype* up, realtype* data, const int* indices) {
    OUT_0 = Y;
    OUT_1 = Z;
}

void calc_out_grad(const realtype t, const realtype* u, const realtype* du, const realtype* up, const realtype* dup, realtype* data, realtype* ddata, const int* indices) {
    DOUT_0 = DY;
    DOUT_1 = DZ;
}

void set_inputs(const realtype* inputs, realtype* data) {
    R = inputs[0];
    K = inputs[1];
}
void set_inputs_grad(const realtype* inputs, const realtype* dinputs, realtype* data, realtype* ddata) {
    DR = dinputs[0];
    DK = dinputs[1];
}

void get_dims(int* states, int* inputs, int* outputs, int* data, const int* indices) {
    *states = 2;
    *inputs = 2;
    *outputs = 2;
    *data = 8;
}
void set_id(realtype* id) {
    id[0] = 0;
    id[1] = 1;
}
void get_out(const realtype* data, realtype** tensor_data, int* tensor_size) {
    *tensor_data = (realtype*) &OUT_0;
    *tensor_size = 2;
}

