#include "lib.h"
#include <math.h>


/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

int main() {
    int retval = 0;
    Sundials* sundials = Sundials_create();
    Options* options = Options_create();
    retval = Sundials_init(sundials, options);
    if (retval != 0) {
        printf("Error in Sundials_init: %d\n", retval);
        return(retval);
    }

    int number_of_inputs = get_number_of_inputs();
    int number_of_outputs = get_number_of_outputs();
    if (number_of_inputs != 2) {
        printf("Error: number of inputs is %d, but should be 2\n", number_of_inputs);
        return(1);
    }
    if (number_of_outputs != 1) {
        printf("Error: number of outputs is %d, but should be 1\n", number_of_outputs);
        return(1);
    }

    int number_of_times = 5;
    Vector *times = Vector_linspace_create(0.0, 1.0, number_of_times);
    Vector *inputs = Vector_create(number_of_inputs);
    Vector *outputs = Vector_create(number_of_outputs * number_of_times);

    const realtype r = 1.0;
    const realtype k = 1.0;
    const realtype y0 = 1.0;
    inputs->data[0] = r; // r
    inputs->data[1] = k; // k

    retval = Sundials_solve(sundials, times->data, number_of_times, inputs->data, outputs->data);
    if (retval != 0) {
        printf("Error in Sundials_solve: %d\n", retval);
        return(retval);
    }

    Vector *y_check = Vector_create(number_of_outputs * number_of_times);
    for (int i = 0; i < number_of_times; i++) {
        realtype t = times->data[i];
        for (int j = 0; j < number_of_outputs; j++) {
            y_check->data[i * number_of_outputs + j] = k / ((k - y0) * (-r * t) / y0 + 1.);
        }
    }

    for (int i = 0; i < number_of_outputs * number_of_times; i++) {
        if (fabs(y_check->data[i] - outputs->data[i]) > 1e-5) {
            printf("Error in output %d: %f != %f\n", i, y_check->data[i], outputs->data[i]);
            retval = 1;
        }
    }

    Sundials_destroy(sundials);
    Options_destroy(options);
    Vector_destroy(times);
    Vector_destroy(inputs);
    Vector_destroy(outputs);
    Vector_destroy(y_check);

    return(retval);
}
