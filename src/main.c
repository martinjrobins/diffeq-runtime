#include "lib.h"
#include <math.h>
#include <argparse.h>
#include <string.h>

static const char *const usages[] = {
    "diffeq [options] [[--] args]",
    "diffeq [options]",
    NULL,
};


/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

int main(int argc, const char *argv[]) {

    const char *path = NULL;
    char *inputs_str = NULL;
    struct argparse_option argparse_options[] = {
        OPT_HELP(),
        OPT_STRING('c', "config", &path, "path to configuration file", NULL, 0, 0),
        OPT_STRING('i', "inputs", &inputs_str, "input vector in csv format", NULL, 0, 0),
        OPT_END(),
    };

    struct argparse argparse;
    argparse_init(&argparse, argparse_options, usages, 0);
    argparse_describe(&argparse, "\nA brief description of what the program does and how it works.", "\nAdditional description of the program after the description of the arguments.");
    argc = argparse_parse(&argparse, argc, argv);


    int number_of_inputs = 0;
    int number_of_outputs = 0;
    int number_of_states = 0;
    int data_len = 0;
    int indices_len = 0;
    get_dims(&number_of_inputs, &number_of_outputs, &number_of_states, &data_len, &indices_len);

    /* read in input */
    char *buffer;
    Vector *inputs = Vector_create(number_of_inputs);
    const size_t token_buffer_len = 20;
    if (inputs_str == NULL) {
        size_t buffer_len = number_of_inputs * 20;
        buffer = (char *) malloc(buffer_len * sizeof(char));
        getline(&buffer, &buffer_len, stdin);
    } else {
        buffer = inputs_str;
    }
    char *token = strtok(buffer, ",");
    int n = 0;
    while(token != NULL) {
        inputs->data[n] = atof(token);
        n++;
        token = strtok(NULL, ",");
    }
    if (n != number_of_inputs) {
        printf("Error: number of inputs is %d, but should be %d\n", n, number_of_inputs);
        return(1);
    }

    int retval = 0;
    Sundials* sundials = Sundials_create();
    Options* options = Options_create();
    retval = Sundials_init(sundials, options);
    if (retval != 0) {
        printf("Error in Sundials_init: %d\n", retval);
        return(retval);
    }

    int number_of_times = 5;
    Vector *times = Vector_linspace_create(0.0, 1.0, number_of_times);
    Vector *outputs = Vector_create(number_of_outputs * number_of_times);

    const realtype r = inputs->data[0];
    const realtype k = inputs->data[1];
    const realtype y0 = 1.0;
    printf("r = %f, k = %f, y0 = %f\n", r, k, y0);

    retval = Sundials_solve(sundials, times->data, number_of_times, inputs->data, outputs->data);
    if (retval != 0) {
        printf("Error in Sundials_solve: %d\n", retval);
        return(retval);
    }

    /* use analytical expression */
    Vector *y_check = Vector_create(number_of_outputs * number_of_times);
    for (int i = 0; i < number_of_times; i++) {
        realtype t = times->data[i];
        y_check->data[i * number_of_outputs + 0] = k / ((k - y0) * exp(-r * t) / y0 + 1.);
        y_check->data[i * number_of_outputs + 1] = 2 * k / ((k - y0) * exp(-r * t) / y0 + 1.);
    }

    /* check that outputs are correct */
    for (int i = 0; i < number_of_outputs * number_of_times; i++) {
        if (fabs(y_check->data[i] - outputs->data[i]) > 1e-5) {
            printf("Error in output %d: %f != %f\n", i, y_check->data[i], outputs->data[i]);
            retval = 1;
        }
    }

    /* write output to stdout in csv format */
    for (int i = 0; i < number_of_times; i++) {
        for (int j = 0; j < number_of_outputs; j++) {
            printf("%f", outputs->data[i * number_of_outputs + j]);
            if (j < number_of_outputs - 1) {
                printf(",");
            } else {
                printf("\n");
            }
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
