#include "diffeq.h"
#include <math.h>
#include <argparse.h>
#include <string.h>

static const char *const usages[] = {
    "diffeq [options] [[--] args]",
    "diffeq [options]",
    NULL,
};

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

Vector *read_csv_vector(char *data, int len) {
    Vector *vec = Vector_create_with_capacity(0, MAX(len, 1));
    char *token = strtok(data, ",");
    while(token != NULL) {
        Vector_push(vec, atof(token));
        token = strtok(NULL, ",");
    }
    return vec;
}

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

int main(int argc, const char *argv[]) {

    const char *path = NULL;
    char *inputs_str = NULL;
    char *dinputs_str = NULL;
    char *times_str = NULL;
    int use_fixed_times = 0;
    struct argparse_option argparse_options[] = {
        OPT_HELP(),
        OPT_STRING('c', "config", &path, "path to configuration file", NULL, 0, 0),
        OPT_STRING('i', "inputs", &inputs_str, "input vector in csv format", NULL, 0, 0),
        OPT_STRING('d', "dinputs", &dinputs_str, "tangent of input vector in csv format", NULL, 0, 0),
        OPT_STRING('t', "times", &times_str, "times vector in csv format", NULL, 0, 0),
        OPT_BOOLEAN('f', "use_fixed_times", &use_fixed_times, "Output at the times given (if unset then solver time points are used and times must be length 2 of the form [start_time, finish_time])", NULL, 0, 0),
        OPT_END(),
    };

    struct argparse argparse;
    argparse_init(&argparse, argparse_options, usages, 0);
    argparse_describe(&argparse, "\nA brief description of what the program does and how it works.", "\nAdditional description of the program after the description of the arguments.");
    argc = argparse_parse(&argparse, argc, argv);


    int number_of_inputs = 0;
    int number_of_outputs = 0;
    int number_of_states = 0;
    int number_of_stop = 0;
    int data_len = 0;
    get_dims(&number_of_inputs, &number_of_outputs, &number_of_states, &data_len, &number_of_stop);

    /* read in input */
    if (inputs_str == NULL) {
        printf("Error: inputs not specified\n");
        return(1);
    }
    Vector *inputs = read_csv_vector(inputs_str, number_of_inputs);
    if (inputs->len != number_of_inputs) {
        printf("Error: inputs vector length (%d) does not match number of inputs (%d)\n", inputs->len, number_of_inputs);
        return(1);
    }

    int fwd_sens = 0;
    Vector *dinputs = NULL;
    if (dinputs_str != NULL) {
        dinputs = read_csv_vector(dinputs_str, number_of_inputs);
        if (dinputs->len != number_of_inputs) {
            printf("Error: dinputs vector length (%d) does not match number of inputs (%d)\n", inputs->len, number_of_inputs);
            return(1);
        }
        fwd_sens = 1;
    }

    if (times_str == NULL) {
        printf("Error: times not specified\n");
        return(1);
    }
    Vector *times = read_csv_vector(times_str, -1);
    if (times->len < 2) {
        printf("Error: times vector length (%d) must be at least 2\n", times->len);
        return(1);
    }

    int retval = 0;
    Sundials* sundials = Sundials_create();
    Options* options = Options_create();
    options->fixed_times = use_fixed_times;
    options->fwd_sens = fwd_sens;
    retval = Sundials_init(sundials, options);
    if (retval != 0) {
        printf("Error in Sundials_init: %d\n", retval);
        return(retval);
    }

    Vector *outputs = Vector_create(number_of_outputs * times->len);
    Vector *doutputs = Vector_create(number_of_outputs * times->len);

    retval = Sundials_solve(sundials, times, inputs, dinputs, outputs, doutputs);
    if (retval != 0) {
        printf("Error in Sundials_solve: %d\n", retval);
        return(retval);
    }

    /* write output to stdout in csv format */
    for (int i = 0; i < times->len; i++) {
        printf("%f,", times->data[i]);
        for (int j = 0; j < number_of_outputs; j++) {
            printf("%f", outputs->data[i * number_of_outputs + j]);
            if (fwd_sens) {
                printf(",%f", doutputs->data[i * number_of_outputs + j]);
            }
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

    return(retval);
}
