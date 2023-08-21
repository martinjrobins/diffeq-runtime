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
    struct argparse_option argparse_options[] = {
        OPT_HELP(),
        OPT_STRING('c', "config", &path, "path to configuration file", NULL, 0, 0),
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
    Vector *inputs = Vector_create(number_of_inputs);
    const size_t token_buffer_len = 20;
    size_t buffer_len = number_of_inputs * 20;
    char *token_buffer = (char *) malloc(token_buffer_len * sizeof(char));
    char *buffer = (char *) malloc(buffer_len * sizeof(char));
    getline(&buffer, &buffer_len, stdin);
    char *start = buffer;
    int n = 0;
    while(start != NULL) {
        char *next = strtok(start, ",");
        memcpy(token_buffer, start, next - start);
        inputs->data[n] = atof(token_buffer);
        n++;
        if (next == NULL) {
            start = NULL;
        } else {
            start = next + 1;
        }
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
        for (int j = 0; j < number_of_outputs; j++) {
            y_check->data[i * number_of_outputs + j] = k / ((k - y0) * (-r * t) / y0 + 1.);
        }
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
            }
        }
    }

    Sundials_destroy(sundials);
    Options_destroy(options);
    Vector_destroy(times);
    Vector_destroy(inputs);
    Vector_destroy(outputs);
    Vector_destroy(y_check);
    free(buffer);
    free(token_buffer);

    return(retval);
}
