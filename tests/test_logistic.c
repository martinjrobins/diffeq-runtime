#include "unity.h"
#include "diffeq.h"

// bring in the model
#include "logistic_growth.c"

Sundials* sundials;
Options* options;
Vector *times;
Vector *inputs;
Vector *outputs;
Vector *dinputs;
Vector *doutputs;
int retval;

void setUp(void) {

  int number_of_inputs = 2;
  inputs = Vector_create(number_of_inputs);
  dinputs = Vector_create(number_of_inputs);
  times = Vector_create(2);
  sundials = Sundials_create();
  outputs = Vector_create(0);
  doutputs = Vector_create(0);
}

void tearDown(void) {
  Sundials_destroy(sundials);
  Options_destroy(options);
  Vector_destroy(times);
  Vector_destroy(inputs);
  Vector_destroy(dinputs);
  Vector_destroy(outputs);
  Vector_destroy(doutputs);
}

void test_solve_twice(void)
{
  inputs->data[0] = 1;
  inputs->data[1] = 2;
  dinputs->data[0] = 1;
  dinputs->data[1] = 0;
  Vector_resize(times, 4);
  times->data[0] = 0.0;
  times->data[1] = 0.1;
  times->data[2] = 0.2;
  times->data[3] = 0.3;
  options = Options_create();
  options->fixed_times = 1;
  options->fwd_sens = 1;
  retval = Sundials_init(sundials, options);
  TEST_ASSERT_EQUAL(retval, 0);
  TEST_ASSERT_EQUAL(sundials->data->number_of_stop, 1);

  retval = Sundials_solve(sundials, times, inputs, dinputs, outputs, doutputs);
  TEST_ASSERT_EQUAL(retval, 0);

  const realtype expected_outputs[] = {
    1.000000,2.000000,
    1.049955,2.099911,
    1.099664,2.199329,
    1.148881,2.297763
  };
  const realtype expected_doutputs[] = {
    0.000000,0.000000,
    0.049955,0.099911,
    0.098995,0.197990,
    0.146666,0.293331
  };

  const int noutputs = 2;
  TEST_ASSERT_EQUAL(outputs->len, noutputs * times->len);
  TEST_ASSERT_EQUAL(doutputs->len, noutputs * times->len);
  TEST_ASSERT_DOUBLE_ARRAY_WITHIN(0.001, expected_outputs, outputs->data, noutputs * times->len);
  TEST_ASSERT_DOUBLE_ARRAY_WITHIN(0.001, expected_doutputs, doutputs->data, noutputs * times->len);

  // test inputs have not been modified
  TEST_ASSERT_EQUAL(inputs->len, 2);
  TEST_ASSERT_EQUAL_FLOAT(inputs->data[0], 1);
  TEST_ASSERT_EQUAL_FLOAT(inputs->data[1], 2);
  TEST_ASSERT_EQUAL(dinputs->len, 2);
  TEST_ASSERT_EQUAL_FLOAT(dinputs->data[0], 1);
  TEST_ASSERT_EQUAL_FLOAT(dinputs->data[1], 0);

  // test a repeated solve does not change the outputs
  retval = Sundials_solve(sundials, times, inputs, dinputs, outputs, doutputs);
  TEST_ASSERT_EQUAL(retval, 0);

  TEST_ASSERT_EQUAL(outputs->len, noutputs * times->len);
  TEST_ASSERT_EQUAL(doutputs->len, noutputs * times->len);
  TEST_ASSERT_DOUBLE_ARRAY_WITHIN(0.001, expected_outputs, outputs->data, noutputs * times->len);
  TEST_ASSERT_DOUBLE_ARRAY_WITHIN(0.001, expected_doutputs, doutputs->data, noutputs * times->len);
}

void test_solve_klu(void)
{
  inputs->data[0] = 1;
  inputs->data[1] = 2;
  dinputs->data[0] = 1;
  dinputs->data[1] = 0;
  Vector_resize(times, 4);
  times->data[0] = 0.0;
  times->data[1] = 0.1;
  times->data[2] = 0.2;
  times->data[3] = 0.3;
  options = Options_create();
  options->fixed_times = 1;
  options->fwd_sens = 1;
  options->linear_solver = LINEAR_SOLVER_KLU;
  options->jacobian = SPARSE_JACOBIAN;
  retval = Sundials_init(sundials, options);
  TEST_ASSERT_EQUAL(retval, 0);

  retval = Sundials_solve(sundials, times, inputs, dinputs, outputs, doutputs);
  TEST_ASSERT_EQUAL(retval, 0);

  const realtype expected_outputs[] = {
    1.000000,2.000000,
    1.049955,2.099911,
    1.099664,2.199329,
    1.148881,2.297763
  };
  const realtype expected_doutputs[] = {
    0.000000,0.000000,
    0.049955,0.099911,
    0.098995,0.197990,
    0.146666,0.293331
  };

  const int noutputs = 2;
  TEST_ASSERT_EQUAL(outputs->len, noutputs * times->len);
  TEST_ASSERT_EQUAL(doutputs->len, noutputs * times->len);
  TEST_ASSERT_DOUBLE_ARRAY_WITHIN(0.001, expected_outputs, outputs->data, noutputs * times->len);
  TEST_ASSERT_DOUBLE_ARRAY_WITHIN(0.001, expected_doutputs, doutputs->data, noutputs * times->len);
}

void test_stop(void)
{
  VECTOR_GET(inputs, 0) = 1;
  VECTOR_GET(inputs, 1) = 2;
  VECTOR_GET(dinputs, 0) = 1;
  VECTOR_GET(dinputs, 1) = 0;
  Vector_resize(times, 2);
  VECTOR_GET(times, 0) = 0.0;
  VECTOR_GET(times, 1) = 1.0;
  options = Options_create();
  retval = Sundials_init(sundials, options);
  TEST_ASSERT_EQUAL(retval, 0);

  retval = Sundials_solve(sundials, times, inputs, dinputs, outputs, doutputs);
  TEST_ASSERT_EQUAL(retval, 0);

  const int noutputs = 2;
  const int ntimes = VECTOR_LEN(times);
  TEST_ASSERT_LESS_THAN(1.0, VECTOR_GET(times, ntimes - 1));
  TEST_ASSERT_EQUAL(1.2, VECTOR_GET(outputs, noutputs * (ntimes - 1) ));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_solve_twice);
    RUN_TEST(test_solve_klu);
    RUN_TEST(test_stop);
    return UNITY_END();
}
