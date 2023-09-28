cat >logistic_growth_expected.txt <<EOL
0.000000,1.000000,2.000000
0.000003,1.000001,2.000003
0.000006,1.000003,2.000006
0.000011,1.000006,2.000011
0.000023,1.000011,2.000023
0.000045,1.000023,2.000045
0.000091,1.000045,2.000091
0.000181,1.000091,2.000181
0.000362,1.000181,2.000362
0.000724,1.000362,2.000724
0.001448,1.000724,2.001448
0.002896,1.001448,2.002896
0.005793,1.002896,2.005793
0.011585,1.005793,2.011585
0.023170,1.011584,2.023169
0.034756,1.017375,2.034750
0.046341,1.023165,2.046329
0.057926,1.028953,2.057906
0.073680,1.036821,2.073642
0.089435,1.044685,2.089370
0.105189,1.052543,2.105086
0.120943,1.060394,2.120789
0.152452,1.076075,2.152149
0.183960,1.091718,2.183436
0.215469,1.107316,2.214632
0.246977,1.122861,2.245722
0.278486,1.138346,2.276693
0.300000,1.148881,2.297763
EOL

./logistic_growth_test -i 1,2 -t 0,0.3 > logistic_growth_output.txt

if cmp -s "logistic_growth_output.txt" "logistic_growth_expected.txt"; then
  exit 0
else
  exit 1
fi
