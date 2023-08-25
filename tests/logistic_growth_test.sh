cat >logistic_growth_expected.txt <<EOL
1.000000,2.000000
1.049956,2.099911
1.099663,2.199325
1.148880,2.297759
EOL

./logistic_growth_test -i 1,2 -t 0,0.1,0.2,0.3 > logistic_growth_output.txt

if cmp -s "logistic_growth_output.txt" "logistic_growth_expected.txt"; then
  exit 0
else
  exit 1
fi
