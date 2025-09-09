cat >logistic_growth_expected.txt <<EOL
0.000000,1.000000,2.000000
0.100000,1.049955,2.099911
0.200000,1.099664,2.199329
0.300000,1.148881,2.297763
EOL

./logistic_growth_test -i 1,2 -t 0,0.1,0.2,0.3 -f > logistic_growth_output.txt

if cmp -s "logistic_growth_output.txt" "logistic_growth_expected.txt"; then
  echo "Test passed"
  exit 0
else
  echo "Test failed"
  echo "Expected:"
  cat logistic_growth_expected.txt
  echo "Got:"
  cat logistic_growth_output.txt
  exit 1
fi
