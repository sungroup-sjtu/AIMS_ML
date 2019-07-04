#!/usr/bin/env python3

import os

# WYZ dataset
# cmd1 = 'python gen-fp.py -i ../data/alkanes-npt-2018v3.txt -e morgan1,simple -o fp --svg'
# cmd2 = 'python split-data.py -i ../data/alkanes-npt-2018v3.txt -o fp'
# cmd3 = 'python train.py -i ../data/alkanes-npt-2018v3.txt -t Cp -f fp/fp_morgan1,fp/fp_simple -p fp/part-1.txt -o out'
# cmd4 = 'python predict.py -d out -e predefinedmorgan1,simple -i C1CCCCCCC1,314,1'

# NIST Tc for CH
cmd1 = 'python gen-fp.py -i ../data/nist-CH-tc.txt -e morgan1,simple -o fp-ch-tc'
cmd2 = 'python split-data.py -i ../data/nist-CH-tc.txt -o fp-ch-tc'
cmd3 = 'python train.py -i ../data/nist-CH-tc.txt -t tc -f fp-ch-tc/fp_morgan1,fp-ch-tc/fp_simple -o out-ch-tc'
cmd4 = 'python predict.py -d out-ch-tc -e predefinedmorgan1,simple -i C1CCCCCCC1'

# NIST Tvap for CH
# cmd1 = 'python gen-fp.py -i ../data/nist-CH-tvap.txt -e morgan1,simple -o fp-ch-tvap'
# cmd2 = 'python split-data.py -i ../data/nist-CH-tvap.txt -o fp-ch-tvap'
# cmd3 = 'python train.py -i ../data/nist-CH-tvap.txt -t tvap -f fp-ch-tvap/fp_morgan1,fp-ch-tvap/fp_simple -o out-ch-tvap'

# NIST Tc for All
# cmd1 = 'python gen-fp.py -i ../data/nist-All-tc.txt -e morgan1,simple -o fp-all-tc'
# cmd2 = 'python split-data.py -i ../data/nist-All-tc.txt -o fp-all-tc'
# cmd3 = 'python train.py -i ../data/nist-All-tc.txt -t tc -f fp-all-tc/fp_morgan1,fp-all-tc/fp_simple -o out-all-tc'

# NIST Tvap for All
# cmd1 = 'python gen-fp.py -i ../data/nist-All-tvap.txt -e morgan1,simple -o fp-all-tvap'
# cmd2 = 'python split-data.py -i ../data/nist-All-tvap.txt -o fp-all-tvap'
# cmd3 = 'python train.py -i ../data/nist-All-tvap.txt -t tvap -f fp-all-tvap/fp_morgan1,fp-all-tvap/fp_simple -o out-all-tvap'

# Simu density for CH
# cmd1 = 'python gen-fp.py -i ../data/result-ML-CH-npt.txt -e morgan1,simple -o fp-ch-npt'
# cmd2 = 'python split-data.py -i ../data/result-ML-CH-npt.txt -o fp-ch-npt'
# cmd3 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f fp-ch-npt/fp_morgan1,fp-ch-npt/fp_simple -p fp-ch-npt/part-1.txt -o out-ch-density'

# Simu Cp for CH
# cmd1 = 'python gen-fp.py -i ../data/result-ML-CH-cp.txt -e morgan1,simple -o fp-ch-cp'
# cmd2 = 'python split-data.py -i ../data/result-ML-CH-cp.txt -o fp-ch-cp'
# cmd3 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f fp-ch-cp/fp_morgan1,fp-ch-cp/fp_simple -p fp-ch-cp/part-1.txt -o out-ch-cp'

# Simu Hvap for CH
# cmd1 = 'python gen-fp.py -i ../data/result-ML-CH-hvap.txt -e morgan1,simple -o fp-ch-hvap'
# cmd2 = 'python split-data.py -i ../data/result-ML-CH-hvap.txt -o fp-ch-hvap'
# cmd3 = 'python train.py -i ../data/result-ML-CH-hvap.txt -t hvap -f fp-ch-hvap/fp_morgan1,fp-ch-hvap/fp_simple -p fp-ch-hvap/part-1.txt -o out-ch-hvap'

# Simu density for All
# cmd1 = 'python gen-fp.py -i ../data/result-ML-All-npt.txt -e morgan1,simple -o fp-all-npt'
# cmd2 = 'python split-data.py -i ../data/result-ML-All-npt.txt -o fp-all-npt'
# cmd3 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f fp-all-npt/fp_morgan1,fp-all-npt/fp_simple -p fp-all-npt/part-1.txt -o out-all-density'

# Simu Cp for All
# cmd1 = 'python gen-fp.py -i ../data/result-ML-All-cp.txt -e morgan1,simple -o fp-all-cp'
# cmd2 = 'python split-data.py -i ../data/result-ML-All-cp.txt -o fp-all-cp'
# cmd3 = 'python train.py -i ../data/result-ML-All-cp.txt -t cp -f fp-all-cp/fp_morgan1,fp-all-cp/fp_simple -p fp-all-cp/part-1.txt -o out-all-cp'

# Simu Hvap for All
# cmd1 = 'python gen-fp.py -i ../data/result-ML-All-hvap.txt -e morgan1,simple -o fp-all-hvap'
# cmd2 = 'python split-data.py -i ../data/result-ML-All-hvap.txt -o fp-all-hvap'
# cmd3 = 'python train.py -i ../data/result-ML-All-hvap.txt -t hvap -f fp-all-hvap/fp_morgan1,fp-all-hvap/fp_simple -p fp-all-hvap/part-1.txt -o out-all-hvap'


if __name__ == '__main__':
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)
    os.system(cmd4)
