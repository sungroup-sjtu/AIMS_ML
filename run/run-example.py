#!/usr/bin/env python3

import os

# WYZ dataset
# cmd1 = 'python gen-fp.py -i ../data/alkanes-npt-2018v3.txt -e morgan1,simple -o fp --svg'
# cmd2 = 'python split-data.py -i ../data/alkanes-npt-2018v3.txt -o fp'
# cmd3 = 'python train.py -i ../data/alkanes-npt-2018v3.txt -t Cp -f fp/fp_morgan1,fp/fp_simple -p fp/part-1.txt -o out'
# cmd4 = 'python predict.py -d out -e predefinedmorgan1,simple -i C1CCCCCCC1,314,1'

# NIST Tc for CH
# cmd1 = 'python gen-fp.py -i ../data/nist-CH-tc.txt -e morgan1,simple -o out-ch-tc'
# cmd2 = 'python split-data.py -i ../data/nist-CH-tc.txt -o out-ch-tc'
# cmd3 = 'python train.py -i ../data/nist-CH-tc.txt -t tc -f out-ch-tc/fp_morgan1,out-ch-tc/fp_simple -o out-ch-tc --epoch 1000,2000,2000 --check 100'
# cmd4 = 'python predict.py -d out-ch-tc -e predefinedmorgan1,simple -i C1CCCCCCC1'

# NIST Tvap for CH
# cmd1 = 'python gen-fp.py -i ../data/nist-CH-tvap.txt -e morgan1,simple -o out-ch-tvap'
# cmd2 = 'python split-data.py -i ../data/nist-CH-tvap.txt -o out-ch-tvap'
# cmd3 = 'python train.py -i ../data/nist-CH-tvap.txt -t tvap -f out-ch-tvap/fp_morgan1,out-ch-tvap/fp_simple -o out-ch-tvap --epoch 1000,2000,2000 --check 100'

# NIST Tc for All
# cmd1 = 'python gen-fp.py -i ../data/nist-All-tc.txt -e morgan1,simple -o out-all-tc'
# cmd2 = 'python split-data.py -i ../data/nist-All-tc.txt -o out-all-tc'
# cmd3 = 'python train.py -i ../data/nist-All-tc.txt -t tc -f out-all-tc/fp_morgan1,out-all-tc/fp_simple -o out-all-tc --epoch 1000,2000,2000 --check 100'

# NIST Tvap for All
# cmd1 = 'python gen-fp.py -i ../data/nist-All-tvap.txt -e morgan1,simple -o out-all-tvap'
# cmd2 = 'python split-data.py -i ../data/nist-All-tvap.txt -o out-all-tvap'
# cmd3 = 'python train.py -i ../data/nist-All-tvap.txt -t tvap -f out-all-tvap/fp_morgan1,out-all-tvap/fp_simple -o out-all-tvap --epoch 1000,2000,2000 --check 100'

# Simu density for CH
cmd1 = 'python gen-fp.py -i ../data/result-ML-CH-npt.txt -e morgan1,simple -o out-ch-density'
cmd2 = 'python split-data.py -i ../data/result-ML-CH-npt.txt -o out-ch-density'
cmd3 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-1.txt -o out-ch-density'
# Simu einter for CH
cmd3 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-1.txt -o out-ch-einter'
# Simu Cp for CH
cmd1 = 'python gen-fp.py -i ../data/result-ML-CH-cp.txt -e morgan1,simple -o out-ch-cp'
cmd2 = 'python split-data.py -i ../data/result-ML-CH-cp.txt -o out-ch-cp'
cmd3 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-1.txt -o out-ch-cp'

# Simu Hvap for Ane
cmd1 = 'python gen-fp.py -i ../data/result-ML-Ane-hvap.txt -e morgan1,simple -o out-ane-hvap'
cmd2 = 'python split-data.py -i ../data/result-ML-Ane-hvap.txt -o out-ane-hvap'
cmd3 = 'python train.py -i ../data/result-ML-Ane-hvap.txt -t hvap -f out-ane-hvap/fp_morgan1,out-ane-hvap/fp_simple -p out-ane-hvap/part-1.txt -o out-ane-hvap'

# Simu density for All
cmd1 = 'python gen-fp.py -i ../data/result-ML-All-npt.txt -e morgan1,simple -o out-all-density'
cmd2 = 'python split-data.py -i ../data/result-ML-All-npt.txt -o out-all-density'
cmd3 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-density'
# Simu einter for CH
cmd3 = 'python train.py -i ../data/result-ML-All-npt.txt -t einter -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-einter'
# Simu Cp for All
cmd1 = 'python gen-fp.py -i ../data/result-ML-All-cp.txt -e morgan1,simple -o out-all-cp'
cmd2 = 'python split-data.py -i ../data/result-ML-All-cp.txt -o out-all-cp'
cmd3 = 'python train.py -i ../data/result-ML-All-cp.txt -t cp -f out-all-cp/fp_morgan1,out-all-cp/fp_simple -p out-all-cp/part-1.txt -o out-all-cp'

# CH-no-c-npt-1bar
# cmd1 = 'python gen-fp.py -i ../data/C123-npt-1bar.txt -e simple,extra -o fp-noc'
# cmd2 = 'python split-data.py -i ../data/C123-npt-1bar.txt -o fp-noc'
# cmd31 = 'python train.py -i ../data/C123-npt-1bar.txt -t einter -f fp-noc/fp_morgan1,fp-noc/fp_simple -p fp-noc/part-1.txt -o out-noc1'
# cmd32 = 'python train.py -i ../data/C123-npt-1bar.txt -t einter -f fp-noc/fp_morgan1,fp-noc/fp_simple -p fp-noc/part-2.txt -o out-noc2'
# cmd33 = 'python train.py -i ../data/C123-npt-1bar.txt -t einter -f fp-noc/fp_morgan1,fp-noc/fp_simple -p fp-noc/part-3.txt -o out-noc3'
# cmd34 = 'python train.py -i ../data/C123-npt-1bar.txt -t einter -f fp-noc/fp_morgan1,fp-noc/fp_simple -p fp-noc/part-4.txt -o out-noc4'
# cmd35 = 'python train.py -i ../data/C123-npt-1bar.txt -t einter -f fp-noc/fp_morgan1,fp-noc/fp_simple -p fp-noc/part-5.txt -o out-noc5'

# extra
# cmd1 = 'python gen-fp.py -i ../data/feedback/extra.txt -e simple,extra -o fp-extra'

# Simu density for CH
# cmd1 = 'python gen-fp.py -i ../data/result-ML-CH-npt.txt -e morgan1,simple -o fp-ch-npt-team'
# cmd12 = 'python gen-teamfp.py -i ../data/result-ML-CH-npt.txt -o fp-ch-npt-team'
# cmd2 = 'python split-data.py -i ../data/result-ML-CH-npt.txt -o fp-ch-npt-team'
# cmd3 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f fp-ch-npt-team/fp_team,fp-ch-npt-team/fp_simple -p fp-ch-npt-team/part-1.txt -o out-ch-npt-team'

# Simu Cp for CH
# cmd1 = 'python gen-fp.py -i ../data/result-ML-CH-cp.txt -e morgan1,simple -o out-ch-cp-team'
# cmd12 = 'python gen-teamfp.py -i ../data/result-ML-CH-cp.txt -o out-ch-cp-team'
# cmd2 = 'python split-data.py -i ../data/result-ML-CH-cp.txt -o out-ch-cp-team'
# cmd3 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp-team/fp_team,out-ch-cp-team/fp_simple -p out-ch-cp-team/part-1.txt -o out-ch-cp-team'

# Simu Cp for All
# cmd1 = 'python gen-fp.py -i ../data/result-ML-All-cp.txt -e morgan1,simple -o out-all-cp-team'
# cmd12 = 'python gen-teamfp.py -i ../data/result-ML-All-cp.txt -o out-all-cp-team'
# cmd2 = 'python split-data.py -i ../data/result-ML-All-cp.txt -o out-all-cp-team'
# cmd3 = 'python train.py -i ../data/result-ML-All-cp.txt -t cp -f out-all-cp-team/fp_team,out-all-cp-team/fp_simple -p out-all-cp-team/part-1.txt -o out-all-cp-team'

# Simu density for C123
# cmd1 = 'python gen-fp.py -i ../data/C123-npt-1bar.txt -e morgan1,simple -o out-c123-team'
# cmd12 = 'python gen-teamfp.py -i ../data/C123-npt-1bar.txt -o out-c123-team'
# cmd2 = 'python split-data.py -i ../data/C123-npt-1bar.txt -o out-c123-team'
# cmd3 = 'python train.py -i ../data/C123-npt-1bar.txt -t einter -f out-c123-team/fp_team,out-c123-team/fp_simple -p out-c123-team/part-2.txt -o out-c123-team-2'
# Feedback
# cmd1 = 'python gen-fp.py -i ../data/fb/C123-npt-1bar.txt -e morgan1,simple -o out-c123-team-fb'
# cmd12 = 'python gen-teamfp.py -i ../data/fb/C123-npt-1bar.txt -o out-c123-team-fb'
# cmd2 = 'python split-data.py -i ../data/fb/C123-npt-1bar.txt -o out-c123-team-fb'
# cmd3 = 'python train.py -i ../data/fb/C123-npt-1bar.txt -t einter -f out-c123-team-fb/fp_team,out-c123-team-fb/fp_simple -p out-c123-team-fb/part-2.txt -o out-c123-team-fb-2'

if __name__ == '__main__':
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)
    # os.system(cmd31)
    # os.system(cmd32)
    # os.system(cmd33)
    # os.system(cmd34)
    # os.system(cmd35)
