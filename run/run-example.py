#!/usr/bin/env python3

import os

# NIST Tc for All
# cmd1 = 'python gen-fp.py -i ../data/nist-All-tc.txt -e morgan1,simple -o out-all-tc'
# cmd2 = 'python split-data.py -i ../data/nist-All-tc.txt -o out-all-tc'
# cmd3 = 'python train.py -i ../data/nist-All-tc.txt -t tc -f out-all-tc/fp_morgan1,out-all-tc/fp_simple -o out-all-tc/out'

# NIST Tvap for All
# cmd1 = 'python gen-fp.py -i ../data/nist-All-tvap.txt -e morgan1,simple -o out-all-tvap'
# cmd2 = 'python split-data.py -i ../data/nist-All-tvap.txt -o out-all-tvap'
# cmd3 = 'python train.py -i ../data/nist-All-tvap.txt -t tvap -f out-all-tvap/fp_morgan1,out-all-tvap/fp_simple -o out-all-tvap/out'

# Simu density for CH
cmd1 = 'python gen-fp.py -i ../data/result-ML-CH-npt.txt -e morgan1,simple -o out-ch-density'
cmd2 = 'python split-data.py -i ../data/result-ML-CH-npt.txt -o out-ch-density'
cmd311 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-1.txt -o out-ch-density/11'
cmd312 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-1.txt -o out-ch-density/12'
cmd313 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-1.txt -o out-ch-density/13'
cmd314 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-1.txt -o out-ch-density/14'
cmd321 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-2.txt -o out-ch-density/21'
cmd322 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-2.txt -o out-ch-density/22'
cmd323 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-2.txt -o out-ch-density/23'
cmd324 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-2.txt -o out-ch-density/24'
cmd331 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-3.txt -o out-ch-density/31'
cmd332 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-3.txt -o out-ch-density/32'
cmd333 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-3.txt -o out-ch-density/33'
cmd334 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-3.txt -o out-ch-density/34'
cmd341 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-4.txt -o out-ch-density/41'
cmd342 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-4.txt -o out-ch-density/42'
cmd343 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-4.txt -o out-ch-density/43'
cmd344 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-4.txt -o out-ch-density/44'
cmd351 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-5.txt -o out-ch-density/51'
cmd352 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-5.txt -o out-ch-density/52'
cmd353 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-5.txt -o out-ch-density/53'
cmd354 = 'python train.py -i ../data/result-ML-CH-npt.txt -t density -f out-ch-density/fp_morgan1,out-ch-density/fp_simple -p out-ch-density/part-5.txt -o out-ch-density/54'
# Simu einter for CH
cmd1 = 'python gen-fp.py -i ../data/result-ML-CH-npt.txt -e morgan1,simple -o out-ch-einter'
cmd2 = 'python split-data.py -i ../data/result-ML-CH-npt.txt -o out-ch-einter'
cmd311 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-1.txt -o out-ch-einter/11'
cmd312 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-1.txt -o out-ch-einter/12'
cmd313 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-1.txt -o out-ch-einter/13'
cmd314 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-1.txt -o out-ch-einter/14'
cmd321 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-2.txt -o out-ch-einter/21'
cmd322 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-2.txt -o out-ch-einter/22'
cmd323 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-2.txt -o out-ch-einter/23'
cmd324 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-2.txt -o out-ch-einter/24'
cmd331 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-3.txt -o out-ch-einter/31'
cmd332 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-3.txt -o out-ch-einter/32'
cmd333 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-3.txt -o out-ch-einter/33'
cmd334 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-3.txt -o out-ch-einter/34'
cmd341 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-4.txt -o out-ch-einter/41'
cmd342 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-4.txt -o out-ch-einter/42'
cmd343 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-4.txt -o out-ch-einter/43'
cmd344 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-4.txt -o out-ch-einter/44'
cmd351 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-5.txt -o out-ch-einter/51'
cmd352 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-5.txt -o out-ch-einter/52'
cmd353 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-5.txt -o out-ch-einter/53'
cmd354 = 'python train.py -i ../data/result-ML-CH-npt.txt -t einter -f out-ch-einter/fp_morgan1,out-ch-einter/fp_simple -p out-ch-einter/part-5.txt -o out-ch-einter/54'
# Simu Cp for CH
cmd1 = 'python gen-fp.py -i ../data/result-ML-CH-cp.txt -e morgan1,simple -o out-ch-cp'
cmd2 = 'python split-data.py -i ../data/result-ML-CH-cp.txt -o out-ch-cp'
cmd311 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-1.txt -o out-ch-cp/11'
cmd312 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-1.txt -o out-ch-cp/12'
cmd313 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-1.txt -o out-ch-cp/13'
cmd314 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-1.txt -o out-ch-cp/14'
cmd321 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-2.txt -o out-ch-cp/21'
cmd322 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-2.txt -o out-ch-cp/22'
cmd323 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-2.txt -o out-ch-cp/23'
cmd324 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-2.txt -o out-ch-cp/24'
cmd331 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-3.txt -o out-ch-cp/31'
cmd332 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-3.txt -o out-ch-cp/32'
cmd333 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-3.txt -o out-ch-cp/33'
cmd334 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-3.txt -o out-ch-cp/34'
cmd341 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-4.txt -o out-ch-cp/41'
cmd342 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-4.txt -o out-ch-cp/42'
cmd343 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-4.txt -o out-ch-cp/43'
cmd344 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-4.txt -o out-ch-cp/44'
cmd351 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-5.txt -o out-ch-cp/51'
cmd352 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-5.txt -o out-ch-cp/52'
cmd353 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-5.txt -o out-ch-cp/53'
cmd354 = 'python train.py -i ../data/result-ML-CH-cp.txt -t cp -f out-ch-cp/fp_morgan1,out-ch-cp/fp_simple -p out-ch-cp/part-5.txt -o out-ch-cp/54'

# Simu density for All
cmd1 = 'python gen-fp.py -i ../data/result-ML-All-npt.txt -e morgan1,simple -o out-all-density'
cmd2 = 'python split-data.py -i ../data/result-ML-All-npt.txt -o out-all-density'
cmd311 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-density/11'
cmd312 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-density/12'
cmd313 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-density/13'
cmd314 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-1.txt -o out-all-density/14'
cmd321 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-2.txt -o out-all-density/21'
cmd322 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-2.txt -o out-all-density/22'
cmd323 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-2.txt -o out-all-density/23'
cmd324 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-2.txt -o out-all-density/24'
cmd331 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-3.txt -o out-all-density/31'
cmd332 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-3.txt -o out-all-density/32'
cmd333 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-3.txt -o out-all-density/33'
cmd334 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-3.txt -o out-all-density/34'
cmd341 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-4.txt -o out-all-density/41'
cmd342 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-4.txt -o out-all-density/42'
cmd343 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-4.txt -o out-all-density/43'
cmd344 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-4.txt -o out-all-density/44'
cmd351 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-5.txt -o out-all-density/51'
cmd352 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-5.txt -o out-all-density/52'
cmd353 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-5.txt -o out-all-density/53'
cmd354 = 'python train.py -i ../data/result-ML-All-npt.txt -t density -f out-all-density/fp_morgan1,out-all-density/fp_simple -p out-all-density/part-5.txt -o out-all-density/54'

if __name__ == '__main__':
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd311)
    os.system(cmd312)
    os.system(cmd313)
    os.system(cmd314)
    os.system(cmd321)
    os.system(cmd322)
    os.system(cmd323)
    os.system(cmd324)
    os.system(cmd331)
    os.system(cmd332)
    os.system(cmd333)
    os.system(cmd334)
    os.system(cmd341)
    os.system(cmd342)
    os.system(cmd343)
    os.system(cmd344)
    os.system(cmd351)
    os.system(cmd352)
    os.system(cmd353)
    os.system(cmd354)
