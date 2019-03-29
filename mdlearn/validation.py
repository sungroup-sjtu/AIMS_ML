
import numpy as np 

def resample(a, nskip, option='min'):
    """ Resample an 1D array. 
        nskip:int;
        option: 'min'/'ave'
    """
    length = (len(a) + nskip - 1)//nskip
    ret = np.zeros(length, dtype=a.dtype)

    assert option in ('min', 'ave')

    if option == 'min':
        for i in range(length):
            ret[i] = np.min(a[i*nskip : min((i+1)*nskip, len(a))])
    elif option == 'ave':
        for i in range(length):
            ret[i] = np.min(a[i*nskip : min((i+1)*nskip, len(a))])
    
    return ret 


def is_converge(x, nskip=50, delay=10, rtol1=1e-2, rtol2=0.2, ltol=0.5, dtol=0.6):
    """ Detect if a sequence of error function is converged.
        In ML, MSE is recommended as x.

    Kwargs:
        nskip: Minimum selection window size;
        delay: Convergence detect window size;
        rtol1: (end - begin) > -rtol1 * average(begin:end)
        rtol2: (end - begin) > -rtol2 * stddev(begin:end)
        ltol: final x - end < ltol * stddev(begin:end)
        dtol: fraction tolerance of continous decreasing;

    Returns:
        convergence: bool, is the sequence converged.
        is_converge_point: bool, is the final point a converge point.
    """
    
    xs = resample(x, nskip, option='min')
    if len(xs) < delay:
        return False, False 

    dtol = int(dtol * delay)
    dx = xs[-1] - xs[-delay]
    lastseg = xs[-delay:]

    def is_continous_decrease():
        i = 0
        while i < delay - dtol - 1:
            j = i
            while lastseg[j+1] - lastseg[j] < 0:
                j += 1
                if j - i == dtol:
                    return True
            i = j + 1
        return False 

    if dx > -rtol1 * np.average(lastseg) and dx > -rtol2 * np.std(lastseg) and not is_continous_decrease():

        if x[-1] - xs[-1] < ltol * np.std(lastseg):
            return True, True 
        else:
            return True, False 

    else:
        return False, False 