import subprocess
import shlex
import os

if __name__=='__main__':
    device = 0
#     path = 'benchmark'
    path = 'log_K100'
    Ks = '10,50,100'
    cmd = 'mkdir {}'.format(path)
    print(cmd)
    #subprocess.Popen(shlex.split(cmd))
    subprocess.call(cmd, shell=True)
    with open(os.path.join(path, 'readme.txt'), 'w') as f:
        f.write("Evaluation with K=10, 50, 100")
        
    cmd = 'python train.py --CITY=NYC --LOG_DIR={} --normalize_weight --WITH_TIME --WITH_GPS --WITH_TIMESTAMP --geo_reg_type=xn --device={} --Ks={}'.format(path, device, Ks)
    print(cmd)
    subprocess.Popen(shlex.split(cmd))
