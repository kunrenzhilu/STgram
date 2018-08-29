import subprocess
import shlex

if __name__=='__main__':
    args = parser.parse_args()
    assert not args.sem_dim is None
    
    devices = [2,3]
    device_change_point = 15
    
    for sem_dim in enumerate([1, 20, 40, 50, 60, 80, 100]): #7
        time.sleep(60*60) # sleep for 1 hour.
        counter = 0
        for gi, geo_temp in enumerate([1e-2, 1e-1, 1, 1e1, 1e2]): #5
            for ti, time_temp in enumerate([0, 1e-4, 1e-3, 1e-2, 1e-1, 1]): #6
                counter += 1
                geo_dim = 100-sem_dim
                device = devices[counter // device_change_point]
                path = 'log_time_{}'.format(time_temp)
                cmd = 'mkdir {}'.format(path)
                print(cmd)
                subprocess.call(cmd, shell=True)
                
                cmd = 'python train.py --CITY=NYC --LOG_DIR={} --normalize_weight --WITH_TIME --WITH_GPS --WITH_TIMESTAMP --geo_reg_type=xn --device={} --time_temp={} --geo_temp={} --sem_dim={} --geo_dim={}'.format(path, device, time_temp, geo_temp, sem_dim, geo_dim)