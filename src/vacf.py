import numpy as np
import numba
import time
import sys

def vacf_compute(data, vdot, Npart, Nconfigs, Nlags, lagstep=1, dt = 1):
    
    start = time.time()
    sys.stdout.write('Computing VACF.')
    sys.stdout.flush()  

    vavg = np.zeros(Nconfigs)
    t, V = vacf_loop(data, vdot, Npart, Nconfigs, Nlags, lagstep=1, dt = 1)
    
    vavg = np.sum(V,  axis=0)/(np.ones(Nconfigs)*range(Nconfigs, 0, -1))
    
    end = time.time()
    sys.stdout.write(' Finished.\n')
    print('Time Elapsed: {:.2f} seconds.'.format(end-start))
    
    # Check for equilibration
    if (vavg[1] - vavg[0])/(t[1] - t[0]) > 0.1:
        print("Warning: Slope of Z(0) >= 0.1, consider checking equilibration.")
        
    return t, vavg
    
@numba.jit
def vacf_loop(data, vdot, Npart, Nconfigs, Nlags, lagstep=1, dt = 1):   
    
    # Loop over particles
    for i in range(Npart):

        # Lag loop - exploit stationarity
        for lag in range(Nlags):
            
            start_point = lag*Npart*lagstep

            # Loop over snapshots
            for Nsnap in range(Nconfigs-lag*lagstep):

                vdot[lag, Nsnap] += np.dot(data[start_point + Npart*Nsnap + i, 4:7], data[start_point + i, 4:7])/ np.dot(data[start_point + i, 4:7], data[start_point + i, 4:7])

    # Create t array
    t = np.linspace(0, Nconfigs, Nconfigs)*dt
          
    return t, vdot/Npart