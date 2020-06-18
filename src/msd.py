import numpy as np
import numba
import time
import sys

def msd_compute(data, Nconfigs, dt, Npart, L=1):
    
    '''
    Computes the meean squared displacement for atomistic data in 
    [x y z] format. Exploits particle averaging and stationarity 
    for improved statistics.
    
    Parameters
    ----------
    data : array_like
        Unwrapped positon data in the form [x, y, z].
        
    Nconfigs : int
        Number of configurations in simulation.
        
    Npart : int
        Number of particles.
  
    dt : float
        Timestep.
        
    L : float
        Spatial scale factor.
        
    Returns
    -------
    t : array_like
        Time array with dt = timestep.
        
    msdavg : array_like
        mean-squared displacement averaged over particles and t0.
    '''

    start = time.time()
    sys.stdout.write('Computing MSD.\n')
    sys.stdout.flush()  
    
    # To be changed in future 
    Nlags = Nconfigs
    
    # Allocate
    rmr0 = np.zeros([Nlags, Nconfigs])
    msdavg = np.zeros(Nconfigs)

    # Comppute MSD
    t, M = msd_loop(data, rmr0, Npart, Nconfigs, dt, Nlags, L, lagstep=1)
    
    # Average over lags
    msdavg = np.sum(M,  axis=0)/(np.ones(Nconfigs)*range(Nconfigs, 0, -1))
    
    end = time.time()
    sys.stdout.write(' Finished.\n')
    print('Time Elapsed: {:.2f} seconds.'.format(end-start))
    
    return t, msdavg
            
@numba.jit
def msd_loop(data, rmr0, Npart, Nconfigs, dt, Nlags, L, lagstep=1):   

    # Loop over particles
    for i in range(Npart):
        print(i)

        # Lag loop - exploit stationarity
        for lag in range(Nlags):
            
            start_point = lag*Npart*lagstep

            # Loop over snapshots
            for Nsnap in range(Nconfigs-lag*lagstep):
                
                # compute dislacements
                dx = np.abs(data[start_point + Npart*Nsnap + i, 0]*L - data[start_point + i, 0]*L)
                dy = np.abs(data[start_point + Npart*Nsnap + i, 1]*L - data[start_point + i, 1]*L)
                dz = np.abs(data[start_point + Npart*Nsnap + i, 2]*L - data[start_point + i, 2]*L)
                
                # squared displacement
                rmr0[lag, Nsnap] += dx*dx + dy*dy + dz*dz

    # Create t array
    t = np.linspace(0, Nconfigs, Nconfigs)*dt
    
    return t, rmr0/Npart
