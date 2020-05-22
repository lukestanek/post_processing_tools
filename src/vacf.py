import numpy as np
import numba
import time
import sys



def vacf_compute(data, Nconfigs, dt, Nparta, Npartb=0, sim='single'):
    
    '''
    Computes the velocity autocorrelation function for artomistic data in 
    [id x y z vx vy vz] format. Exploits particle averaging and stationarity 
    for improved statistics.
    
    Parameters
    ----------
    data : array_like
        Atomistic data with the form [ID, x, y, z, vx, vy, vz].
        
    Nconfigs : int
        Number of configurations in simulation.
        
    Nparta : int
        Number of particles of type a.
        
    Nartb : int
        Number of particles of type b.
        
    dt : float
        Timestep
        
    sim : string
        Simulation type. Can be 'single' or 'binary'
        
    Returns
    -------
    t : array_like
        Time array with dt = timestep.
        
    vavg : array_like
        Averaged (over particles and t0) veelocity autocrrelation function.
    
    D : array_like
        Self-diffusion coefficient computed via Green-Kubo relation in 3D.
        The VACF is integrated up to the corresponding point in time.
    
    '''

    start = time.time()
    sys.stdout.write('Computing VACF.')
    sys.stdout.flush()  
    
    # To be changed in future 
    Nlags = Nconfigs
    

    if sim == 'single':
        
        # Allocate
        vdot = np.zeros([Nlags, Nconfigs])
        vavg = np.zeros(Nconfigs)
        
        # Comppute vacf
        t, V = vacf_loop(data, vdot, Nparta, Nconfigs, dt, Nlags, lagstep=1)

        # Average over lags
        vavg = np.sum(V,  axis=0)/(np.ones(Nconfigs)*range(Nconfigs, 0, -1))
        
    elif sim == 'binary':
        
        # Separate data
        a, b, = np.unique(data[:,0])

        ra_inds = np.where(data[:,0] == a)
        ra_array = data[ra_inds, :]
        rb_inds = np.where(data[:,0] == b)
        rb_array = data[rb_inds, :]
        
        # Allocate
        vdota = np.zeros([Nlags, Nconfigs])
        vdotb = np.zeros([Nlags, Nconfigs])
        vavg = np.zeros([Nconfigs, 2])
        
        # Compute vacf of species a and b
        t, Va, Vb = multi_vacf_loop(ra_array[0,:,:], rb_array[0,:,:], vdota, vdotb, 
                                    Nparta, Npartb, Nconfigs, dt, Nlags, lagstep=1)
        
        # Average over the lags
        vavga = np.sum(Va,  axis=0)/(np.ones(Nconfigs)*range(Nconfigs, 0, -1))
        vavgb = np.sum(Vb,  axis=0)/(np.ones(Nconfigs)*range(Nconfigs, 0, -1))
        
        vavg[:, 0] = vavga
        vavg[:, 1] = vavgb
    
    end = time.time()
    sys.stdout.write(' Finished.\n')
    print('Time Elapsed: {:.2f} seconds.'.format(end-start))
    
#     # Check for equilibration
#     if (vavg[1] - vavg[0])/(t[1] - t[0]) > 0.1:
#         print("Warning: Slope of Z(0) >= 0.1, consider checking equilibration.")
        
    # Compute diffusion coefficiet from Green-Kubo formulation in 3D
    if sim=='single':
        D = np.zeros(len(vavg))
        for step in range(1, len(vavg)):
            D[step] = 1/3 * np.trapz(vavg[0:step], dx=dt)
    elif sim=='binary':
        D = np.zeros([Nconfigs, 2])
        for sp in range(2):
            for step in range(1, Nconfigs):
                D[step, sp] = 1/3 * np.trapz(vavg[0:step, sp], dx=dt)        

    return t, vavg, D
    
@numba.jit
def vacf_loop(data, vdot, Npart, Nconfigs, dt, Nlags, lagstep=1,):   

    # Loop over particles
    for i in range(Npart):
        if i%10 == 0:
            print(i)

        # Lag loop - exploit stationarity
        for lag in range(Nlags):
            
            start_point = lag*Npart*lagstep

            # Loop over snapshots
            for Nsnap in range(Nconfigs-lag*lagstep):
                
                # Compute the autocorrelation function
                vdot[lag, Nsnap] += np.dot(data[start_point + Npart*Nsnap + i, 4:7], 
                                           data[start_point + i, 4:7])

    # Create t array
    t = np.linspace(0, Nconfigs, Nconfigs)*dt
    
            
    return t, vdot/Npart

@numba.jit
def multi_vacf_loop(ra_array, rb_array, vdota, vdotb, Nparta, Npartb, Nconfigs, dt, 
                    Nlags, lagstep=1):   
    
    # Loop over particles a
    print('\nLooping over particles of type a')
    for i in range(Nparta):
        
        if i%10==0:
            print(i)
            
        # Lag loop - exploit stationarity
        for lag in range(Nlags):
            
            start_point = lag*Nparta*lagstep

            # Loop over snapshots
            for Nsnap in range(Nconfigs-lag*lagstep):

                # Compute the autocorrelation function
                vdota[lag, Nsnap] += np.dot(ra_array[start_point + Nparta*Nsnap + i, 4:7], 
                                            ra_array[start_point + i, 4:7])
                
    # Loop over particles b
    print('Looping over particles of type b')
    for i in range(Npartb):
        
        if i%10==0:
            print(i)
            
        # Lag loop - exploit stationarity
        for lag in range(Nlags):
            
            start_point = lag*Npartb*lagstep

            # Loop over snapshots
            for Nsnap in range(Nconfigs-lag*lagstep):

                # Compute the autocorrelation function
                vdotb[lag, Nsnap] += np.dot(rb_array[start_point + Npartb*Nsnap + i, 4:7], 
                                            rb_array[start_point + i, 4:7])
    # Create t array
    t = np.linspace(0, Nconfigs, Nconfigs)*dt
    
    return t, vdota/Nparta, vdotb/Npartb 