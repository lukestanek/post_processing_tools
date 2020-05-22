import numpy as np
import numba
import time
import sys


def iccf_compute(data, Nconfigs, dt, Nparta, Npartb):
    
    '''
    Computes the interdiffusion current correlation function for artomistic data in [id x y z vx vy vz] format. 
    Exploits particle averaging and stationarity for improved statistics.
    
    Parameters
    ----------
    data : array_like
        Atomistic data with the form [ID, x, y, z, vx, vy, vz].
        
    Nconfigs : int
        Number of configurations in simulation.
                
    dt : float
        Timestep.
        
    Nparta : int
        Number of particles of type a.
        
    Nartb : int
        Number of particles of type b.

    Returns
    -------
    t : array_like
        Time array with dt = timestep.
        
    ICCFavg : array_like
        Averaged (over t0) intrdiffusion current correlation function.
    
    Dij : array_like
        Integgral of the ICCF up to the corresponding point in time.
        Note that this is simply the integral of the autocorrelation function 
        and not the interdiffusion coefficient. To compute the interdiffusion 
        coefficient, multilpy by the necessary prefactor which can be 
        found here: https://link.aps.org/doi/10.1103/PhysRevE.90.023104
        
    '''
    
    # To be changed
    Nlags = Nconfigs
    
    # Allocate
    jdot = np.zeros([Nlags, Nconfigs])

        
    a, b, = np.unique(data[:,0])

    ra_inds = np.where(data[:,0] == a)
    ra_array = data[ra_inds, :]
    rb_inds = np.where(data[:,0] == b)
    rb_array = data[rb_inds, :]
    
    # Compute concentration
    X1 = Nparta/(Nparta + Npartb)
    X2 = Npartb/(Nparta + Npartb)
    
    # Compute ICCF
    ICCF = iccf_loop(ra_array[0,:,:], rb_array[0,:,:], jdot, Nparta, X1, Npartb, X2, Nconfigs)
        
    # Average over lags
    ICCFavg = np.sum(ICCF,  axis=0)/(np.ones(Nconfigs)*range(Nconfigs, 0, -1))
        
    # Create t array
    t = np.linspace(0, Nconfigs, Nconfigs)*dt
    
    # Compute integral of correlation function - still need to multiply by prefactor
    Dij = np.zeros(len(ICCFavg))
    for step in range(1,len(ICCFavg)):
        Dij[step] = np.trapz(ICCFavg[0:step], dx=dt)
        
    return t, ICCFavg, Dij

@numba.jit
def iccf_loop(ra_array, rb_array, jdot, Nparta, X1, Npartb, X2, Nconfigs):
    Nlags = Nconfigs

    # Lag loop - exploit stationarity
    for lag in range(Nlags):
        if lag%(int(Nlags/10)) == 0:         
            print(lag/Nlags * 100,"% Complete.")
            
        # Loop over snapshot in simulation
        for Nsnap in range(Nconfigs-lag):
            
            # Start and end points in simulation data for particle type a
            starta = lag*Nparta + Nsnap*Nparta 
            enda = lag*Nparta + (Nsnap + 1)*Nparta
            
            # Start and end points in simulation data for particle type b
            startb = lag*Npartb + Nsnap*Npartb
            endb = lag*Npartb + (Nsnap + 1)*Npartb

            # Compute j(0)
            j0 = X2*np.sum(ra_array[lag*Nparta:lag*Nparta + Nparta, 4:7], axis=0) - X1* np.sum(rb_array[lag*Npartb:lag*Npartb + Npartb, 4:7], axis=0)
        
            # Compute j(t)
            jt = X2*np.sum(ra_array[starta:enda, 4:7], axis=0) - X1*np.sum(rb_array[startb:endb, 4:7], axis = 0)

            # Compute correlation function
            jdot[lag, Nsnap] = np.dot(jt, j0)

    return jdot