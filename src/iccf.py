import numpy as np
import numba

def interdiffusion_current(data, Nconfigs, dt, Nparta, Npartb):
    
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

        
    sim : string
        Simulation type. Can be 'single' or 'binary'    
    
    '''
    
    # Allocate
    jdot = np.zeros([Nlags, Nconfigs])

    # Pick out each species
    a, b, = np.unique(data[:,0])
    ra_inds = np.where(data[:,0] == a)
    ra_array = data[ra_inds,4:7]
    rb_inds = np.where(data[:,0] == b)
    rb_array = data[rb_inds,4:7]
    
    # Compute concentration
    X1 = Nparta/(Nparta + Npartb)
    X2 = Npartb/(Nparta + Npartb)
    
    # Compute ICCF
    ICCF = interdiffusion_current(ra_array[0,:,1:], rb_array[0,:,1:], jdot, Nparta, X1, Npartb, X2, Nconfigs):
        
        
    # Create t array
    t = np.linspace(0, Nconfigs, Nconfigs)*dt
    
    return t, ICCF

@numba.jit
def interdiffusion_current(ra_array, rb_array, jdot, Nparta, X1, Npartb, X2, Nconfigs):
    Nlags = Nconfigs

    # Lag loop - exploit stationarity
    for lag in range(Nlags):
        if lag%10 == 0:
            print(lag)
            
        # Loop over snapshot in simulation
        for Nsnap in range(Nconfigs-lag):
            
            # Start and end points in simulation data for particle type a
            starta = lag*Nparta + Nsnap*Nparta 
            enda = lag*Nparta + (Nsnap + 1)*Nparta
            
            # Start and end points in simulation data for particle type a
            startb = lag*Npartb + Nsnap*Npartb
            endb = lag*Npartb + (Nsnap + 1)*Npartb

            # Compute normanlization factor
            j0 = X2*np.sum(ra_array[0, lag*Nparta:lag*Nparta+ Nparta, :], axis=0) - X1* np.sum(rb_array[0, lag*Npartb:lag*Npartb+Npartb, :], axis=0)
        
            # Compute j(t)
            jt = X2*np.sum(ra_array[0, starta:enda, :], axis=0) - X1*np.sum(rb_array[0, startb:endb, :], axis = 0)

            # Compute normalized correlation function
            jdot[lag, Nsnap] = np.dot(jt, j0)/np.dot(j0, j0)

    return jdot