import numpy as np
import numba
import time
import sys

class Spatial:
    ''' This class computes all spatial correlation functions. E.g. radial 
    distribution function static structure factor.
    '''
    
    def __init__(self, data, nconfigs, nbins, L, Nparta, Npartb = 0, method='XYZ'):
        
        self.data = data
        self.nconfigs = nconfigs
        self.nbins = 2*nbins # Not using bins after L/2. 
        self.L = L
        self.Nparta = Nparta 
        self.Npartb = Npartb

        if self.Npartb == 0:
            self.sim = 'single'
        else:
            self.sim =  'binary'

        self.dr = self.L/self.nbins

        # Intialize histogram counter for each step
        self.hist_count = np.zeros([self.nconfigs, self.nbins])

        self.method = method
        
    def rdf(self):
        ''' Wrapper method for numba'd histgram calculating loop'rdf_compute()'.

        Parameters
        ----------

        Returns
        -------
        r : array_like
            Position of optimal historgram locations according to [cite Pohl 
            and Ashcroft].

        g(r) : array_like
            Radial distribution function at location r.

        '''

        if self.sim == 'single':
            
            start = time.time()
            sys.stdout.write('Computing g(r).')
            sys.stdout.flush()
              
            avg_hist = np.mean(self.rdf_comppute(self.hist_count, self.dr, 
                               self.L, self.Nparta, self.data), axis=0)

            r = np.zeros(self.nbins+1)
            g = np.zeros(self.nbins+1)
            
            # Number density
            n = self.Nparta/self.L**3

            for i in range(1, self.nbins+1):
                
                rn = (i + 1e-12)*self.dr
                eps = self.dr/rn
                num = (1 + eps)**4 - 1
                den = (1 + eps)**3 - 1
                bin_locs = 0.75*rn*( num/den )
                r[i] = bin_locs
                
                shell_vol = (i*self.dr)**3 - ((i-1)*self.dr)**3 
                NFAC = 3/(4 * np.pi * n * self.Nparta * shell_vol)
                g[i] = 2 * avg_hist[i-1] * NFAC

            end = time.time()
            sys.stdout.write(' Finished.\n')
            print('Time Elapsed: {:.2f} seconds.'.format(end-start))

        if self.sim == 'binary':
            
            # Atom identifier (e.g. Element name, atomic mass, ID, etc.)
            a, b, = np.unique(self.data[:,0])

            # Atom a info
            ra_inds = np.where(self.data[:,0] == a)
            ra_array = self.data[ra_inds,:]

            # Atom b info
            rb_inds = np.where(self.data[:,0] == b)
            rb_array = self.data[rb_inds,:]
            
            # Intialize histogram counter for each case
            hista_count = np.zeros([self.nconfigs, self.nbins])        
            histb_count = np.zeros([self.nconfigs, self.nbins])  
            histab_count = np.zeros([self.nconfigs, self.nbins])  
                      
            # Numba this for speedup (~400 X)
            start = time.time()
            sys.stdout.write('Computing g(r).')
            sys.stdout.flush()
              
            hista, histb, histab = self.binary_rdf_compute(hista_count, 
                                                        histb_count, 
                                                        histab_count, self.dr, 
                                                        self.L, self.Nparta, 
                                                        self.Npartb, 
                                                        ra_array[0,:,1:], 
                                                        rb_array[0,:,1:])
            
            # Avg over snapples
            avg_hista = np.mean(hista, axis=0) 
            avg_histb = np.mean(histb, axis=0)
            avg_histab = np.mean(histab, axis=0)
            
            # Initialize r array
            r = np.zeros(self.nbins+1)

            # Allocate arrays for rdfs
            gaa = np.zeros(self.nbins+1)
            gbb = np.zeros(self.nbins+1)
            gab = np.zeros(self.nbins+1)


            # Number density
            na = self.Nparta/self.L**3
            nb = self.Npartb/self.L**3
            
            # Mixture number density
            Npartab = self.Nparta + self.Npartb
            nab = Npartab/self.L**3

            for i in range(1, self.nbins+1):
                rn = (i + 1e-12)*self.dr
                eps = self.dr/rn
                num = (1 + eps)**4 - 1
                den = (1 + eps)**3 - 1
                bin_locs = 0.75*rn*( num/den )
                r[i] = bin_locs
                
                shell_vol = (i*self.dr)**3 - ((i-1)*self.dr)**3 
                
                NFACa = 3/(4 * np.pi * na * self.Nparta * shell_vol)
                NFACb = 3/(4 * np.pi * nb * self.Npartb * shell_vol)
                NFACab = 3/(4 * np.pi * nab/2 * Npartab * shell_vol)
                
                gaa[i] = 2 * avg_hista[i-1] * NFACa
                gbb[i] = 2 * avg_histb[i-1] * NFACb
                gab[i] = 2 * avg_histab[i-1] * NFACab

            end = time.time()
            sys.stdout.write(' Finished.\n')
            print('Time Elapsed: {:.2f} seconds.'.format(end-start))
                  
            g = np.zeros([self.nbins+1, 3])
            g[:,0] = gaa
            g[:,1] = gbb
            g[:,2] = gab

        return r[0:int(len(r)/2)], g[0:int(len(g)/2)]

    @staticmethod
    @numba.njit
    def rdf_compute(hist_count, dr, L, Nparta, data):

        Lfac = L # Length factor to convert from crystal coords. 

        for Nsnap in range(len(hist_count)):
            # print(Nsnap)

            for i in range(Nsnap*Nparta, (Nsnap + 1)*Nparta):
                for j in range(i+1, (Nsnap + 1)*Nparta):     

                    # Compute x difference and use minimum image convention
                    x_diff = data[i,0]*Lfac - data[j,0]*Lfac
                    if x_diff > L/2:
                        x_diff -= L
                    elif x_diff < -L/2:
                        x_diff += L

                    # Compute y difference and use minimum image convention
                    y_diff = data[i,1]*Lfac - data[j,1]*Lfac
                    if y_diff > L/2:
                        y_diff -= L
                    elif y_diff < -L/2:
                        y_diff += L

                    # Compute z difference and use minimum image convention
                    z_diff = data[i,2]*Lfac - data[j,2]*Lfac
                    if z_diff > L/2:
                        z_diff -= L
                    elif z_diff < -L/2:
                        z_diff += L

                    # Calculate total distance between particles i,j
                    dist = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                    # Add to histogram
                    hist_index = int(dist/dr) # get index (starting at 0)

                    hist_count[Nsnap, hist_index] += 1

        return hist_count

    @staticmethod
    @numba.njit
    def binary_rdf_compute(hista_count, histb_count, histab_count, dr, L, 
                           Nparta, Npartb, ra_array, rb_array):

        for Nsnap in range(len(hista_count)):

            # Species a
            for i in range(Nsnap*Nparta, (Nsnap + 1)*Nparta):
                for j in range(i+1, (Nsnap + 1)*Nparta):     

                    # Compute x difference and use minimum image convention
                    x_diff = ra_array[i,0]*1 - ra_array[j,0]*1
                    if x_diff > L/2:
                        x_diff -= L
                    elif x_diff < -L/2:
                        x_diff += L

                    # Compute y difference and use minimum image convention
                    y_diff = ra_array[i,1]*1 - ra_array[j,1]*1
                    if y_diff > L/2:
                        y_diff -= L
                    elif y_diff < -L/2:
                        y_diff += L

                    # Compute z difference and use minimum image convention
                    z_diff = ra_array[i,2]*1 - ra_array[j,2]*1
                    if z_diff > L/2:
                        z_diff -= L
                    elif z_diff < -L/2:
                        z_diff += L

                    # Calculate total distance between particles i,j
                    dist = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                    # Add to histogram
                    hist_index = int(dist/dr) # get index (starting at 0)
            
                    hista_count[Nsnap, hist_index] += 1
                
            # Species b
            for i in range(Nsnap*Npartb, (Nsnap + 1)*Npartb):
                for j in range(i+1, (Nsnap + 1)*Npartb):     

                    # Compute x difference and use minimum image convention
                    x_diff = rb_array[i,0]*1 - rb_array[j,0]*1
                    if x_diff > L/2:
                        x_diff -= L
                    elif x_diff < -L/2:
                        x_diff += L

                    # Compute y difference and use minimum image convention
                    y_diff = rb_array[i,1]*1 - rb_array[j,1]*1
                    if y_diff > L/2:
                        y_diff -= L
                    elif y_diff < -L/2:
                        y_diff += L

                    # Compute z difference and use minimum image convention
                    z_diff = rb_array[i,2]*1 - rb_array[j,2]*1
                    if z_diff > L/2:
                        z_diff -= L
                    elif z_diff < -L/2:
                        z_diff += L

                    # Calculate total distance between particles i,j
                    dist = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                    # Add to histogram
                    hist_index = int(dist/dr) # get index (starting at 0)
            
                    histb_count[Nsnap, hist_index] += 1
                
            # Mixture
            for i in range(Nsnap*Nparta, (Nsnap + 1)*Nparta):
                for j in range(Nsnap*Npartb, (Nsnap + 1)*Npartb):     

                    # Compute x difference and use minimum image convention
                    x_diff = ra_array[i,0]*1 - rb_array[j,0]*1
                    if x_diff > L/2:
                        x_diff -= L
                    elif x_diff < -L/2:
                        x_diff += L

                    # Compute y difference and use minimum image convention
                    y_diff = ra_array[i,1]*1 - rb_array[j,1]*1
                    if y_diff > L/2:
                        y_diff -= L
                    elif y_diff < -L/2:
                        y_diff += L

                    # Compute z difference and use minimum image convention
                    z_diff = ra_array[i,2]*1 - rb_array[j,2]*1
                    if z_diff > L/2:
                        z_diff -= L
                    elif z_diff < -L/2:
                        z_diff += L

                    # Calculate total distance between particles i,j
                    dist = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                    # Add to histogram
                    hist_index = int(dist/dr) # get index (starting at 0)
            
                    histab_count[Nsnap, hist_index] += 1

        return hista_count, histb_count, histab_count


    def ssk(self, L_array, k_array):
        ''' Compute the static structure factor for atomistic simulation data.

        Parameters
        ----------
        L_array : array_like
            The length of simulation cell in each direction.

        k_array : array_like
            The components of the wave vector in each direction.

        Returns
        -------
        k : array_like
            The wave number.

        self.k_counts : array_like
            Number of points in each histogram to contribute to avg value

        S : array_like
            Static strucure factor at location k.

        '''

        # Unpack L and k values in each direction
        self.Lx, self.Ly, self.Lz = L_array
        self.kx, self.ky, self.kz = k_array
        
        self.k_arr, self.S, self.k_counts, k_unique  = self.kspace_setup()
        k = k_unique * 2*np.pi/self.Lx # Take L to be in the x direction

        if self.sim == "single":
            start = time.time()
            S = self.sk_compute(self.S, self.k_arr, self.Lx, self.Ly, self.Lz, 
                                self.nconfigs, self.Nparta, self.data, 
                                self.k_counts, self.method)
            end = time.time()

            print("Elapsed Time: ", end-start, " seconds")

            Sout = np.zeros([len(k)])
            Sout = S.real

        if self.sim == "binary":

            a, b, = np.unique(self.data[:,0])

            ra_inds = np.where(self.data[:,0] == a)
            ra_array = self.data[ra_inds,:]
            rb_inds = np.where(self.data[:,0] == b)
            rb_array = self.data[rb_inds,:]
            
        
            start = time.time()
            Sii, Sjj, Sij = self.binary_sk_compute(self.S, self.k_arr, self.Lx, self.Ly, self.Lz, self.nconfigs, 
                                             self.Nparta, self.Npartb, ra_array[0,:,1:], 
                                             rb_array[0,:,1:], self.k_counts, self.method)
            end = time.time()

            print("Elapsed Time: ", end-start, " seconds")

            Sout = np.zeros([len(k), 3])
            Sout[:,0] = Sii.real
            Sout[:,1] = Sjj.real
            Sout[:,2] = Sij.real
            
        return k, self.k_counts, Sout

    def kspace_setup(self):
        ''' Sets up the k-space for computing the static structure factor by
        computing all possible kpoints.
        '''

        # Initialize wave number integer array
        kx_arr = np.arange(0, self.kx, 1)
        ky_arr = np.arange(0, self.ky, 1)
        kz_arr = np.arange(0, self.kz, 1)

        # Obtain all possible permutations of the wave number arrays
        k_arr = [np.array([i, j, k]) for i in kx_arr  
                                     for j in ky_arr 
                                     for k in kz_arr] 

        # Compute wave number magnitude - dont use |k| = 0 (skipping first entry
        # in k_arr)
        k_mag = np.sqrt(np.sum(np.array(k_arr)**2, axis=1)[..., None])

        # Add magnitude to wave number array
        k_arr = np.concatenate((k_arr, k_mag), 1)
        
        # Sort from lowest to highest magnitude
        ind = np.argsort(k_arr[:,-1])
        k_arr = k_arr[ind]

        # Count how many times a |k| value appears
        k_unique, k_counts = np.unique(k_arr[1:,-1], return_counts=True)

        # Generate a 1D array containing index to be used in S array
        k_index = np.repeat(range(len(k_counts)), k_counts)[..., None]

        # Add index to k_array
        k_arr = np.concatenate((k_arr[1:,:], k_index), 1)
        
        # Set-up S array
        S = np.zeros(len(k_unique), dtype=complex)

        return k_arr, S, k_counts, k_unique

    @staticmethod
    @numba.njit 
    def sk_compute(S, k_arr, Lx, Ly, Lz, Nconfigs, Npart, r_array, 
                   k_counts, method):
        ''' Compute the static structure factor.

        Returns
        -------
        S : array_like
            Normalized static structure factor for each k-point.

        '''
        if method=='VASP': # convert from crystal coords.
            Lfacx = Lx
            Lfacy = Ly
            Lfacz = Lz
        else: 
            Lfacx = 1
            Lfacy = 1
            Lfacz = 1
        
        # Loop over snapshots
        for Nsnap in range(Nconfigs):

            if Nsnap%1000 ==0:
                print(np.floor(Nsnap/Nconfigs * 100), "% complete")

            # fix k-value
            for k in range(len(k_arr)):
                
                # Set accumulator to  for each k-value in a snapshot
                S1 = 0      
                
                # Loop over particles
                for i in range(Nsnap*Npart, (Nsnap + 1)*Npart):

                    xa = r_array[i,0]*Lfacx # x value
                    ya = r_array[i,1]*Lfacy # y value
                    za = r_array[i,2]*Lfacz # z value
      
                    kdotra = (k_arr[k,0]/Lx * xa + k_arr[k,1]/Ly * ya +  
                              k_arr[k,2]/Lz * za) * 2*np.pi
           
                    S1 += np.exp(-1j * kdotra)
                S2 = np.conj(S1)
                
                # Add S(k) to appropriate k location
                S[int(k_arr[k,-1])] += S1 * S2

        return S/(Npart * Nconfigs * k_counts)

    @staticmethod
    @numba.njit 
    def binary_sk_compute(S, k_arr, Lx, Ly, Lz, Nconfigs, Nparta, Npartb, 
                         ra_array, rb_array, k_counts, method):
        
        Npartab = Nparta + Npartb
        
        Saa = np.zeros_like(S)
        Sbb = np.zeros_like(S)
        Sab = S
        
        if method=='VASP':
            Lfacx = Lx
            Lfacy = Ly
            Lfacz = Lz
        else:
            Lfacx = 1
            Lfacy = 1
            Lfacz = 1
        
        # Loop over snapshots
        for Nsnap in range(Nconfigs):

            if Nsnap%100 ==0:
                print(Nsnap)

            # fix k
            for k in range(len(k_arr)):
                
                # Set accumulator to  for each k-value in a snapshot
                Saa1 = 0      
                Sbb1 = 0
                
                # Loop over particle type a
                for i in range(Nsnap*Nparta, (Nsnap + 1)*Nparta):

                    xa = ra_array[i,0]*Lfacx # x value
                    ya = ra_array[i,1]*Lfacy # y value
                    za = ra_array[i,2]*Lfacz # z value
      
                    kdotra = (k_arr[k,0]/Lx * xa + k_arr[k,1]/Ly * ya + \
                              k_arr[k,2]/Lz * za) * 2*np.pi
                  
                    Saa1 += np.exp(-1j * kdotra)
                Saa2 = np.conj(Saa1)
            
                # Loop over particle type b
                for i in range(Nsnap*Npartb, (Nsnap + 1)*Npartb):

                    xb = rb_array[i,0]*Lfacz # x value
                    yb = rb_array[i,1]*Lfacy # y value
                    zb = rb_array[i,2]*Lfacz # z value

                    kdotrb = (k_arr[k,0]/Lx * xb + k_arr[k,1]/Ly * yb + \
                              k_arr[k,2]/Lz * zb) * 2*np.pi
                    
                    Sbb1 += np.exp(-1j * kdotrb)
                Sbb2 = np.conj(Sbb1)
                
                # Add S(k) to appropriate k location
                Saa[int(k_arr[k,-1])] += Saa1 * Saa2
                Sbb[int(k_arr[k,-1])] += Sbb1 * Sbb2
                Sab[int(k_arr[k,-1])] += Saa2 * Sbb1
               
        return Saa/(Nparta * Nconfigs * k_counts), \
               Sbb/(Npartb * Nconfigs * k_counts), \
               Sab/(Npartab * Nconfigs * k_counts) 

