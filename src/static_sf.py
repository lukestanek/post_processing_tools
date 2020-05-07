import time
import sys
import numpy as np
import numba

def ssk(L, k, data, Nconfigs, Nparta, Npartb = 0, sim="single"):
    
    # Unpack L and k values in each direction
    Lx, Ly, Lz = L
    kx, ky, kz = k
    
    k_arr, S, k_counts, k_unique  = kspace_setup(kx, ky, kz)
    k = k_unique * 2*np.pi/Lx    
    
    if sim == "single":
        start = time.time()
        S = sk_compute(S, k_arr, Lx, Ly, Lz, Nconfigs, Nparta, data[:,1:], k_counts)
        end = time.time()

        print("Elapsed Time: ", end-start, " seconds")
        
    if sim == "mixture":
        
        a, b, = np.unique(data[:,0])

        ra_inds = np.where(data[:,0] == a)
        ra_array = data[ra_inds,:]
        rb_inds = np.where(data[:,0] == b)
        rb_array = data[rb_inds,:]
        
        start = time.time()
        Sii, Sjj, Sij = multi_sk_compute(S, k_arr, Lx, Ly, Lz, Nconfigs, Nparta, 
                                         Npartb, ra_array[0,:,1:], rb_array[0,:,1:], k_counts)
        end = time.time()

        print("Elapsed Time: ", end-start, " seconds")
    
    if sim == "single":
        Sout = np.zeros([len(k)])
        Sout = S.real()
        
    if sim == "mixture":
        Sout = np.zeros([len(k), 3])
        Sout[:,0] = Sii.real()
        Sout[:,1] = Sjj.real()
        Sout[:,2] = Sij.real()
        
    return k, k_counts, Sout


def kspace_setup(kx, ky, kz):

    # Initialize wave number integer array
    kx_arr = np.arange(0, kx, 1)
    ky_arr = np.arange(0, ky, 1)
    kz_arr = np.arange(0, kz, 1)

    # Obtain all possible permutations of the wave number arrays
    k_arr = [np.array([i, j, k]) for i in kx_arr  
                                 for j in ky_arr 
                                 for k in kz_arr] 

    # Compute wave number magnitude - dont use |k| (skipping first entry in k_arr)
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

@numba.jit 
def sk_compute(S, k_arr, Lx, Ly, Lz, Nconfigs, Npart, r_array, k_counts):
    
    # Loop over snapshots
    for Nsnap in range(Nconfigs):

        if Nsnap%100 ==0:
            print(Nsnap)

        # fix k
        for k in range(len(k_arr)):
            
            # Set accumulator to  for each k-value in a snapshot
            S1 = 0      
            
            # Loop over particle type a
            for i in range(Nsnap*Npart, (Nsnap + 1)*Npart):

                xa = r_array[i,0] # x value
                ya = r_array[i,1] # y value
                za = r_array[i,2] # z value
  
                kdotra = (k_arr[k,0]/Lx * xa + k_arr[k,1]/Ly * ya +  k_arr[k,2]/Lz * za) * 2*np.pi
       
                S1 += np.exp(-1j * kdotra)
            S2 = np.conj(S1)
            
            # Add S(k) to appropriate k location
            S[int(k_arr[k,-1])] += S1 * S2
    return S/(Npart * Nconfigs * k_counts)

@numba.jit 
def multi_sk_compute(S, k_arr, Lx, Ly, Lz, Nconfigs, Nparta, Npartb, ra_array, rb_array, k_counts):
    
    Npartab = Nparta + Npartb
    
    Saa = np.zeros_like(S)
    Sbb = np.zeros_like(S)
    Sab = S
    
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

                xa = ra_array[i,0] # x value
                ya = ra_array[i,1] # y value
                za = ra_array[i,2] # z value
  
                kdotra = (k_arr[k,0]/Lx * xa + k_arr[k,1]/Ly * ya +  k_arr[k,2]/Lz * za) * 2*np.pi
              
                Saa1 += np.exp(-1j * kdotra)
            Saa2 = np.conj(Saa1)
        
            # Loop over particle type b
            for i in range(Nsnap*Npartb, (Nsnap + 1)*Npartb):

                xb = rb_array[i,0] # x value
                yb = rb_array[i,1] # y value
                zb = rb_array[i,2] # z value

                kdotrb = (k_arr[k,0]/Lx * xb + k_arr[k,1]/Ly * yb +  k_arr[k,2]/Lz * zb) * 2*np.pi
                
                Sbb1 += np.exp(-1j * kdotrb)
            Sbb2 = np.conj(Sbb1)
            
            # Add S(k) to appropriate k location
            Saa[int(k_arr[k,-1])] += Saa1 * Saa2
            Sbb[int(k_arr[k,-1])] += Sbb1 * Sbb2
            Sab[int(k_arr[k,-1])] += Saa2 * Sbb1
           
    return Saa/(Nparta * Nconfigs * k_counts), Sbb/(Npartb * Nconfigs * k_counts), Sab/(Npartab * Nconfigs * k_counts) 

