import numpy as np
import numba
import time
import sys

def calc_rdf(data, nconfigs, nbins, L, Nparta, Npartb=0, sim='single'):
    
    @numba.jit
    def hist_loop(hist_count, dr, L, Nparta, data):
        
        for Nsnap in range(len(hist_count)):

            for i in range(Nsnap*Nparta, (Nsnap + 1)*Nparta):
                for j in range(i+1, (Nsnap + 1)*Nparta):     

                    # Compute x difference and use minimum image convention
                    x_diff = data[i,0]*1 - data[j,0]*1
                    if x_diff > L/2:
                        x_diff -= L
                    elif x_diff < -L/2:
                        x_diff += L

                    # Compute y difference and use minimum image convention
                    y_diff = data[i,1]*1 - data[j,1]*1
                    if y_diff > L/2:
                        y_diff -= L
                    elif y_diff < -L/2:
                        y_diff += L

                    # Compute z difference and use minimum image convention
                    z_diff = data[i,2]*1 - data[j,2]*1
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


    
    @numba.jit
    def multi_hist_loop(hista_count, histb_count, histab_count, dr, L, Nparta, Npartb, ra_array, rb_array):
        
        
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
                
            # Mixx
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


    # we will not return values after L/2 due to box size
    nbins=nbins*2

    # Bin width
    dr = L/nbins
    
    
    if sim=='single':
        
        # Intialize histogram counter for each step
        hist_count = np.zeros([nconfigs, nbins])
        
        start = time.time()
        sys.stdout.write('Computing g(r).')
        sys.stdout.flush()
          
        avg_hist = np.mean(hist_loop(hist_count, dr, L, Nparta, data[:,1:]), axis=0)

        r = np.zeros(nbins+1)
        g = np.zeros(nbins+1)
        
        # Number density
        n = Nparta/L**3

        for i in range(1, nbins+1):
            
            rn = (i + 1e-12)*dr
            eps = dr/rn
            num = (1 + eps)**4 - 1
            den = (1 + eps)**3 - 1
            bin_locs = 0.75*rn*( num/den )
            r[i] = bin_locs
            
            shell_vol = (i*dr)**3 - ((i-1)*dr)**3 
            NFAC = 3/(4 * np.pi * n * Nparta * shell_vol)
            g[i] = 2 * avg_hist[i-1] * NFAC

        end = time.time()
        sys.stdout.write(' Finished.\n')
        print('Time Elapsed: {:.2f} seconds.'.format(end-start))

          
    if sim == 'mixture':
          
        a, b, = np.unique(data[:,0])
        ra_inds = np.where(data[:,0] == a)
        ra_array = data[ra_inds,:]
        rb_inds = np.where(data[:,0] == b)
        rb_array = data[rb_inds,:]
        
        # Intialize histogram counter for each step
        hista_count = np.zeros([nconfigs, nbins])        
        histb_count = np.zeros([nconfigs, nbins])  
        histab_count = np.zeros([nconfigs, nbins])  
                  
        # Numba this for speedup (~400 X)
        start = time.time()
        sys.stdout.write('Computing g(r).')
        sys.stdout.flush()
          
        hista, histb, histab = multi_hist_loop(hista_count, histb_count, histab_count, dr, L, Nparta, Npartb, ra_array[0,:,1:], rb_array[0,:,1:])
        avg_hista = np.mean(hista, axis=0) 
        avg_histb = np.mean(histb, axis=0)
        avg_histab = np.mean(histab, axis=0)
        
        r = np.zeros(nbins+1)

        gaa = np.zeros(nbins+1)
        gbb = np.zeros(nbins+1)
        gab = np.zeros(nbins+1)
        
        # Number density
        na = Nparta/L**3
        nb = Npartb/L**3
        
        Npartab = Nparta + Npartb
        nab = Npartab/L**3

        for i in range(1, nbins+1):
            rn = (i + 1e-12)*dr
            eps = dr/rn
            num = (1 + eps)**4 - 1
            den = (1 + eps)**3 - 1
            bin_locs = 0.75*rn*( num/den )
            r[i] = bin_locs
            
            shell_vol = (i*dr)**3 - ((i-1)*dr)**3 
            
            NFACa = 3/(4 * np.pi * na * Nparta * shell_vol)
            NFACb = 3/(4 * np.pi * nb * Npartb * shell_vol)
            NFACab = 3/(4 *np.pi * nab/2 * Npartab * shell_vol)
            
            gaa[i] = 2 * avg_hista[i-1] * NFACa
            gbb[i] = 2 * avg_histb[i-1] * NFACb
            gab[i] = 2 * avg_histab[i-1] * NFACab

        end = time.time()
        sys.stdout.write(' Finished.\n')
        print('Time Elapsed: {:.2f} seconds.'.format(end-start))
              
        g = np.zeros([nbins+1, 3])
        g[:,0] = gaa
        g[:,1] = gbb
        g[:,2] = gab

    
    return r[0:int(len(r)/2)], g[0:int(len(g)/2)]
