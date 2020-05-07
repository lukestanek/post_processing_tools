import numpy as np
import numba
import time
import sys

def calc_rdf(data, nconfigs, nbins, Npart, L):
    
    nbins=nbins*2

    # Bin width
    dr = L/nbins

    # Number density
    n = Npart/L**3
    
    # Intialize histogram counter for each step
    hist_count = np.zeros([nconfigs, nbins])
    
    # Numba this for speedup (~400 X)
    start = time.time()
    sys.stdout.write('Computing g(r).')
    sys.stdout.flush()
    
    @numba.jit
    def hist_loop(hist_count, dr, L):
        
        
        for Nsnap in range(len(hist_count)):

            for i in range(Nsnap*Npart, (Nsnap + 1)*Npart):
                for j in range(i+1, (Nsnap + 1)*Npart):     

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

    avg_hist = np.mean(hist_loop(hist_count, dr, L), axis=0)
     
    r = np.zeros(nbins+1)
    g = np.zeros(nbins+1)
    
    # LUKE : THINK ABOUT WHERE YOU ARE REEPOTING VALUES AT R = 0??
    for i in range(1, nbins+1):
        r[i] = (i - 0.5)*dr
        shell_vol = ((i)*dr)**3 - ((i-1)*dr)**3 
        NFAC = 3/(4 * np.pi * n * Npart * shell_vol)
        g[i] = 2 * avg_hist[i-1] * NFAC

    end = time.time()
    sys.stdout.write(' Finished.\n')
    print('Time Elapsed: {:.2f} seconds.'.format(end-start))

    
    return r[0:int(len(r)/2)], g[0:int(len(g)/2)]

# def plot_gr(r, g, L):
#     plt.plot(r, g)
#     plt.ylim(0, 2)
#     plt.xlim(0, L/2)
