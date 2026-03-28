import main

class params:
    # geometry and mesh
    Lx = 0.2
    Ly = 2
    Nx = 10
    Ny = 50
    theta = 68.26 # d = 0.27 # Lc = 0.25 # only pass one of these
    
    # water
    g = 9.81
    H0 = 0.1
    rho0 = 997
    
    # buoy
    buoy = True # if False, no buoy is modeled
    M = 0.1
    alpha = 0.6418818298648808
    # calculated using atan(3*M*tan(radians(theta))/rho0/(Ly-Lb)**3)
    # with Lb = 1.8996831214658347 (from mesh coordinates)
    
    # generator
    m = 0.1
    a = 0.04
    D = 0.2769e-3
    K = 0.53
    L = 0.08
    sigma = 5.96e7
    nq = 1
    VT = 2.05
    Isat = 0.02
    alphah = 0.2
    Hm = 0.2
    C = float('inf') # a capacitance of inf models no capacitor
    
    # wavemaker
    wavenumber = 6
    smooth = False  # if wavemaker start/stop is smooth or not
    up_period = 2   # periods taken to reach max amplitude (when smooth)
    off_period = 10 # number of periods until wavemaker is turned off
    
    # time
    time_scheme = 'SE' # SE or SV
    T = 20
    T_in_periods = True # if True, T *= period
    
    # Lagrange polynomial degree
    CGN = 1
    
    # save parameters
    save_folder = 'outputs'
    gif_save = 10   # save .gif and/or .pvd file every n time-steps
    n_yvals = 101   # number of centerline points to sample for .gif
    pvd_save = None # if None, no file is created

main.main(params)