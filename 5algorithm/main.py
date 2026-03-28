import numpy as np
import firedrake as fd
from meshing import get_mesh

def main(params):
    def getparam(param):
        value = getattr(params, param, None)
        if param not in ['d','Lc','theta'] and value is None:
            print(f'WARNING: parameter "{param}" not passed - the script may fail, or produce unexpected results.')
        return value
    
    #%% Problem parameters
    # geometry and mesh
    Lx = getparam('Lx')
    Ly = getparam('Ly')
    Nx = getparam('Nx')
    Ny = getparam('Ny')
    d_init = getparam('d')
    Lc_init = getparam('Lc')
    theta_init = getparam('theta')
    
    provided = [p for p in (d_init, Lc_init, theta_init) if p is not None]
    if len(provided) != 1:
        raise ValueError('Exactly one of d, Lc or theta must be given')
    if provided[0] == 0:
        raise ValueError('The provided parameter (d, Lc, or theta) cannot be zero')
    if d_init:
        d = d_init
        Lc = (d**2 - (Lx/2)**2)**0.5
        theta = np.degrees(np.acos(Lx/2/d))
    elif Lc_init:
        Lc = Lc_init
        theta = np.degrees(np.atan(2*Lc/Lx))
    elif theta_init:
        theta = theta_init
        Lc = Lx/2*np.tan(np.radians(theta))
    Lr = Ly - Lc
    
    # water
    g = getparam('g')
    H0 = getparam('H0')
    rho0 = getparam('rho0')
    
    # buoy
    buoy = getparam('buoy')
    if buoy:
        M = getparam('M')
        alpha = getparam('alpha')
        Lb = Ly - (3*M*np.tan(np.radians(theta))/rho0/np.tan(alpha))**(1/3)
        Z0 = H0 - (3*M*np.tan(np.radians(theta))*np.tan(alpha)**2/rho0)**(1/3)
    else:
        Lb = None
    
    # generator
    if buoy:
        mu0 = 4*np.pi*1e-7
        m = getparam('m')
        a = getparam('a')
        D = getparam('D')
        K = getparam('K')
        L = getparam('L')
        sigma = getparam('sigma')
        nq = getparam('nq')
        VT = getparam('VT')
        Isat = getparam('Isat')
        alphah = getparam('alphah')
        Hm = getparam('Hm')
        gamma = mu0*m*a**2/2/D
        Li = K*mu0*np.pi*a**2*L/D**2
        Ri = 8*a*L/sigma/D**3
        Rc = Ri
        Rl = nq*VT/Isat
        def approxG(Z):
            return ((a**2 + (Z0 + alphah*Hm - L/2 - Z)**2)**-1.5
                    - (a**2 + (Z0 + alphah*Hm + L/2 - Z)**2)**-1.5)
        GZ0 = approxG(Z0)
        C = getparam('C')
    
    # wavemaker
    wavenumber = getparam('wavenumber')
    omega = np.pi*(g*H0)**0.5/(Ly/wavenumber) # omega = 2pi*f = 2pi*u/wavelength = 2pi*sqrt(gH0)/wavelength = pi*sqrt(gH0)/(Ly/wavenumber)
    A = 0.2*omega*(Lr/Ny)
    smooth = getparam('smooth')
    up = getparam('up_period')
    off = getparam('off_period')
    period = 2*np.pi/omega
    Tw_up = up*period
    Tw_down = (off-up)*period
    Tw_end = off*period
    if smooth:
        def Rdot(t):
            if t < 0: # begin at 0
                ramp = 0
            elif t < Tw_up: # go from 0 to 1
                ramp = t/Tw_up
            elif t < Tw_down: # stay at 1
                ramp = 1
            elif t < Tw_end: # go from 1 to 0
                ramp = (Tw_end - t)/(Tw_end - Tw_down)
            else: # stay at 0
                ramp = 0
            return ramp*A*np.sin(omega*t)
    else:
        def Rdot(t):
            if t < 0:
                ramp = 0
            elif t < Tw_end:
                ramp = 1
            else:
                ramp = 0
            return ramp*A*np.sin(omega*t)
    
    # time
    time_scheme = getparam('time_scheme')
    T = getparam('T')
    T_in_periods = getparam('T_in_periods')
    if T_in_periods: T *= period
    dt = 0.6*2.0/(np.pi*(g*H0)**0.5)/((Nx/Lx)**2 + (Ny/Lr)**2)**0.5
    t = 0
    step = 0
    
    #%% Firedrake set-up
    mesh = get_mesh(Lx, Ly, Nx, Ny, d=d_init, Lc=Lc_init, theta=theta_init, Lb=Lb)
    if buoy:
        x, y = fd.SpatialCoordinate(mesh)
        area = Lx*(Lr+Ly)/2
    
    V_CGN = fd.FunctionSpace(mesh, 'CG', getparam('CGN'))
    v_CGN = fd.TestFunction(V_CGN)
    if buoy:
        V_R0 = fd.FunctionSpace(mesh, 'R', 0)
        V_mixed = V_CGN*V_R0
        v1, v2 = fd.TestFunctions(V_mixed)
    
    # water rest-depth
    if buoy:
        H = fd.Function(V_CGN).interpolate(fd.conditional(y < Lb, H0, Z0 + (Ly-y)*np.tan(alpha)))
    else:
        H = H0
    
    # functions to hold spatial solutions
    phi_old = fd.Function(V_CGN, name='phi')
    if time_scheme == 'SV': phi_mid = fd.Function(V_CGN)
    phi_new = fd.Function(V_CGN, name='phi')
    
    eta_old = fd.Function(V_CGN, name='eta')
    eta_new = fd.Function(V_CGN, name='eta')
    
    if buoy:
        w_old = fd.Function(V_mixed)
        lambda_old, F_old = fd.split(w_old)
        w_new = fd.Function(V_mixed)
        lambda_new, F_new = fd.split(w_new)
    else:
        lambda_old = fd.Function(V_CGN)
        lambda_new = fd.Function(V_CGN)
    
    # python floats to hold ODE solutions
    if buoy:
        W = 0
        Z = 0
        Q = 0
        I = 0
    
    # containters to hold spatially-invariant quantities in forms
    Rcont = fd.Constant(Rdot(t))
    if buoy:
        Wcont = fd.Constant(W)
        Zcont = fd.Constant(Z)
        Icont = fd.Constant(I)
    
    # boundary IDs
    wavemaker = 0
    if buoy:
        left_of_waterline = (0, 1)
        waterline = 2
    
    # solver options
    if buoy:
        solver_parameters = {
            'mat_type': 'nest',
            'ksp_type': 'gmres',
            'ksp_gmres_restart': 100,
            'ksp_gmres_modifiedgramschmidt': True,
            'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'multiplicative',
            'fieldsplit_0': {
                'ksp_type': 'preonly',
                'pc_type': 'lu'
            },
            'fieldsplit_1': {
                'ksp_type': 'preonly',
                'pc_type': 'none'
            }
        }
    
    #%% Equations
    # Symplectic Euler (3.1) p.189 (206) http://www.mat.unimi.it/users/sansotte/pdf_files/hamsys/HaiLubWan-2006.pdf
    if time_scheme == 'SE':
        F_phi = v_CGN*((phi_new-phi_old)/dt + g*eta_old + lambda_old)*fd.dx
        phi_full = fd.NonlinearVariationalSolver(
            fd.NonlinearVariationalProblem(F_phi, phi_new),
            solver_parameters={'pc_type': 'cholesky'}
        )
        
        F_eta = ((v_CGN*(eta_new-eta_old)/dt - H*fd.dot(fd.grad(v_CGN),fd.grad(phi_new)))*fd.dx
                    - H0*v_CGN*Rcont*fd.ds(wavemaker))
        eta_full = fd.NonlinearVariationalSolver(
            fd.NonlinearVariationalProblem(F_eta, eta_new),
            solver_parameters={'pc_type': 'cholesky'}
        )
        
        if buoy:
            F_lambda = (H*fd.dot(fd.grad(v1),fd.grad(lambda_new + g*eta_new))
                        + v1/M*(F_new - gamma*GZ0*Icont))*fd.dx
            F_F = v2*(F_new/area - rho0*lambda_new)*fd.dx
            
            lambda_full = fd.NonlinearVariationalSolver(
                fd.NonlinearVariationalProblem(
                    F_lambda + F_F, w_new,
                    bcs=[
                        fd.DirichletBC(V_mixed.sub(0), 0, left_of_waterline),
                        fd.DirichletBC(V_mixed.sub(0), g*(eta_new - Zcont), waterline)
                    ]
                ),
                solver_parameters=solver_parameters
            )
        
    # Stormer-Verlet (3.4) p.189 (206) http://www.mat.unimi.it/users/sansotte/pdf_files/hamsys/HaiLubWan-2006.pdf
    elif time_scheme == 'SV':
        F_phi_half = v_CGN*((phi_mid-phi_old)/(dt/2) + g*eta_old + lambda_old)*fd.dx
        phi_half = fd.NonlinearVariationalSolver(
            fd.NonlinearVariationalProblem(F_phi_half, phi_mid),
            solver_parameters={'pc_type': 'cholesky'}
        )
        
        F_eta = ((v_CGN*(eta_new-eta_old)/dt - H*fd.dot(fd.grad(v_CGN),fd.grad(phi_mid)))*fd.dx
                    - H0*v_CGN*Rcont*fd.ds(wavemaker))
        eta_full = fd.NonlinearVariationalSolver(
            fd.NonlinearVariationalProblem(F_eta, eta_new),
            solver_parameters={'pc_type': 'cholesky'}
        )
        
        if buoy:
            F_lambda = (H*fd.dot(fd.grad(v1),fd.grad(lambda_new + g*eta_new))
                        + v1/M*(F_new - gamma*GZ0*Icont))*fd.dx
            F_F = v2*(F_new/area - rho0*lambda_new)*fd.dx
            
            lambda_full = fd.NonlinearVariationalSolver(
                fd.NonlinearVariationalProblem(
                    F_lambda + F_F, w_new,
                    bcs=[
                        fd.DirichletBC(V_mixed.sub(0), 0, left_of_waterline),
                        fd.DirichletBC(V_mixed.sub(0), g*(eta_new - Zcont), waterline)
                    ]
                ),
                solver_parameters=solver_parameters
            )
        
        F_phi_full = v_CGN*((phi_new-phi_mid)/(dt/2) + g*eta_new + lambda_new)*fd.dx
        phi_full = fd.NonlinearVariationalSolver(
            fd.NonlinearVariationalProblem(F_phi_full, phi_new),
            solver_parameters={'pc_type': 'cholesky'}
        )
    
    else: raise ValueError(f'Unknown time_scheme identifier {time_scheme}')
    
    if buoy:
        lambda_old, F_old = w_old.subfunctions
        lambda_old.rename('lambda')
        lambda_new, F_new = w_new.subfunctions
        lambda_new.rename('lambda')
    
    #%% Initialise outputs
    save_folder = getparam('save_folder')
    
    if buoy:
        data = [{
            't': t,
            'F': float(F_new),
            'W': W,
            'Z': Z,
            'Q': Q,
            'I': I,
            'Ew': 0,
            'Eb': 0,
            'Ei': 0,
            'E': 0,
            'Pg': 0,
            'Pl': 0,
        }]
    else:
        data = [{'t': t, 'E': 0}]
    
    gif_save = getparam('gif_save')
    if gif_save:
        n_yvals = getparam('n_yvals')
        yvals = [i * Ly / (n_yvals - 1) for i in range(n_yvals)]
        points = [[Lx/2, yval] for yval in yvals]
        pointEvaluator = fd.PointEvaluator(mesh, points)
        gif_data = {
            'yvals': yvals,
            'eta': [{'t': t, 'values': [0]*n_yvals}]
        }
        if buoy:
            gif_data['lambda'] = [{'t': t, 'values': [0]*n_yvals}]
    
    pvd_save = getparam('pvd_save')
    if pvd_save:
        pvdfile = fd.output.VTKFile(save_folder+'/pvd/pvd.pvd')
        if buoy:
            pvdfile.write(phi_new, eta_new, lambda_new, time=t)
        else:
            pvdfile.write(phi_new, eta_new, time=t)
    else:
        import os
        os.makedirs(save_folder, exist_ok=True)
    
    import matplotlib.pyplot as plt
    if buoy:
        plt.plot([0, Lb, Ly], [H((Lx/2, yval)) for yval in [0, Lb, Ly]])
        plt.title(r'The rest-state water depth $H(y)$ along the centreline')
        plt.xlabel(r'$y$')
        plt.ylabel(r'$H(y)$')
        plt.vlines(Lb, H0, Z0, 'k', 'dashed')
        plt.savefig(save_folder+'/H.png', bbox_inches='tight')
    
    fig, ax = plt.subplots(figsize=(4, 16), subplot_kw=dict(xticks=[], yticks=[]))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fd.triplot(mesh, axes=ax)
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    legend = ax.legend(handles, labels)
    for text in legend.get_texts():
        text.set_rotation(90)
        text.set_verticalalignment('bottom')
    plt.savefig(save_folder+'/mesh.png', bbox_inches='tight')
    from PIL import Image
    Image.open(save_folder+'/mesh.png').rotate(-90, expand=True).save(save_folder+'/mesh.png')
    
    #%% Time-stepping
    while t <= T:
        step += 1
        
        # solve equations
        if time_scheme == 'SE':
            phi_full.solve()
            
            if buoy:
                W = W + dt/M*(float(F_old) - gamma*GZ0*I)
                Wcont.assign(W)
                
                Q = Q + dt*I
            
            t += dt
            Rcont.assign(Rdot(t))
            eta_full.solve()
            
            if buoy:
                Z = Z + dt*W
                Zcont.assign(Z)
                
                I = (Li*I + dt*(gamma*GZ0*W - Q/C - I/2*(Ri+Rc+Rl)))/(Li + dt/2*(Ri+Rc+Rl))
                Icont.assign(I)
                
                lambda_full.solve()
        
        elif time_scheme == 'SV':
            phi_half.solve()
            
            if buoy:
                W = W + dt/(2*M)*(float(F_old) - gamma*GZ0*I)
                Wcont.assign(W)
                
                Q = Q + dt*I/2
            
            t += dt/2
            Rcont.assign(Rdot(t))
            eta_full.solve()
            
            if buoy:
                Z = Z + dt*W
                Zcont.assign(Z)
                
                I = (Li*I + dt*(gamma*GZ0*W - Q/C - I/2*(Ri+Rc+Rl)))/(Li + dt/2*(Ri+Rc+Rl))
                Icont.assign(I)
                
                lambda_full.solve()
            
            t += dt/2
            phi_full.solve()
            
            if buoy:
                W = W + dt/(2*M)*(float(F_new) - gamma*GZ0*I)
                Wcont.assign(W)
                
                Q = Q + dt*I/2
        
        # calculate & store outputs
        Ew = rho0/2*fd.assemble((H*fd.dot(fd.grad(phi_new),fd.grad(phi_new)) + g*eta_new**2)*fd.dx)
        if buoy:
            Eb = M/2*W*W
            Ei = Li/2*I*I + 1/(2*C)*Q*Q
            E = Ew + Eb + Ei
            Pg = I*(I*Rl + Q/C)
            Pl = I*I*(Ri+Rc)
            data.append({
                't': t,
                'F': float(F_new),
                'W': W,
                'Z': Z,
                'Q': Q,
                'I': I,
                'Ew': Ew,
                'Eb': Eb,
                'Ei': Ei,
                'E': E,
                'Pg': Pg,
                'Pl': Pl,
            })
        else:
            data.append({'t': t, 'E': Ew})
        if gif_save and step % gif_save == 0:
            gif_data['eta'].append({
                't': t,
                'values': pointEvaluator.evaluate(eta_new).tolist()
            })
            if buoy:
                gif_data['lambda'].append({
                    't': t,
                    'values': pointEvaluator.evaluate(lambda_new).tolist()
                })
        if pvd_save and step % pvd_save == 0:
            if buoy:
                pvdfile.write(phi_new, eta_new, lambda_new, time=t)
            else:
                pvdfile.write(phi_new, eta_new, time=t)
        
        # set-up next time-step
        phi_old.assign(phi_new)
        eta_old.assign(eta_new)
        if buoy:
            w_old.assign(w_new)
    
    #%% Get spy patterns - need to do this here as matrices only assembled during solves
    import scipy.sparse as sp
    import matspy
    matspy.params.title = False
    matspy.params.indices = False
    matspy.params.shading = 'binary'

    phi_mat = phi_full.snes.ksp.getOperators()[0]
    phi_mat = sp.csr_matrix(phi_mat.getValuesCSR()[::-1], shape=phi_mat.getSize())
    fig, ax = matspy.spy_to_mpl(phi_mat)
    ax.set_axis_off()
    fig.savefig(save_folder+'/phi_sparsity', bbox_inches='tight')
    
    lambda_mat = lambda_full.snes.ksp.getOperators()[0]
    mat00 = lambda_mat.getNestSubMatrix(0,0).convert('dense').getDenseArray()
    mat01 = lambda_mat.getNestSubMatrix(0,1).convert('dense').getDenseArray()
    mat10 = lambda_mat.getNestSubMatrix(1,0).convert('dense').getDenseArray()
    mat11 = lambda_mat.getNestSubMatrix(1,1).convert('dense').getDenseArray()
    lambda_mat = sp.bmat(
        [[sp.csr_matrix(mat00), sp.csr_matrix(mat01)],
         [sp.csr_matrix(mat10), sp.csr_matrix(mat11)]]
    )
    fig, ax = matspy.spy_to_mpl(lambda_mat)
    ax.set_axis_off()
    fig.savefig(save_folder+'/lambda_sparsity', bbox_inches='tight')
    
    #%% Save outputs
    import pandas as pd
    data = pd.DataFrame(data)
    data.to_json(save_folder+'/data.json', orient='records', double_precision=15, indent=2)
    
    if gif_save:
        import json
        with open(save_folder+'/gif.json', 'w') as f:
            json.dump(gif_data, f, indent=2)
    
    if buoy: return np.trapezoid(data['Pg'], data['t'])