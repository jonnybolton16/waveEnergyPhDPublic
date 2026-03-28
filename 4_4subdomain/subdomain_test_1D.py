import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plotLabelAndSave(field, label, section):
    plt.figure()
    ax = plt.axes(xlabel='$x$', ylabel='$u_{'+label+'}(x)$')
    fd.plot(field, axes=ax)
    if 'sub' in section and 'num' in label:
        for i in range(degree+1):
            ax.axvline(0.4 + np.cos(i*np.pi/degree)/20 + 0.05, c='k', ls='--')
        ax.hlines(0, 0.4, 1, 'k', '--')
    plt.savefig(f'{section}_{nel}_{degree}_{label}', bbox_inches='tight')

# solve u_xx = f with u(0.5) = g and u_x(1) = h
# for x >= 0.5 with x in [0, 1]

# analytical solution
# u(x) = 0                                          x <  0.5
# u(x) = 0.5fx^2 + (h-f)x + g - 0.125f + 0.5(f-h)   x >= 0.5

#%% reduced domain
# weak form for x in [0.5, 1]
# int v_xu_x dx = v(1)h - int vf dx
# u(0.5) = g

def reduced1D(nel, degree):
    mesh = fd.IntervalMesh(nel, 0.5, 1)
    x, = fd.SpatialCoordinate(mesh)
    
    V = fd.FunctionSpace(mesh, 'CG', degree)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    
    a = fd.dot(fd.grad(v), fd.grad(u)) * fd.dx
    L = v*h*fd.ds(2) - v*f*fd.dx
    
    bcs = fd.DirichletBC(V, g, 1)
    
    u_num = fd.Function(V)
    fd.solve(a==L, u_num, bcs=bcs)
    
    u_sol = fd.Function(V).interpolate(
        0.5*f*x**2 + (h-f)*x + g - 0.125*f + 0.5*(f-h)
    )
    
    section = '1D_reduced'
    plotLabelAndSave(u_num, 'num', section)
    plotLabelAndSave(u_sol, 'sol', section)
    plotLabelAndSave(
        fd.Function(V).interpolate(abs(u_sol - u_num)), 'diff', section
    )
    
    return fd.errornorm(u_sol, u_num)

#%% sub-domain
# weak form for x in [0, 1]
# int v_xu_x dx = v(1)h - int vf dx
# u(x<0.5) = 0, u(0.5) = g

def sub1D(nel, degree):
    coords = np.linspace(0, 1, 2*nel+1, dtype=np.double).reshape(-1, 1)
    cells = np.dstack(
        (
            np.arange(0, len(coords) - 1, dtype=np.int32),
            np.arange(1, len(coords), dtype=np.int32),
        )
    ).reshape(-1, 2)
    plex = fd.mesh.plex_from_cell_list(1, cells, coords, fd.COMM_WORLD)
    plex.createLabel(fd.cython.dmcommon.FACE_SETS_LABEL)
    coordinates = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    vStart, vEnd = plex.getDepthStratum(0)
    for v in range(vStart, vEnd):
        vcoord = plex.vecGetClosure(coord_sec, coordinates, v)
        if vcoord[0] == coords[nel]:
            plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, v, 1)
        if vcoord[0] == coords[-1]:
            plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, v, 2)
        if vcoord[0] < coords[nel]:
            plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, v, 0)
    mesh = fd.mesh.Mesh(plex)
    x, = fd.SpatialCoordinate(mesh)
    
    V = fd.FunctionSpace(mesh, 'CG', degree)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    
    a = fd.dot(fd.grad(v), fd.grad(u)) * fd.dx
    L = v*h*fd.ds(2) - v*f*fd.dx
    
    G = fd.Constant(g)
    bcs = [fd.DirichletBC(V, 0, 0), fd.DirichletBC(V, G, 1)]
    
    u_num = fd.Function(V)
    problem = fd.LinearVariationalProblem(a, L, u_num, bcs=bcs)
    solver = fd.LinearVariationalSolver(problem)
    solver.solve()
    
    u_sol = fd.Function(V).interpolate(fd.conditional(
        x<0.5, 0,
        0.5*f*x**2 + (h-f)*x + g - 0.125*f + 0.5*(f-h)
    ))
    
    section = '1D_sub'
    plotLabelAndSave(u_num, 'num', section)
    plotLabelAndSave(u_sol, 'sol', section)
    plotLabelAndSave(
        fd.Function(V).interpolate(abs(u_sol - u_num)), 'diff', section
    )
    
    X = fd.Function(V).interpolate(x).dat.data_ro
    subdom_idx = np.where(X < 0.5)[0]
    for u in ((u_sol, 'sol'), (u_num, 'num')):
        abs_u = abs(u[0].dat.data_ro[subdom_idx])
        if np.any(abs_u != 0):
            nonzero_idx = np.where(abs_u != 0)[0]
            X_nonzero = np.unique(X[subdom_idx][nonzero_idx])
            print(f'x values where u_{u[1]} != 0: {X_nonzero}')
    
    return fd.errornorm(u_sol, u_num), G, solver, u_num

#%% time-dependent
# g -> g*cos(t)

T = np.linspace(0, 2*np.pi, 25)

def animate(i, ax):
    print(f'Animation step {i+1} of {len(T)}')
    
    global plots, text
    for plot in plots:
        plot.remove()
    text.remove()
    
    G.assign(g*np.cos(T[i]))
    solver.solve()
    
    plots = fd.plot(u_num, axes=ax, edgecolor='C0')
    text = ax.text(0.02, 0.95, f't = {T[i]:.2f}s')

#%%
f = 15
g = 1
h = 5
nel = 5
for degree in (1,3):
    print(f'Doing degree {degree} with nel = {nel}')
    print(f'Reduced domain norm: {reduced1D(nel, degree)}')
    norm, G, solver, u_num = sub1D(nel, degree)
    print(f'Subdomain norm: {norm}')
    fig = plt.figure(tight_layout=True)
    ax = plt.axes(xlabel='$x$', ylabel='$u(x,t)$', ylim=[-1.34125, 1.76625])
    for i in range(degree+1):
        ax.axvline(0.4 + np.cos(i*np.pi/degree)/20 + 0.05, c='k', ls='--')
    ax.hlines(0, 0.4, 1, 'k', '--')
    plots = []
    text = ax.text(0.02, 0.95, '')
    FuncAnimation(
        fig, animate, frames=len(T), fargs=[ax], init_func=lambda: None
    ).save(f'1D_time_{nel}_{degree}.gif', writer='pillow')
    print('')