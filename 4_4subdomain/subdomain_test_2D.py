import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plotLabelAndSave(field, label, section):
    plt.figure()
    ax = plt.axes(
        projection='3d', xlabel='$x$', ylabel='$y$', zlabel='$u_{'+label+'}(x,y)$'
    )
    fd.trisurf(field, axes=ax)
    plt.savefig(f'{section}_{nel}_{degree}_{label}')

# solve nabla^2u = 0
# with u(0,y,t) = cos(pi(y+cos(t))), u_x(1,y,t) = 0
# and u_y(x,-1,t) = u_y(x,1,t) = pi*cosh(pi(x-1))/cosh(pi)*sin(pi*cos(t))
# for x >= 0 with (x,y) in [-1, 1] x [-1, 1]

# analytical solution
# u(x,y,t) = 0                                          x <  0
# u(x,y,t) = cosh(pi(x-1))/cosh(pi)*cos(pi(y+cos(t)))   x >= 0

# weak form
# int grad(v).grad(u) dxdy = pi*sin(pi*cos(t)) int cosh(pi(x-1))/cosh(pi)*(v(x,1)-v(x,-1)) dx
# u(x<0,y,t) = 0, u(0,y,t) = cos(pi(y+cos(t)))

def sub2D(nel, degree):
    xcoords = np.linspace(-1, 1, nel+1, dtype=np.double)
    ycoords = np.linspace(-1, 1, nel+1, dtype=np.double)
    coords = np.asarray(
        np.meshgrid(xcoords, ycoords)
    ).swapaxes(0, 2).reshape(-1, 2)
    i, j = np.meshgrid(
        np.arange(nel, dtype=np.int32),
        np.arange(nel, dtype=np.int32)
    )
    cells = [
        i*(nel+1) + j,
        i*(nel+1) + j+1,
        (i+1)*(nel+1) + j+1,
        (i+1)*(nel+1) + j
    ]
    cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)
    plex = fd.mesh.plex_from_cell_list(2, cells, coords, fd.COMM_WORLD)
    plex.createLabel(fd.cython.dmcommon.FACE_SETS_LABEL)
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    xtol = 0.5 * min(xcoords[1] - xcoords[0], xcoords[-1] - xcoords[-2])
    ytol = 0.5 * min(ycoords[1] - ycoords[0], ycoords[-1] - ycoords[-2])
    x0 = xcoords[nel//2]
    x1 = xcoords[-1]
    y0 = ycoords[0]
    y1 = ycoords[-1]
    for face in range(*plex.getDepthStratum(1)):
        face_coords = plex.vecGetClosure(coord_sec, coords, face)
        if min(face_coords[0], face_coords[2]) < x0 - xtol:
            plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, face, 0)
        else:
            if abs(face_coords[0] - x0) < xtol and abs(face_coords[2] - x0) < xtol:
                plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, face, 1)
            if abs(face_coords[0] - x1) < xtol and abs(face_coords[2] - x1) < xtol:
                plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, face, 2)
            if abs(face_coords[1] - y0) < ytol and abs(face_coords[3] - y0) < ytol:
                plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, face, 3)
            if abs(face_coords[1] - y1) < ytol and abs(face_coords[3] - y1) < ytol:
                plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, face, 4)
    mesh = fd.mesh.Mesh(plex)
    x, y = fd.SpatialCoordinate(mesh)
    
    V = fd.FunctionSpace(mesh, 'CG', degree)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    
    C = fd.Constant(np.cos(0))
    a = fd.dot(fd.grad(v), fd.grad(u)) * fd.dx
    integrand = fd.cosh(np.pi*(x-1))*v
    L = np.pi/np.cosh(np.pi)*fd.sin(np.pi*C)*(integrand*fd.ds(4)-integrand*fd.ds(3))
    
    bcs = [fd.DirichletBC(V, 0, 0), fd.DirichletBC(V, fd.cos(np.pi*(y+C))-1, 1)]

    u_num = fd.Function(V)
    problem = fd.LinearVariationalProblem(a, L, u_num, bcs=bcs)
    solver = fd.LinearVariationalSolver(problem)
    solver.solve()
    
    u_sol = fd.Function(V).interpolate(fd.conditional(
        x<0, 0,
        fd.cosh(np.pi*(x-1))/np.cosh(np.pi)*fd.cos(np.pi*(y+C)) - 1
    ))
    
    section = '2D'
    plotLabelAndSave(u_num, 'num', section)
    plotLabelAndSave(u_sol, 'sol', section)
    plotLabelAndSave(
        fd.Function(V).interpolate(abs(u_sol - u_num)), 'diff', section
    )
    
    fd.triplot(mesh)
    plt.legend(loc=10)
    plt.savefig(f'{section}_{nel}_mesh', bbox_inches='tight')
    
    X = fd.Function(V).interpolate(x).dat.data_ro
    subdom_idx = np.where(X < 0)[0]
    for u in ((u_sol, 'sol'), (u_num, 'num')):
        abs_u = abs(u[0].dat.data_ro[subdom_idx])
        if np.any(abs_u != 0):
            nonzero_idx = np.where(abs_u != 0)[0]
            X_nonzero = np.unique(X[subdom_idx][nonzero_idx])
            print(f'x values where u_{u[1]} != 0: {X_nonzero}')
    
    return fd.errornorm(u_sol, u_num), C, solver, u_num

T = np.linspace(0, 2*np.pi, 25)

def animate(i, ax):
    print(f'Animation step {i+1} of {len(T)}')
    
    global plot, text
    plot.remove()
    text.remove()

    C.assign(np.cos(T[i]))
    solver.solve()
    
    plot = fd.trisurf(u_num, axes=ax)
    text = ax.text(-1, 0, -3, f't = {T[i]:.2f}s')

#%%
nel = 50
for degree in (1,3):
    print(f'Doing degree {degree} with nel = {nel}')
    norm, C, solver, u_num = sub2D(nel, degree)
    print(f'Subdomain norm: {norm}')
    fig = plt.figure()
    ax = plt.axes(
        projection='3d', xlabel='$x$', ylabel='$y$', zlabel='$u(x,y,t)$'
    )
    plot = fd.trisurf(u_num, axes=ax)
    text = ax.text(-1, 0, -3, '')
    FuncAnimation(
        fig, animate, frames=len(T), fargs=[ax], init_func=lambda: None
    ).save(f'2D_{nel}_{degree}.gif', writer='pillow')
    print('')