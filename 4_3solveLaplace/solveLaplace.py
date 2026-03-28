import numpy as np
import firedrake as fd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import matspy
cos, pi, sinh = fd.cos, fd.pi, fd.sinh

def plotLabelAndSave(field, zlabel, nel, degree):
    plt.figure()
    ax = plt.axes(
        projection='3d', xlabel='$x$', ylabel='$y$', zlabel='$u_{'+zlabel+'}$'
    )
    fd.trisurf(field, axes=ax)
    plt.savefig(
        f'solveLaplace_{nel:02d}_{degree}_u_{zlabel}', bbox_inches='tight'
    )

def solveLaplace(nel, degree):
    mesh = fd.RectangleMesh(nel, nel, W/2, H/2, -W/2, -H/2, quadrilateral=True)
    x, y = fd.SpatialCoordinate(mesh)
    
    V = fd.FunctionSpace(mesh, 'CG', degree)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    
    a = fd.dot(fd.grad(v), fd.grad(u)) * fd.dx
    L = 0
    
    bcs = [
        fd.DirichletBC(V,  3*cos(pi*y/H),     1 ),
        fd.DirichletBC(V,    cos(pi*y/H),     2 ),
        fd.DirichletBC(V, -2*cos(pi*x/W), (3, 4))
    ]
    
    u_num = fd.Function(V)
    fd.solve(a==L, u_num, bcs=bcs)
    
    u_sol = fd.Function(V).interpolate(
          (3*sinh(pi*(W-2*x)/(2*H)) + sinh(pi*(W+2*x)/(2*H)))/sinh(pi*W/H)*cos(pi*y/H)
        - 2*(sinh(pi*(H-2*y)/(2*W)) + sinh(pi*(H+2*y)/(2*W)))/sinh(pi*H/W)*cos(pi*x/W)
    )
    
    if (nel == n and degree == 1) or (nel == 16*n and degree == 4):
        plotLabelAndSave(u_num, 'num', nel, degree)
        plotLabelAndSave(u_sol, 'sol', nel, degree)
        plotLabelAndSave(
            fd.Function(V).interpolate(abs(u_sol - u_num)), 'diff', nel, degree
        )
    elif nel == 2*n and degree == 1:
        petscMat = fd.assemble(a).petscmat
        scipyMat = sp.csr_matrix(
            petscMat.getValuesCSR()[::-1], shape=petscMat.getSize()
        )
        fig, ax = matspy.spy_to_mpl(
            scipyMat, title=False, indices=False, color_full='black'
        )
        fig.savefig(
            f'solveLaplace_{nel:02d}_{degree}_sparsity', bbox_inches='tight'
        )
    
    return fd.errornorm(u_sol, u_num)

W, H = 1, 2
n = 4

norms = np.array(
    [[solveLaplace(n*2**i, degree) for degree in range(1, 5)] for i in range(5)]
)
orders = np.array([np.log2(norms[i]/norms[i+1]) for i in range(4)])

with np.printoptions(precision=2):
    print(norms)
    print('')
with np.printoptions(precision=4):
    print(orders)