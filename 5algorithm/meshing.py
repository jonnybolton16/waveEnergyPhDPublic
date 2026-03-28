import numpy as np
import firedrake as fd

#%% create_grid
def create_grid(Lx, Ly, Nx, Ny, d=None, Lc=None, theta=None, Lb=None):
    '''
    Builds an array containing the coordinates and indexing of the nodes used to
    generate a quadrilateral mesh for a tank with a V-shaped contraction.
    
    Each node is represented by a unique number between 0 and Nn-1, where
    Nn = (Nx+1)*(Ny+1) + (Nx+2)*Nx/2 is the total number of nodes. The total
    number of elements is Nx*Ny + Nx*(Nx+1)/2, and the total number of
    connections is Nx*(Ny+1)+(Nx+1)*Ny + (2*Nx+3)*Nx/2.
    
    Each row of the array corresponds to a unique element. The first 8 columns
    contain the (x,y) coordinates (eg x0,y0,x1,y1,x2,y2,x3,y3) and the last 4
    contain the indices (eg 0,1,2,3) that define the element, given in a
    clockwise order starting from the bottom-left-most node.
    
    An optional argument Lb can be passed, which denotes the position of the
    waterline of a buoy placed in the contraction. If Lb is passed, one of the
    straight lateral lines in the contraction is fixed to lie on the waterline,
    yielding differing left- and right- y-direction grid resolutions; the line
    that minimises the ratio of these resolutions is chosen.
    
    Parameters
    ----------
    Lx : float or int
        The width of the tank.
    Ly : float or int
        The length of the (entire) tank.
    Nx : int, must be even
        The number of elements in the x direction.
    Ny : int
        The number of elements in the y direction (in the rectangular section
        of the tank).
    d : float or int, optional
        The (diagonal) length of the contraction. Exactly one of d, Lc or theta
        must be given. The other two values are calculated.
    Lc : float or int, optional
        The (horizontal) length of the contraction. Exactly one of d, Lc or
        theta must be given. The other two values are calculated.
    theta : float or int, optional
        The angle of the contraction, in degrees. Exactly one of d, Lc or theta
        must be given. The other two values are calculated.
    Lb : float or int, optional
        The position of the waterline of a buoy placed in the contraction. Not
        passing Lb (or passing None) is the same as saying there is no buoy.
    
    Returns
    -------
    mesh_data : numpy.ndarray
        The array storing the mesh data.
    Nn : int
        The total number of nodes.
    Lr : float
        The length of the rectangular part of the tank.
    dycl : float
        The left resolution in the contraction.
    dycr : float
        The right resolution in the contraction.
    '''
    if not all(v is None or v > 0 for v in [Lx, Ly, Nx, Ny, d, Lc, theta, Lb]):
        raise ValueError('All of the inputs must be positive')
    
    if any(v%1 for v in [Nx, Ny]):
        raise ValueError('Nx and Ny must both be integers')
    Nx = int(Nx)
    Ny = int(Ny)
    
    if Nx%2 != 0:
        raise ValueError('Nx must be even')
    
    if [d, Lc, theta].count(None) != 2:
        raise ValueError('Exactly one of d, Lc or theta must be given')
    
    if d:
        if d <= Lx/2:
            raise ValueError('The chosen dimensions are not geometrically' \
                ' possible; the contraction is not closed')
        Lc = (d**2 - (Lx/2)**2)**0.5
        if Ly <= Lc:
            raise ValueError('The chosen dimensions are not geometrically' \
                ' possible; the entire tank is shorter than the contraction')
        theta = np.degrees(np.acos(Lx/2/d))
    elif Lc:
        if Ly <= Lc:
            raise ValueError('The chosen dimensions are not geometrically' \
                ' possible; the entire tank is shorter than the contraction')
        theta = np.degrees(np.atan(2*Lc/Lx))
    elif theta:
        if 90 <= theta:
            raise ValueError('The chosen dimensions are not geometrically' \
                ' possible; the contraction is not closed')
        Lc = Lx/2*np.tan(np.radians(theta))
        if Ly <= Lc:
            raise ValueError('The chosen dimensions are not geometrically' \
                ' possible; the entire tank is shorter than the contraction')
    Lr = Ly - Lc
    
    if Lb:
        if Ly <= Lb:
            raise ValueError('The given waterline is not geometrically' \
            ' possible; it is not in the tank!')
        if Lb <= Lr:
            raise ValueError('The given waterline is not geometrically'
            ' possible; it is in the rectangular part')
        if Nx < 4:
            raise ValueError('Nx must be at-least 4 when a waterline exists')
    
    mesh_data = []
    
    ### Rectangular region
    Nk = Nx*Ny          # number of elements in total
    
    dx = Lx/Nx          # grid resolution in x direction
    dy = Lr/Ny          # grid resolution in y direction
    
    for k in range(Nk):
        col = k//Nx
        row = k%Nx
        
        xB = row*dx
        xT = xB + dx
        
        yL = col*dy
        yR = yL + dy
        
        n1 = col*(Nx+1) + row
        n2 = n1 + 1
        n3 = n2 + Nx+1
        n4 = n3 - 1
        
        mesh_data.append([xB, yL, xT, yL, xT, yR, xB, yR, n1, n2, n3, n4])
    
    ### Contraction region - Nx columns of elements
    Nn = (Nx+1)*Ny # num of first node in region
    
    if Lb: # choose where to place the waterline
        Lcl = Lb - Lr # contraction length to the left of the waterline
        Lcr = Ly - Lb # contraction length to the right of the waterline
        ratios = []   # list of ratios of left to right resolutions
        
        for i in range(Nx//2 - 1): # loop over options
            Nl = 2*(i+1)  # num columns to the left of the waterline
            Nr = Nx - Nl  # num columns to the right of the waterline
            dycl = Lcl/Nl # y grid resolution to the left of the waterline
            dycr = Lcr/Nr # y grid resolution to the right of the waterline
            ratios.append(max(dycl,dycr)/min(dycl,dycr))
        
        choice = ratios.index(min(ratios)) # chosen waterline index
        Nl = 2*(choice+1)
        Nr = Nx - Nl
        Ns = [Nl, Nr]
        dycl = Lcl/Nl
        dycr = Lcr/Nr
        print(f'The ratio of resolutions for the chosen waterline is {ratios[choice]}')

        left = True # True = left of waterline, False = right of waterline
    
    else: # no waterline
        Ns = [Nx]
        dycl = None
        dycr = None
    
    for N in Ns:
        if Lb:
            dyc = dycl if left else dycr
        else:
            dyc = Lc/Nx # grid resolution in y direction
        
        for col in range(N): # loop over columns
            nk = (Nr if Lb and left else 0) + N - col # num elements in column
            nnL = nk + 1 + col%2                      # num nodes on left-hand line
            nnR = nk + (col+1)%2                      # num nodes on right-hand line
            
            yL = (Lb if Lb and not left else Lr) + col*dyc # y pos of LH line
            yR = yL + dyc                                  # y pos of RH line
            
            xL = Lx*(Ly-yL)/Lc # length of LH line
            xLb = (Lx-xL)/2    # x pos of bottom of LH line
            xLt = (Lx+xL)/2    # x pos of top of LH line
            dxl = xL/nk        # grid resolution along LH line
            
            xR = Lx*(Ly-yR)/Lc # length of RH line
            xRb = (Lx-xR)/2    # x pos of bottom of RH line
            xRt = (Lx+xR)/2    # x pos of top of RH line
            if nk != 1:
                dxr = xR/(nk-1)  # grid resolution along RH line
            
            for j in range(nk): # loop over elements, grouped by shape
                if j < nk/2 - 1: # bottom parallelogram
                    x1 = xLb + j*dxl
                    x2 = x1 + dxl
                    x3 = xRb + (j+1)*dxr
                    x4 = x3 - dxr
                    
                    y1 = yL
                    y2 = yL
                    y3 = yR
                    y4 = yR
                    
                    n1 = Nn + j
                    n2 = n1 + 1
                    n3 = n2 + nnL
                    n4 = n3 - 1
                
                elif col%2 == 0 and j == nk/2 - 1: # bottom scalene quad
                    x1 = Lx/2 - dxl
                    x2 = Lx/2
                    x3 = Lx/2
                    x4 = (Lx-dxr)/2
                    
                    y1 = yL
                    y2 = yL
                    y3 = yL + dyc/2
                    y4 = yR
                    
                    n1 = Nn + j
                    n2 = n1 + 1
                    n3 = n2 + nnL
                    n4 = n3 - 1
                
                elif col%2 == 0 and j == nk/2: # top scalene quad
                    x1 = Lx/2
                    x2 = Lx/2 + dxl
                    x3 = (Lx+dxr)/2
                    x4 = Lx/2
                    
                    y1 = yL
                    y2 = yL
                    y3 = yR
                    y4 = yL + dyc/2
                    
                    n1 = Nn + j
                    n2 = n1 + 1
                    n3 = n2 + nnL
                    n4 = n3 - 1
                
                elif j > nk/2: # top parallelogram
                    x1 = xLt - (nk-j)*dxl
                    x2 = x1 + dxl
                    x3 = xRt - (nk-j-1)*dxr
                    x4 = x3 - dxr
                    
                    y1 = yL
                    y2 = yL
                    y3 = yR
                    y4 = yR
                    
                    n1 = Nn + nnL - (nk-j+1)
                    n2 = n1 + 1
                    n3 = n2 + nnR
                    n4 = n3 - 1
                
                else: # kite
                    x1 = (Lx-dxl)/2
                    x2 = Lx/2
                    x3 = (Lx+dxl)/2
                    x4 = Lx/2
                    
                    y1 = yL
                    y2 = yL - dyc/2
                    y3 = yL
                    y4 = yR
                    
                    n1 = Nn + j
                    n2 = n1 + 1
                    n3 = n2 + 1
                    n4 = n3 + nnR
                
                mesh_data.append([x1, y1, x2, y2, x3, y3, x4, y4, n1, n2, n3, n4])
            
            Nn += nnL # num of first node on LH line of next column
        
        left = False
    
    ### Finish
    Nn += 1 # Nn currently gives num of last node; +1 to get total num of nodes
    
    return np.array(mesh_data), Nn, Lr, dycl, dycr

#%% get_mesh
def get_mesh(Lx, Ly, Nx, Ny, d=None, Lc=None, theta=None, Lb=None):
    '''
    Generates a Firedrake MeshGeometry object representing a quadrilateral mesh
    for a tank with a wavemaker and a V-shaped contraction. The wavemaker is on
    the y=0 boundary, and has a boundary ID of 0.
    
    An optional argument Lb can be passed, which denotes the position of the
    waterline of a buoy placed in the contraction. If Lb is passed, all nodes
    to the left of the waterline have an ID of 1 (other than the wavemaker,
    which retains its ID of 0), while the waterline itself has an ID of 2.
    
    Parameters
    ----------
    Lx : float or int
        The width of the tank.
    Ly : float or int
        The length of the (entire) tank.
    Nx : int, must be even
        The number of elements in the x direction.
    Ny : int
        The number of elements in the y direction (in the rectangular section
        of the tank).
    d : float or int, optional
        The (diagonal) length of the contraction. Exactly one of d, Lc or theta
        must be given. The other two values are calculated.
    Lc : float or int, optional
        The (horizontal) length of the contraction. Exactly one of d, Lc or
        theta must be given. The other two values are calculated.
    theta : float or int, optional
        The angle of the contraction, in degrees. Exactly one of d, Lc or theta
        must be given. The other two values are calculated.
    Lb : float or int, optional
        The position of the waterline of a buoy placed in the contraction. Not
        passing Lb (or passing None) is the same as saying there is no buoy.
    
    Returns
    -------
    mesh : firedrake.mesh.MeshGeometry
        The mesh, to be used with Firedrake.
    '''
    
    ### Generate PETSc DMPlex using arrays containing node indices of each cell
    # and coordinates of each node - see 'plex_from_cell_list' at
    # https://www.firedrakeproject.org/_modules/firedrake/mesh.html
    
    # load parameters, and write and read grid array
    grid, Nn, Lr, dycl, dycr = create_grid(Lx, Ly, Nx, Ny, d, Lc, theta, Lb)
    
    # get node indices of each cell
    cells = grid[:, 8:].astype(int) # grid is read as float, so convert to int
    # cells[i, j] returns global index of jth local node of element i
    
    # get coordinates of each node
    xys = grid[:, :8].reshape(-1, 4, 2)
    # xys[i, j] returns [x,y] coords of jth local node of element i
    coords = np.empty((Nn, 2), dtype=float)
    coords[cells] = xys
    # coords[i] returns [x,y] coords of ith global node
    
    plex = fd.mesh.plex_from_cell_list(2, cells, coords, fd.COMM_WORLD)
    
    ### Apply boundary IDs to DMPlex - see for example
    # https://www.firedrakeproject.org/_modules/firedrake/utility_meshes.html#TensorRectangleMesh
    coord_sec = plex.getCoordinateSection() # Section for vecGetClosure()
    coords = plex.getCoordinates()          # Vec for vecGetClosure()
    
    tol = Lr/Ny/2
    ltol = dycl/2
    rtol = dycr/4
    if Lb:
        for face in range(*plex.getDepthStratum(1)):
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            # on wavemaker
            if face_coords[1] < tol and face_coords[3] < tol:
                plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, face, 0)
            # left of waterline
            elif min(face_coords[1], face_coords[3]) < Lb - ltol:
                plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, face, 1)
            # on waterline
            elif Lb - ltol < face_coords[1] < Lb + rtol and Lb - ltol < face_coords[3] < Lb + rtol:
                plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, face, 2)
    else:
        for face in range(*plex.getDepthStratum(1)):
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            # on wavemaker
            if face_coords[1] < tol and face_coords[3] < tol:
                plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, face, 0)
    
    return fd.mesh.Mesh(plex)

#%% get_mesh_and_submesh
def get_mesh_and_submesh(Lx, Ly, Nx, Ny, d=None, Lc=None, theta=None, Lb=None):
    '''
    Generates a Firedrake MeshGeometry object representing a quadrilateral mesh
    for a tank with a wavemaker and a V-shaped contraction. The wavemaker is on
    the y=0 boundary, and has a boundary ID of 0.
    
    An optional argument Lb can be passed, which denotes the position of the
    waterline of a buoy placed in the contraction. If Lb is passed, a Submesh
    for the buoy-water interface is generated.
    
    Parameters
    ----------
    Lx : float or int
        The width of the tank.
    Ly : float or int
        The length of the (entire) tank.
    Nx : int, must be even
        The number of elements in the x direction.
    Ny : int
        The number of elements in the y direction (in the rectangular section
        of the tank).
    d : float or int, optional
        The (diagonal) length of the contraction. Exactly one of d, Lc or theta
        must be given. The other two values are calculated.
    Lc : float or int, optional
        The (horizontal) length of the contraction. Exactly one of d, Lc or
        theta must be given. The other two values are calculated.
    theta : float or int, optional
        The angle of the contraction, in degrees. Exactly one of d, Lc or theta
        must be given. The other two values are calculated.
    Lb : float or int, optional
        The position of the waterline of a buoy placed in the contraction. Not
        passing Lb (or passing None) is the same as saying there is no buoy.
    
    Returns
    -------
    supmesh : firedrake.mesh.MeshGeometry
        The mesh, to be used with Firedrake.
    submesh : firedrake.mesh.MeshGeometry
        The submesh, to be used with Firedrake.
    '''
    
    ### Generate PETSc DMPlex using arrays containing node indices of each cell
    # and coordinates of each node - see 'plex_from_cell_list' at
    # https://www.firedrakeproject.org/_modules/firedrake/mesh.html
    
    # load parameters, and write and read grid array
    grid, Nn, Lr, dycl, dycr = create_grid(Lx, Ly, Nx, Ny, d, Lc, theta, Lb)
    
    # get node indices of each cell
    cells = grid[:, 8:].astype(int) # grid is read as float, so convert to int
    # cells[i, j] returns (global) index of jth (local) node of element i
    
    # get coordinates of each node
    xys = grid[:, :8].reshape(-1, 4, 2)
    # xys[i, j] returns [x,y] coords of jth (local) node of element i
    coords = np.empty((Nn, 2), dtype=float)
    coords[cells] = xys
    # coords[i] returns [x,y] coords of ith (global) node
    
    plex = fd.mesh.plex_from_cell_list(2, cells, coords, fd.COMM_WORLD)
    
    ### Apply wavemaker boundary ID to DMPlex - see for example
    # https://www.firedrakeproject.org/_modules/firedrake/utility_meshes.html#TensorRectangleMesh
    coord_sec = plex.getCoordinateSection() # Section for vecGetClosure()
    coords = plex.getCoordinates()          # Vec for vecGetClosure()
    
    tol = Lr/Ny/2                                                           # tolerance for round-off error on wavemaker
    plex.markBoundaryFaces('boundary_faces')                                # set label for edges along boundary
    for face in plex.getStratumIS('boundary_faces', 1).getIndices():        # loop over faces on boundary
        face_coords = plex.vecGetClosure(coord_sec, coords, face)           # x0,y0,x1,y1 of nodes forming face
        if face_coords[1] < tol and face_coords[3] < tol:                   # if face is on y=0 boundary
            plex.setLabelValue(fd.cython.dmcommon.FACE_SETS_LABEL, face, 0) # ID = 0
    
    ### Generate Firedrake MeshGeometry object from DMPlex
    supmesh = fd.mesh.Mesh(plex)
    
    ### Generate Firedrake Submesh if Lb is passed
    if Lb:
        x, y = fd.SpatialCoordinate(supmesh)
        V = fd.FunctionSpace(supmesh, 'DG', 0)
        ind_expr = fd.conditional(y >= Lb, 1, 0)
        ind_func = fd.Function(V).interpolate(ind_expr)
        submesh_id = 999
        supmesh = fd.RelabeledMesh(supmesh, [ind_func], [submesh_id])
        submesh = fd.Submesh(supmesh, 2, submesh_id)
    else:
        submesh = None
    
    return supmesh, submesh