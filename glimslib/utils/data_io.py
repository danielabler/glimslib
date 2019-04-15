import os

import numpy as np
import meshio as mio
import SimpleITK as sitk

from glimslib import fenics_local as fenics, config
import glimslib.utils.file_utils as fu
import glimslib.utils.vtk_utils as vtu


# ==============================================================================
# FUNCTIONS FOR IMPORTING 2D MESH DATA
# ==============================================================================

# TODO: Find better conversion between image data and mesh data
# currently: cell image data is converted to point meshdata and vice versa,
# this implies that in each conversion step (image->mesh) the number of cells is reduced by 1 per dimension!!
# check issues/fenics_to_img.py
#
# Maybe use VTK:
# image -> vti; cell data -> point data; point data -> fenics mesh
# fenics mesh -> vti with point data; point data -> cell data; vti -> image

# Replace by create_image_from_fenics_function and create_fenics_function_from_image
#
#
def image2fct2D(image_select):
    """
    This function converts a 2D slice of a 3D SimpleITK image instance into a 2D FEniCS function.
    We use this function to extract a 2D 'labelmap' from a 3D segmentation image.
    :param image_select: slice id in z direction
    :return: instance of fenics.Function()
    """
    image_select_np = sitk.GetArrayFromImage(image_select)
    image_select_np_flat = image_select_np.flatten()
    origin = image_select.GetOrigin()
    spacing= image_select.GetSpacing()
    height = image_select.GetHeight()
    width  = image_select.GetWidth()
    depts  = image_select.GetDepth()

    # construct rectangular mesh with dofs on pixels
    p1 = fenics.Point(origin[0],origin[1])
    p2 = fenics.Point(origin[0]+spacing[0]*width,origin[1]+spacing[1]*height)
    nx = int(width - 1)
    ny = int(height - 1)
    fenics.parameters["reorder_dofs_serial"] = False
    mesh_image = fenics.RectangleMesh(p1,p2,nx,ny)
    n_components = image_select.GetNumberOfComponentsPerPixel()
    if n_components==1:
        V = fenics.FunctionSpace(mesh_image, "CG", 1)
    else:
        V = fenics.VectorFunctionSpace(mesh_image, "CG", 1)
    gdim   = mesh_image.geometry().dim()
    coords = V.tabulate_dof_coordinates().reshape((-1,gdim))
    f_img = fenics.Function(V)
    f_img.vector()[:] = image_select_np_flat
    fenics.parameters["reorder_dofs_serial"] = True
    return f_img

def fct2image2D(function, nx, ny):
    """
    This function interpolates a fenics.Function over a 2D grid with nx x ny elements.
    :param function: fenics.Function
    :param nx: number of elements in x direction
    :param ny: number of elements in y direction
    :return: SimpleITK image
    """
    mesh   = function.function_space().mesh()
    coords = mesh.coordinates()
    x_min  = min(coords[:,0])
    x_max  = max(coords[:,0])
    y_min  = min(coords[:,1])
    y_max  = max(coords[:,1])
    x_lin  = np.linspace(x_min, x_max, nx+1)
    y_lin  = np.linspace(y_min, y_max, ny+1)
    xv, yv = np.meshgrid(x_lin, y_lin)
    array = np.ones((nx, ny))*np.nan
    for i in range(nx):
        for j in range(ny):
            p = fenics.Point(xv[i,j],yv[i,j])
            array[i,j] = function(p)
    # compose image
    img = sitk.GetImageFromArray(array)
    origin = function.function_space().mesh().coordinates().min(axis=0)
    img.SetOrigin(origin)
    spacing_x = abs(x_lin[1] - x_lin[0])
    spacing_y = abs(y_lin[1] - y_lin[0])
    img.SetSpacing((spacing_x, spacing_y))
    return img






def get_measures_from_structured_mesh(fenics_mesh):
    """
    This function computes various measures between the nodes of a mesh:
    - extent
    - spacing
    - size
    It requires the input mesh to be structured meshe.
    """
    dim = fenics_mesh.geometry().dim()
    coords = fenics_mesh.coordinates()
    size   = np.zeros(dim, dtype=int)
    spacing= np.zeros(dim, dtype=float)
    extent = np.zeros( (2,dim), dtype=float)
    for i in range(0,dim):
        unique = np.unique(coords[:,i])
        size[i] = len(unique)
        extent[0, i] = np.min(unique)
        extent[1, i] = np.max(unique)
        spacing[i] = compute_spacing(unique)
    # origin -> ITK defines as origin the edge with smallest coordinate values
    origin = np.min(extent, axis=0)
    return origin, size, spacing, extent, dim

def compute_spacing(number_list):
    diff = np.diff(number_list)
    if np.allclose(diff, diff[0], 1E-4):
        return diff[0]
    else:
        print("Spacing not constant in ", number_list)
        return 0

def get_value_dimension_from_function(fenics_function):
    # get value dimensionality
    signature_string = fenics_function.function_space().element().signature()
    if signature_string.startswith('FiniteElement'):
        vdim = 1
    elif signature_string.startswith('VectorElement'):
        start_pos = signature_string.find('dim=')
        vdim = int(signature_string[start_pos + 4])
    else:
        print("Teach me how to deal with '%s' ;-)")
        vdim=None
    return vdim


def get_measures_from_function(fenics_function):
    mesh = fenics_function.function_space().mesh()
    origin, size, spacing, extent, dim = get_measures_from_structured_mesh(mesh)
    vdim = get_value_dimension_from_function(fenics_function)
    return origin, size, spacing, extent, dim, vdim


def get_measures_from_image(sitk_image):
    origin = sitk_image.GetOrigin()
    spacing = sitk_image.GetSpacing()
    height = sitk_image.GetHeight()
    width = sitk_image.GetWidth()
    depts = sitk_image.GetDepth()
    dim = sitk_image.GetDimension()
    # size
    size = np.zeros(dim, dtype=int)
    size[0] = width
    size[1] = height
    if dim==3:
        size[2]=depts
    # extent
    extent = np.zeros((2, dim), dtype=float)
    for i in range(0, dim):
        extent[0, i] = origin[i]
        extent[1, i] = origin[i] + spacing[i] * (size[i]-1) # n-1 distances between n points
    # value dimensionality
    vdim = sitk_image.GetNumberOfComponentsPerPixel()
    return np.array(origin), size, np.array(spacing), extent, dim, vdim


def create_image_from_fenics_function(fenics_function, size_new=None):
    # When creating a sitk image from np.array, indexing order changes
    # i.e. np[z, y, x] <-> sitk[x, y, z]
    origin, size, spacing, extent, dim, vdim = get_measures_from_function(fenics_function)
    linspaces=[]
    if size_new is None:
        size_new = size
    spacing_new = np.zeros(dim, dtype=float)
    for i in range(0, dim):
        linspace = np.linspace(extent[0,i], extent[1,i], size_new[i])
        linspaces.append(linspace)
        spacing_new[i] = compute_spacing(linspace)
    meshgrid_tuple = np.meshgrid(*linspaces, indexing='ij')
    # populate new data array
    if vdim==1:
        data_array = np.ones(size_new) * np.nan
        is_rgb_data = False
    else: # create
        data_array = np.ones( (*size_new,vdim) ) * np.nan
        is_rgb_data = True
    if dim==2:
        xv, yv = meshgrid_tuple
        #print(xv, yv)
        for i in range(size_new[0]):
            for j in range(size_new[1]):
                #print(i, j)
                #print(xv[i, j], yv[i, j])
                p = fenics.Point(xv[i, j], yv[i, j])
                val = fenics_function(p)
                if vdim==1:
                    data_array[i, j] = val
                else:
                    data_array[i, j, :] = val
        data_array = np.swapaxes(data_array, 0, 1)#swap x, y
    elif dim==3:
        xv, yv, zv = meshgrid_tuple
        for i in range(size_new[0]):
            for j in range(size_new[1]):
                for k in range(size_new[2]):
                    p = fenics.Point(xv[i, j, k], yv[i, j, k], zv[i, j, k])
                    if vdim == 1:
                        data_array[i, j, k] = fenics_function(p)
                    else:
                        data_array[i, j, k, :] = fenics_function(p)
        data_array = np.swapaxes(data_array, 0, 2) #swap x, z
    # compose image
    img = sitk.GetImageFromArray(data_array, isVector=is_rgb_data)
    img.SetOrigin(list(origin))
    img.SetSpacing(list(spacing_new))
    return img


def create_fenics_function_from_image_quick(image):
    origin, size, spacing, extent, dim, vdim = get_measures_from_image(image)
    # fenics expects number of elements as input argument to Rectangle/BoxMesh
    # i.e., n_nodes - 1
    size_new = size - np.ones_like(size, dtype=int)
    # construct rectangular/box mesh with dofs on pixels
    p_min = fenics.Point(extent[0, :])
    p_max = fenics.Point(extent[1, :])
    if dim==2:
        mesh_image = fenics.RectangleMesh(p_min, p_max, *list(size_new))
    elif dim==3:
        mesh_image = fenics.BoxMesh(p_min, p_max, *list(size_new))
    # define value dimensionality
    if vdim == 1:
        fenics.parameters["reorder_dofs_serial"] = False
        V = fenics.FunctionSpace(mesh_image, "CG", 1)
    else:
        fenics.parameters["reorder_dofs_serial"] = True
        V = fenics.VectorFunctionSpace(mesh_image, "CG", 1)
    # get and assign values
    image_np = sitk.GetArrayFromImage(image)
    image_np_flat = image_np.flatten()
    f_img = fenics.Function(V)
    f_img.vector()[:] = image_np_flat
    fenics.parameters["reorder_dofs_serial"] = False
    return f_img


def get_labelfunction_from_image(path_to_file, z_slice=0, data_name='label'):
    """
    Convenience function that (1) extracts 2D slice at z_slice position from 3D SimpleITK image, and
    (2) creates a fenics function from that slice.
    :param path_to_file: path to 3D image file
    :param z_slice: z slice to be extracted
    :return: fenics.Function
    """
    image_label = sitk.ReadImage(path_to_file)
    image_label_select = image_label[:, :, z_slice]
    #image_label_select_np = sitk.GetArrayFromImage(image_label_select)
    f_img_label = image2fct2D(image_label_select)
    f_img_label.rename(data_name, "label")
    return f_img_label







def get_dof_coordinate_map(functionspace):
    mesh = functionspace.mesh()
    dof_coord_map = functionspace.tabulate_dof_coordinates()
    dof_coord_map = dof_coord_map.reshape((-1, mesh.geometry().dim()))
    return dof_coord_map

def get_dofs_by_subspace(functionspace):
    dofs_by_subspace = {}
    for i in range(0,functionspace.num_sub_spaces()):
        dofs_by_subspace[i] = functionspace.sub(i).dofmap().dofs()
    return dofs_by_subspace

def get_dofs_from_coord(dof_coord_map, coord, eps=1E-5):
    dim = dof_coord_map.shape[1]
    condlist = []
    for i in range(0, dim):
        cond= np.logical_and(dof_coord_map[:, i] < coord[i] + eps,
                             dof_coord_map[:, i] > coord[i] - eps)
        condlist.append(cond)

    if dim==2:
        cond_all = np.logical_and(condlist[0], condlist[1])
    elif dim==3:
        cond_all = np.logical_and(condlist[0],
                                  np.logical_and(condlist[1],
                                                 condlist[2]))
    index_tuple = np.where(cond_all)
    if len(index_tuple[0])>0:
        return index_tuple[0]
    else:
        print("Did not find vertex close to (%s) in mesh"%", ".join(map(str,coord)))


def assign_values_to_fenics_function(fenics_function, coord_iterable, value_iterable):
    print("== Start: asign values to fenics function")
    funspace = fenics_function.function_space()
    n_subspaces = funspace.num_sub_spaces()
    if value_iterable.shape[1] != n_subspaces:
        print("Value dimension of value iterable and function do not match!")
    if value_iterable.shape[1] == 1:
        value_iterable = value_iterable.flatten()
    # initialize
    dof_coord_map = get_dof_coordinate_map(funspace)
    dofs_by_subspace = get_dofs_by_subspace(funspace)
    # iterate through coords and values
    coords_not_found = []
    if fenics.is_version("=2018.1.x"):
        for i, coord in enumerate(coord_iterable):
            index_tuple = get_dofs_from_coord(dof_coord_map, coord)
            if i%1000==0:
                print("  - index %i of %i"%(i, len(coord_iterable)))
            if index_tuple is not None:
                if len(index_tuple)>1:
                    for subspace in range(0, n_subspaces):
                        dofs_subspace_set = set(dofs_by_subspace[subspace])
                        for index in index_tuple:
                            if index in dofs_subspace_set:
                                fenics_function.vector()[index] = value_iterable[i, subspace]
                else:
                    idx = index_tuple[0]
                    #print(idx, value_iterable[i])
                    fenics_function.vector()[idx] = value_iterable[i]
            else:
                coords_not_found.append(coord)

    else:
        for i, coord in enumerate(coord_iterable):
            index_tuple = get_dofs_from_coord(dof_coord_map, coord)
            if i % 1000 == 0:
                print("  - index %i of %i" % (i, len(coord_iterable)))
            if index_tuple is not None:
                if len(index_tuple) > 1:
                    for subspace in range(0, n_subspaces):
                        for index in index_tuple:
                            if index in dofs_by_subspace[subspace]:
                                fenics_function.vector()[index] = value_iterable[i, subspace]
                else:
                    idx = index_tuple[0]
                    # print(idx, value_iterable[i])
                    fenics_function.vector()[idx] = value_iterable[i]
            else:
                coords_not_found.append(coord)

    return coords_not_found


def get_coord_value_array_for_image(image, flat=False):
    origin, size, spacing, extent, dim, vdim = get_measures_from_image(image)
    coord_array = np.zeros( (*size, dim))
    value_array = np.zeros( (*size, vdim))
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            index = (i, j)
            coord = image.TransformIndexToPhysicalPoint( index )
            coord_array[i, j, :] = coord
            value = image.GetPixel( index )
            value_array[i, j, :] = value
    if flat:
        if dim==2:
            new_shape_first = size[0]*size[1]
        elif dim==3:
            new_shape_first = size[0] * size[1] * size[2]
        coord_array = coord_array.reshape(new_shape_first, dim)
        value_array = value_array.reshape(new_shape_first, vdim)
    return coord_array, value_array



def create_fenics_function_from_image(image):
    origin, size, spacing, extent, dim, vdim = get_measures_from_image(image)
    # fenics expects number of elements as input argument to Rectangle/BoxMesh
    # i.e., n_nodes - 1
    size_new = size - np.ones_like(size, dtype=int)
    # construct rectangular/box mesh with dofs on pixels
    p_min = fenics.Point(extent[0, :])
    p_max = fenics.Point(extent[1, :])
    if dim == 2:
        mesh_image = fenics.RectangleMesh(p_min, p_max, *list(size_new))
    elif dim == 3:
        mesh_image = fenics.BoxMesh(p_min, p_max, *list(size_new))
    # define value dimensionality
    if vdim == 1:
        V = fenics.FunctionSpace(mesh_image, "CG", 1)
    else:
        V = fenics.VectorFunctionSpace(mesh_image, "CG", 1, dim=2)
    # get and assign values
    f_img = fenics.Function(V)
    coord_array, value_array = get_coord_value_array_for_image(image, flat=True)
    unasigned_coords = assign_values_to_fenics_function(f_img, coord_array, value_array)
    return f_img


# ==============================================================================
# FUNCTIONS FOR IMPORTING 3D MESH DATA
# ==============================================================================

def identify_orphaned_vertices(mesh_in):
    """
    Checks for vertices of a fenics.Mesh that do not belong to any cell.
    Returns list of these orphaned vertices.
    """
    print("Checking for orphaned vertices:")
    vertex_ids = []
    for v in fenics.vertices(mesh_in):
        connected_cells = [c for c in fenics.cells(v)]
        n_cells = len(connected_cells)
        if n_cells <1 :
            print(" - vertex id %i, connected cells %i"%(v.index(), n_cells) )
            vertex_ids.append(v.index())
    print(" Found %i orphaned vertices"%len(vertex_ids))
    return vertex_ids

def remove_orphaned_vertices(mesh_in, vertex_ids):
    """
    Creates new fenics.Mesh from existing mesh and list of orphaned vertices.
    Each vertex in vertex_ids list is removed and cell ids are changed accordingly to account for new number of
    vertices and connectivity.
    This function should be applied to custom meshes generated from CGAL to remove orphaned indices.
    Presence of orphaned indices result in PETSC error code 76.

    WARNING: This function assumes that vertices in vertex_ids list are not connected to any element.
    """
    print("Creating new mesh without vertices %s"%(", ".join(map(str, vertex_ids))))
    n_vertices = mesh_in.coordinates().shape[0]
    mask = np.ones_like(mesh_in.coordinates()).astype(bool)
    for vertex_id in vertex_ids:
        mask[vertex_id, :] = False
    vertices = mesh_in.coordinates()[mask].reshape(((n_vertices - len(vertex_ids), 3)))

    # -- create new connectivity
    vertex_ids.sort(reverse=True)
    connectivity = np.copy(mesh_in.cells())
    for vertex_id in vertex_ids:
        vertex_selection = np.where(mesh_in.cells() > vertex_id)
        connectivity[vertex_selection] = connectivity[vertex_selection] - 1

    #-- create new mesh
    n_points = vertices.shape[0]
    n_cells = connectivity.shape[0]

    editor = fenics.MeshEditor()
    mesh = fenics.Mesh()
    editor.open(mesh, 'tetrahedron', 3, 3)
    editor.init_vertices(n_points)  # number of vertices
    editor.init_cells(n_cells)  # number of cells
    for p_i in range(n_points):
        editor.add_vertex(p_i, vertices[p_i, :].astype(float))
    for c_i in range(n_cells):
        editor.add_cell(c_i, connectivity[c_i, :].astype(np.uintp))
    editor.close()
    return mesh

def convert_meshio_to_fenics_mesh(meshio_mesh, domain_array_name='ElementBlockIds'):
    """
    This function converts a meshio mesh into a Fenics mesh.
    :param meshio_mesh: mesh in meshio format
    :param domain_array_name: name of cell array that indicates subdomains
    :return: fenics.Mesh
    """
    cell_type = list(meshio_mesh.cells.keys())[0]
    if cell_type=='triangle':
        dim=2
        cell_type_fenics = 'triangle'
    elif cell_type=='tetra':
        dim=3
        cell_type_fenics = 'tetrahedron'
    else:
        print("Do not understand cell type '%s'"%cell_type)
    # read cells
    cells  = meshio_mesh.cells[cell_type]
    # read vertices
    points = meshio_mesh.points
    # check if third dimension in vertex array is all identical 0
    # -> this may be the case when reading 2d mesh from vtu
    if dim==2:
        if np.alltrue(points[:,2]==0): # remove third dimension
            points = points[:, :2]
        else:
            print("ERROR: expect third coordinate of all points to be identicial 0 ... not the case")
    n_points = points.shape[0]
    n_cells  = cells.shape[0]

    editor = fenics.MeshEditor()
    mesh = fenics.Mesh()
    editor.open(mesh, cell_type_fenics, dim, dim)
    editor.init_vertices(n_points)  # number of vertices
    editor.init_cells(n_cells)      # number of cells
    for p_i in range(n_points):
        editor.add_vertex(p_i, points[p_i,:].astype(float))
    for c_i in range(n_cells):
        editor.add_cell(c_i, cells[c_i,:].astype(np.uintp))
    editor.close()

    # check connectivity and fix mesh
    orph_vertex_ids_orig = identify_orphaned_vertices(mesh)
    if len(orph_vertex_ids_orig)>0:
        new_mesh = remove_orphaned_vertices(mesh, orph_vertex_ids_orig)
        orph_vertex_ids_new = identify_orphaned_vertices(new_mesh)
    else:
        new_mesh = mesh

    # create subdomains mesh function from 'material' array
    material = meshio_mesh.cell_data[cell_type][domain_array_name]
    subdomains = fenics.MeshFunction("size_t", new_mesh, new_mesh.geometry().dim())
    subdomains.set_all(0)
    subdomains.array()[:] = material.astype(np.uint64)

    return new_mesh, subdomains


def convert_fenics_mesh_to_meshio(fenics_mesh, subdomains=None):
    """
    This function converts a meshio mesh into a Fenics mesh.
    :param meshio_mesh: mesh in meshio format
    :param domain_array_name: name of cell array that indicates subdomains
    :return: fenics.Mesh
    """
    dim = fenics_mesh.geometry().dim()
    mio_points = fenics_mesh.coordinates()
    cells = fenics_mesh.cells()
    if dim==2:
        cell_type = 'triangle'
    elif dim==3:
        cell_type = 'tetrahedron'
    mio_cells = {cell_type : cells}
    mio_mesh = mio.Mesh(mio_points, mio_cells)
    if subdomains is not None:
        data = subdomains.array().astype(np.int)
        mio_cell_data = {'ElementBlockIds' : data }
        mio_mesh.cell_data[cell_type] = mio_cell_data
    return mio_mesh


def remove_mesh_subdomain(fenics_mesh, subdomains, lower_thr, upper_thr, temp_dir=config.output_dir_temp):
    """
    Creates new fenics mesh containing only subdomains lower_thr to upper_thr.
    :return: fenics mesh and subdomains
    """
    path_to_temp_vtk = os.path.join(temp_dir, 'mesh.vtu')
    fu.ensure_dir_exists(path_to_temp_vtk)
    path_to_temp_vtk_thresh = os.path.join(temp_dir, 'mesh_thresh.vtu')
    # 1) convert fenics mesh and subdomains to vtk mesh using meshio
    mio_mesh = convert_fenics_mesh_to_meshio(fenics_mesh, subdomains=subdomains)
    mio.write(path_to_temp_vtk, mio_mesh)
    # 2) load mesh using vtk
    mesh_vtk = vtu.read_vtk_data(path_to_temp_vtk)
    # 3) apply threshold filter to vtk mesh to remove subdomain
    mesh_vtk_thresh = vtu.threshold_vtk_data(mesh_vtk, 'cell', 'ElementBlockIds', lower_thr=lower_thr, upper_thr=upper_thr)
    # 4) save as vtk mesh
    vtu.write_vtk_data(mesh_vtk_thresh, path_to_temp_vtk_thresh)
    # 5) load thresholded mesh using meshio and convert to fenics mesh
    mio_mesh_thresh = mio.read(path_to_temp_vtk_thresh)
    mesh_thresh, subdomains_thresh = convert_meshio_to_fenics_mesh(mio_mesh_thresh)
    return mesh_thresh, subdomains_thresh


# ==============================================================================
# POSTPROCESSING VTU OUTPUT
# ==============================================================================

def merge_vtus_timestep(base_path, timestep, remove=False, reference_file_path=None):
    """
    This function merges data arrays from multiple vtu files into single vtu file.
    We use this function to join simulation outputs, such as 'concentration' and 'displacement' from multiple
    into a single file per time step.
    :param base_path: path to directory where simulation results are stored
    :param timestep: current time step
    :param remove: boolean flag indicating whether original files should be removed
    :param reference_file_path: path to file that includes labelmap
    """
    print("-- Creating joint vtu for timestep %d" % timestep)
    if reference_file_path is None:
        reference_file_path = os.path.join(base_path, "label_map", create_file_name("label_map", 0))
    if os.path.exists(reference_file_path):
        mio_mesh_label = mio.read(reference_file_path)
        names = ['concentration', 'proliferation', 'growth', 'displacement']
        for name in names:
            path_to_vtu = os.path.join(base_path, name, create_file_name(name, timestep))
            if os.path.exists(path_to_vtu):
                mio_mesh = mio.read(path_to_vtu)
                if name in mio_mesh.point_data.keys():
                    point_array = mio_mesh.point_data[name]
                    mio_mesh_label.point_data[name] = point_array
                    if remove:
                        remove_vtu(path_to_vtu)
            else:
                print("   - File '%s' not found"%(path_to_vtu))
        # save joint vtu
        path_to_merged = os.path.join(base_path, 'merged', create_file_name("all", timestep))
        print("   - Saving joint file to '%s'" % (path_to_merged))
        fu.ensure_dir_exists(path_to_merged)
        mio.write(path_to_merged, mio_mesh_label)
    else:
        print("   - Could not find reference file '%s'... skipping"%(reference_file_path))


def create_file_name(name, step):
    file_name = "%s_%05d000000.vtu" % (name, step)
    return file_name

def remove_vtu(path_to_file):
    os.remove(path_to_file)

def merge_VTUs(base_path, delta_t, t_max, remove=False, reference=None):
    """
    This function merges all VTU outputs of a simulation run using `merge_vtus_timestep`.
    """
    for timestep in range(len(np.arange(0, t_max, delta_t)) + 1):
        merge_vtus_timestep(base_path, timestep, remove=remove, reference_file_path=reference)




# ==============================================================================
# FENICS MESH IO for parallell processing
# ==============================================================================

def save_mesh_hdf5(mesh_in, path_to_file, subdomains=None, boundaries=None):
    """
    Saves fenics mesh in hdf5 format.
    See https://fenicsproject.org/qa/5337/importing-marked-mesh-for-parallel-use/
    :param mesh_in: fenics mesh
    :param path_to_file: path to hdf5 file
    :param subdomains: fenics meshfunction dim
    :param boundaries: fenics meshfunction dim-1
    :return:
    """
    hdf = fenics.HDF5File(mesh_in.mpi_comm(), path_to_file, "w")
    hdf.write(mesh_in, "/mesh")
    if subdomains is not None:
        hdf.write(subdomains, "/subdomains")
    if boundaries is not None:
        hdf.write(boundaries, "boundaries")
    hdf.close()

def read_mesh_hdf5(path_to_file):
    """
    Reads mesh from hdf5 file to fenics mesh format
    :param path_to_file: path to file
    :return: mesh, subdomain meshfunction, boundary meshfunction
    """
    # mesh
    mesh = fenics.Mesh()
    hdf = fenics.HDF5File(mesh.mpi_comm(),  path_to_file, "r")
    if fenics.is_version("=2018.1.x") and config.USE_ADJOINT:
        hdf.read(mesh, "/mesh", False, annotate=False)
    else:
        hdf.read(mesh, "/mesh", False)
    # subdomains
    subdomains = fenics.MeshFunction("size_t", mesh, mesh.geometry().dim())
    if hdf.has_dataset('subdomains'):
        if fenics.is_version("=2018.1.x") and config.USE_ADJOINT:
            hdf.read(subdomains, "/subdomains", annotate=False)
        else:
            hdf.read(subdomains, "/subdomains")
    else:
        subdomains.set_all(0)
    # boundaries
    boundaries = fenics.MeshFunction("size_t", mesh, mesh.geometry().dim()-1)
    if hdf.has_dataset('boundaries'):
        if fenics.is_version("=2018.1.x") and config.USE_ADJOINT:
            hdf.read(boundaries, "/boundaries", annotate=False)
        else:
            hdf.read(boundaries, "/boundaries")
    else:
        boundaries.set_all(0)
    hdf.close()
    return mesh, subdomains, boundaries


def save_functions_hdf5(function_dict, path_to_file, time_step=None):
    """
    Saves fenics function in hdf5 format.
    See https://fenicsproject.org/qa/6675/hdf-file-read-write-for-time-series/

    Warning: Functions are saved with continuous numbering, i.e. time_step information only serves for identifying
    file opening mode and write mode, but is not preserved!

    :param function_dict: {function name : function}
    :param path_to_file: path to hdf5 file
    :param time_step: current time_step
    """
    if len(function_dict)>0:
        print("Writing %i functions to '%s'"%(len(function_dict), path_to_file))
        # retrieve one of the functions to get mesh.mpi_comm
        f = list(function_dict.values())[0]
        mpi_comm = f.function_space().mesh().mpi_comm()
        # create hdf file
        # (a) if time step == None or timestep == 0-> assume single write and open as 'w'
        if (time_step == 1) or (time_step is None):
            hdf = fenics.HDF5File(mpi_comm, path_to_file, "w")
        # (b) else, open in append mode
        else:
            hdf = fenics.HDF5File(mpi_comm, path_to_file, "a")
        #
        for name, function in function_dict.items():
            if time_step is None:
                hdf.write(function, name)
            else:
                hdf.write(function, name, time_step)
        hdf.close()
    else:
        print("No functions provided ... cannot write.")


def read_function_hdf5(name, functionspace, path_to_file):
    if os.path.exists(path_to_file):
        f = fenics.Function(functionspace)
        hdf = fenics.HDF5File(functionspace.mesh().mpi_comm(), path_to_file, "r")
        attr = hdf.attributes(name)
        #nsteps = attr['count']
        dataset = name+"/vector_0"
        hdf.read(f, dataset)
        hdf.close()
        return f


def save_function_mesh(function, path_to_hdf5_function, labelfunction=None, subdomains=None):
    if path_to_hdf5_function.endswith('.h5'):
        path_to_hdf5_mesh = path_to_hdf5_function[:-3] + '_mesh.h5'
    else:
        print("Provide path to '.h5' file")
    mesh = function.function_space().mesh()
    fu.ensure_dir_exists(path_to_hdf5_mesh)
    if labelfunction is not None:
        from glimslib.simulation_helpers import SubDomains
        # create subdomains
        subdomains = SubDomains(mesh)
        subdomains.setup_subdomains(label_function=labelfunction)
        # save mesh as hdf5
        save_mesh_hdf5(mesh, path_to_hdf5_mesh, subdomains=subdomains.subdomains)
    elif subdomains is not None:
        save_mesh_hdf5(mesh, path_to_hdf5_mesh, subdomains=subdomains)
    else:
        save_mesh_hdf5(mesh, path_to_hdf5_mesh)
    # save function
    save_functions_hdf5({"function": function}, path_to_hdf5_function, time_step=None)


def load_function_mesh(path_to_hdf5_function, functionspace='function', degree=1):
    if path_to_hdf5_function.endswith('.h5'):
        path_to_hdf5_mesh = path_to_hdf5_function[:-3] + '_mesh.h5'
    else:
        print("Provide path to '.h5' file")
    if os.path.exists(path_to_hdf5_mesh):
        mesh, subdomains, boundaries = read_mesh_hdf5(path_to_hdf5_mesh)
    else:
        print("Could not find mesh file: '%s'" % path_to_hdf5_mesh)
    if os.path.exists(path_to_hdf5_function):
        if functionspace == 'function':
            functionspace = fenics.FunctionSpace(mesh, "Lagrange", degree)
        elif functionspace == 'vector':
            functionspace = fenics.VectorFunctionSpace(mesh, "Lagrange", degree)
        function = read_function_hdf5("function", functionspace, path_to_hdf5_function)
    return function, mesh, subdomains, boundaries


