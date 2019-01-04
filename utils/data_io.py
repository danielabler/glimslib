import os

import numpy as np
import meshio as mio
import SimpleITK as sitk

import fenics_local as fenics
import utils.file_utils as fu


# ==============================================================================
# FUNCTIONS FOR IMPORTING 2D MESH DATA
# ==============================================================================

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
    mesh_image = fenics.RectangleMesh(p1,p2,nx,ny)
    fenics.parameters["reorder_dofs_serial"] = False
    V = fenics.FunctionSpace(mesh_image, "CG", 1)
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
    :return: numpy array
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
    return array



def get_labelfunction_from_image(path_to_file, z_slice=0):
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
    f_img_label.rename("label", "label")
    return f_img_label



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
    points = meshio_mesh.points
    cells  = meshio_mesh.cells['tetra']
    n_points = points.shape[0]
    n_cells  = cells.shape[0]

    #fenics.parameters["reorder_dofs_serial"] = False
    # new fenics mesh
    editor = fenics.MeshEditor()
    mesh = fenics.Mesh()
    editor.open(mesh, 'tetrahedron', 3, 3)
    editor.init_vertices(n_points)  # number of vertices
    editor.init_cells(n_cells)      # number of cells
    for p_i in range(n_points):
        editor.add_vertex(p_i, points[p_i,:].astype(float))
    for c_i in range(n_cells):
        editor.add_cell(c_i, cells[c_i,:].astype(np.uintp))
    editor.close()
    #fenics.parameters["reorder_dofs_serial"] = True

    # check connectivity and fix mesh
    orph_vertex_ids_orig = identify_orphaned_vertices(mesh)
    if len(orph_vertex_ids_orig)>0:
        new_mesh = remove_orphaned_vertices(mesh, orph_vertex_ids_orig)
        orph_vertex_ids_new = identify_orphaned_vertices(new_mesh)
    else:
        new_mesh = mesh

    # create subdomains mesh function from 'material' array
    material = meshio_mesh.cell_data['tetra'][domain_array_name]
    subdomains = fenics.MeshFunction("size_t", new_mesh, new_mesh.geometry().dim())
    subdomains.set_all(0)
    subdomains.array()[:] = material.astype(np.uint64)

    return new_mesh, subdomains



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
    mesh = fenics.Mesh()
    hdf = fenics.HDF5File(mesh.mpi_comm(),  path_to_file, "r")
    hdf.read(mesh, "/mesh", False)
    subdomains = fenics.MeshFunction("size_t", mesh, mesh.geometry().dim())
    if hdf.has_dataset('subdomains'):
        hdf.read(subdomains, "/subdomains")
    else:
        subdomains.set_all(0)
    boundaries = fenics.MeshFunction("size_t", mesh, mesh.geometry().dim()-1)
    if hdf.has_dataset('boundaries'):
        hdf.read(boundaries, "/boundaries")
    else:
        boundaries.set_all(0)
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
        return f
