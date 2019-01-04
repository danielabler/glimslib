"""
This script illustrates conversion of vtu mesh into fenics hdf5 format that can be loaded during parallel execution
"""
import os

import meshio as mio

import test_cases.test_simulation_tumor_growth.testing_config as test_config
import utils.data_io as dio
import utils.file_utils as fu


# load from VTU
inpath = os.path.join(test_config.test_data_dir,'brain_atlas_mesh_3d.vtu')
mymesh = mio.read(inpath)
mesh_tmp, subdomains_tmp = dio.convert_meshio_to_fenics_mesh(mymesh)

# save as hdf5 file for parallel processing
path_to_hdf5_mesh = os.path.join(test_config.test_data_dir,'brain_atlas_mesh_3d.hdf5')
fu.ensure_dir_exists(path_to_hdf5_mesh)
dio.save_mesh_hdf5(mesh_tmp, path_to_hdf5_mesh, subdomains=subdomains_tmp)



# load from image
path_to_atlas   = os.path.join(test_config.test_data_dir,'brain_atlas_image_3d.mha')
labelfunction = dio.get_labelfunction_from_image(path_to_atlas, 87)
mesh          = labelfunction.function_space().mesh()
# create subdomains
from simulation.helpers.helper_classes import SubDomains
subdomains = SubDomains(mesh)
subdomains.setup_subdomains(label_function=labelfunction)
#save mesh as hdf5
path_to_hdf5_mesh = os.path.join(test_config.test_data_dir,'brain_atlas_mesh_2d.hdf5')
fu.ensure_dir_exists(path_to_hdf5_mesh)
dio.save_mesh_hdf5(mesh, path_to_hdf5_mesh, subdomains=subdomains.subdomains)

# save labelfunction
path_to_hdf5_label = os.path.join(test_config.test_data_dir,'brain_atlas_labelfunction_2d.hdf5')
dio.save_functions_hdf5({"labelfunction" : labelfunction}, path_to_hdf5_label, time_step=None)