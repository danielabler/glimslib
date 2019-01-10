import os
from builtins import filter

import SimpleITK as sitk

import test_cases.test_image_based_optimisation.testing_config as test_config
import visualisation.plotting as plott

import fenics_local as fenics
import utils.file_utils as fu
import utils.data_io as dio
import utils.vtk_utils as vtu
import utils.image_registration_utils as reg
import numpy as np
from matplotlib import pyplot as plt
import visualisation.helpers as vh


# ==============================================================================
# Output DIR
# ==============================================================================
output_path = os.path.join(test_config.output_path,'03_estimate_deformation_from_image')

# ==============================================================================
# Registration to obtain displacement field
# ==============================================================================
path_to_reference_image = test_config.path_to_2d_image
path_to_deformed_image  = os.path.join(test_config.output_path, '02_create_deformed_image', 'T1_img_warped.nii')

path_to_warped_image = os.path.join(output_path, 'registered_image_deformed_to_reference')
path_to_warp_field = os.path.join(output_path, 'registered_image_deformed_to_reference_warp.nii.gz')

reg.register_ants(path_to_reference_image, path_to_deformed_image, path_to_warped_image,
                  path_to_transform=path_to_warp_field, registration_type='Syn',
                  image_ext='nii', fixed_mask=None, moving_mask=None, verbose=1, dim=2)

# ==============================================================================
# Read Registration Result, convert to fenics function, save
# ==============================================================================

image_warp     = sitk.ReadImage(path_to_warp_field)
image_warp_np = sitk.GetArrayFromImage(image_warp)

plt.imshow(image_warp_np[:,:,0])
fig = plt.gcf()
path_to_fig = os.path.join(output_path, 'displacement_from_registration_0.png')
fig.savefig(path_to_fig)

plt.imshow(image_warp_np[:,:,1])
fig = plt.gcf()
path_to_fig = os.path.join(output_path, 'displacement_from_registration_1.png')
fig.savefig(path_to_fig)

# convert to fenics ... this is very slow
f_img = dio.create_fenics_function_from_image(image_warp)

plott.show_img_seg_f(function=f_img, show=True,
                     path=os.path.join(output_path, 'displacement_from_registration_fenics.png'),
                     dpi=300)

path_to_displacement_from_image = os.path.join(output_path, 'displacement_from_image.h5')
dio.save_function_mesh(f_img, path_to_displacement_from_image)

# ==============================================================================
# Load concentration and deformation fields from image
# ==============================================================================
path_to_displacement_from_image = os.path.join(output_path, 'displacement_from_image.h5')
disp_est, mesh_est, subdomains_est, boundaries_est = dio.load_function_mesh(path_to_displacement_from_image,
                                                                functionspace='vector')
plott.show_img_seg_f(function=disp_est, show=True,
                     path=os.path.join(output_path, 'displacement_field_from_registration.png'),
                     dpi=300)

# ==============================================================================
# Load concentration and deformation fields at last time step
# ==============================================================================
sim_out_path = os.path.join(test_config.output_path, '01_forward_simulation')
path_to_conc_sim = os.path.join(sim_out_path, 'concentration_simulated.h5')
path_to_disp_sim = os.path.join(sim_out_path, 'displacement_simulated.h5')

conc_sim, mesh_sim, subdomains_sim, boundaries_sim = dio.load_function_mesh(path_to_conc_sim, functionspace='function')
disp_sim, mesh_sim, subdomains_sim, boundaries_sim = dio.load_function_mesh(path_to_disp_sim, functionspace='vector')


plott.show_img_seg_f(function=conc_sim, show=True,
                     path=os.path.join(output_path, 'concentration_field_from_simulation.png'),
                     dpi=300)
plott.show_img_seg_f(function=disp_sim, show=True,
                     path=os.path.join(output_path, 'displacement_field_from_simulation.png'),
                     dpi=300)
plott.show_img_seg_f(function=vh.convert_meshfunction_to_function(mesh_sim, subdomains_sim), show=True,
                     path=os.path.join(output_path, 'labelmap_from_simulation.png'),
                     dpi=300)

# ==============================================================================
# Compare estimated and actual displacement fields
# ==============================================================================

def interpolate_non_matching(source_function, target_funspace):
    function_new = fenics.Function(target_funspace)
    fenics.LagrangeInterpolator().interpolate(function_new, source_function)
    return function_new

#-- chose simulation mesh as reference
funspace_ref = disp_sim.function_space()
#-- project/interpolate estimated displacement field over that mesh
disp_est_ref = interpolate_non_matching(disp_est, funspace_ref)
plott.show_img_seg_f(function=disp_est_ref, show=True,
                     path=os.path.join(output_path, 'displacement_field_from_registration_ref_space.png'),
                     dpi=300)
# compute errornorm
print(fenics.errornorm(disp_sim, disp_est_ref))
# compute difference field
disp_diff = fenics.project(disp_sim-disp_est_ref, funspace_ref)
plott.show_img_seg_f(function=disp_diff, show=True,
                     path=os.path.join(output_path, 'displacement_field_difference.png'),
                     dpi=300)

# ==============================================================================
# Use reduced domain as reference
# ==============================================================================

path_to_hdf5_mesh = os.path.join(test_config.test_data_dir, 'brain_atlas_mesh_2d_reduced_domain.h5')
mesh_reduced, subdomains_reduced, boundaries_reduced = dio.read_mesh_hdf5(path_to_hdf5_mesh)

#-- chose simulation mesh as reference
funspace_disp_reduced = fenics.VectorFunctionSpace(mesh_reduced, 'Lagrange', 1)
funspace_conc_reduced = fenics.FunctionSpace(mesh_reduced, 'Lagrange', 1)
#-- project/interpolate displacement field over that mesh
disp_sim_reduced = interpolate_non_matching(disp_sim, funspace_disp_reduced)
disp_est_reduced = interpolate_non_matching(disp_est, funspace_disp_reduced)
conc_sim_reduced = interpolate_non_matching(conc_sim, funspace_conc_reduced)

plott.show_img_seg_f(function=disp_sim_reduced, show=True,
                     path=os.path.join(output_path, 'displacement_field_from_simulation_reduced.png'),
                     dpi=300)
plott.show_img_seg_f(function=disp_est_reduced, show=True,
                     path=os.path.join(output_path, 'displacement_field_from_registration_reduced.png'),
                     dpi=300)
plott.show_img_seg_f(function=conc_sim_reduced, show=True,
                     path=os.path.join(output_path, 'cocnentration_field_from_simulation_reduced.png'),
                     dpi=300)

# compute errornorm
print(fenics.errornorm(disp_sim_reduced, disp_est_reduced))
# compute difference field
disp_diff_reduced = fenics.project(disp_sim_reduced - disp_est_reduced, funspace_disp_reduced)
plott.show_img_seg_f(function=disp_diff_reduced, show=True,
                     path=os.path.join(output_path, 'displacement_field_difference_reduced.png'),
                     dpi=300)

# save functions on reduced domain
path_to_sim_conc_reduced = os.path.join(output_path, 'concentration_from_simulation_reduced.h5')
dio.save_function_mesh(conc_sim_reduced, path_to_sim_conc_reduced, subdomains=subdomains_reduced)

path_to_sim_disp_reduced = os.path.join(output_path, 'displacement_from_simulation_reduced.h5')
dio.save_function_mesh(disp_sim_reduced, path_to_sim_disp_reduced, subdomains=subdomains_reduced)

path_to_est_disp_reduced = os.path.join(output_path, 'displacement_from_registration_reduced.h5')
dio.save_function_mesh(disp_est_reduced, path_to_est_disp_reduced, subdomains=subdomains_reduced)