"""
Loads simulation result.
The original image is warped by the simulated deformation field to obtain a 'synthetic' tumor-bearing image.
"""
import os

import SimpleITK as sitk

import test_cases.test_image_based_optimisation.testing_config as test_config

import fenics_local as fenics
import utils.file_utils as fu
import utils.data_io as dio
import utils.vtk_utils as vtu
import utils.image_registration_utils as reg

output_path = test_config.path_02_deformed_image

# 1) Load reference image
T1 = sitk.ReadImage(test_config.path_to_2d_image)
size_T1 = T1.GetSize()

# 2) Convert simulation result to labelmap with predefined resolution
resolution = (*size_T1,1)#(100,100,1)
# read
name_sim_vtu = dio.create_file_name('all', test_config.params_sim["sim_time"])
path_to_sim_vtu = os.path.join(test_config.path_01_forward_simulation, 'merged', name_sim_vtu)
sim_vtu = vtu.read_vtk_data(path_to_sim_vtu)
# convert vtu to vti
sim_vti = vtu.resample_to_image(sim_vtu,resolution)
path_to_sim_vti = os.path.join(output_path,'simulation_as_image.vti')
vtu.write_vtk_data(sim_vti, path_to_sim_vti)

if fenics.is_version("<2018.1.x"):
    # convert vti to normal image
    label_img = vtu.convert_vti_to_img(sim_vti, array_name='label_map', RGB=False)
    path_label_img = os.path.join(output_path, 'label_img_orig.nii')
    sitk.WriteImage(label_img, path_label_img)

# 3) Create deformed labelmap
sim_vtu_warped = vtu.warpVTU(sim_vtu, 'point', 'displacement')

sim_vti_warped = vtu.resample_to_image(sim_vtu_warped, resolution)
path_to_sim_vti_warped = os.path.join(output_path,'simulation_as_image_warped.vti')
vtu.write_vtk_data(sim_vti_warped, path_to_sim_vti_warped)

if fenics.is_version("<2018.1.x"):
    label_img_warped = vtu.convert_vti_to_img(sim_vti_warped, array_name='label_map', RGB=False)
    path_label_img_warped = os.path.join(output_path, 'label_img_warped.nii')
    sitk.WriteImage(label_img_warped, path_label_img_warped)

# 4) Extract displacement field
disp_img_RGB = vtu.convert_vti_to_img(sim_vti, array_name='displacement', RGB=True)
path_disp_img = os.path.join(output_path, 'displacement_img.nii')
sitk.WriteImage(disp_img_RGB, path_disp_img)

disp_img_RGB_inv = vtu.convert_vti_to_img(sim_vti, array_name='displacement', RGB=True, invert_values=True)
path_disp_img_inv = os.path.join(output_path, 'displacement_img_inv.nii')
sitk.WriteImage(disp_img_RGB_inv, path_disp_img_inv)

# 5) Apply deformation field to warp image
os.environ.copy()

#-- Warp atlas T1 image:
#- 1) atlas T1 by deformation field
output_T1_warped = os.path.join(output_path, 'T1_img_warped.nii')

reg.ants_apply_transforms(input_img=test_config.path_to_2d_image, output_file=output_T1_warped, reference_img=test_config.path_to_2d_image,
                          transforms=[path_disp_img_inv], dim=2)

if fenics.is_version("<2018.1.x"):
    #- 2) resample label map to T1
    output_label_resampled = os.path.join(output_path, 'label_img_resampledToT1.nii')
    reg.ants_apply_transforms(input_img=path_label_img, output_file=output_label_resampled, reference_img=test_config.path_to_2d_image,
                              transforms=[], dim=2)

    #- 3) resample label map to T1
    output_label_resampled_warped = os.path.join(output_path, 'label_img_resampledToT1_warped.nii')
    reg.ants_apply_transforms(input_img=path_label_img, output_file=output_label_resampled_warped, reference_img=test_config.path_to_2d_image,
                              transforms=[path_disp_img_inv], dim=2)

