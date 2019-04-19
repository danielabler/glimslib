import os

import SimpleITK as sitk

import test_cases.test_image_based_optimization_TCGA.testing_config as config

from glimslib import fenics_local as fenics, visualisation as plott, visualisation as vh
import glimslib.utils.data_io as dio
import glimslib.utils.image_registration_utils as reg
from matplotlib import pyplot as plt

# ==============================================================================
# Output DIR
# ==============================================================================
output_path = config.path_00_patient_specific_reference

# ==============================================================================
# Affine registration to match atlas to patient image
# ==============================================================================

# TODO Add mask for registration to obatain patient reference

path_to_reference_image = config.path_patient_image
path_to_moving_image    = config.path_atlas_image

path_to_output_image = '.'.join(config.path_3d_atlas_reg_to_patient_image.split('.')[:-1])
image_ext = 'nii'
path_to_affine_transform = os.path.join(output_path, 'atlas_to_patient.mat')

reg.register_ants(path_to_reference_image, path_to_moving_image, path_to_output_image,
                  path_to_transform=path_to_affine_transform, registration_type='Affine',
                  image_ext=image_ext, fixed_mask=None, moving_mask=None, verbose=1, dim=3)

# ==============================================================================
# Apply transform to labels
# ==============================================================================
path_to_input_image = config.path_atlas_seg
path_to_output_image = config.path_3d_atlas_reg_to_patient_seg

reg.ants_apply_transforms(path_to_input_image, path_to_reference_image, path_to_output_image,
                          [path_to_affine_transform], dim=3)

