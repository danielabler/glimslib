"""
Extracts 2D data from 3D image and atlas and saves as hdf5.
"""
import os
import SimpleITK as sitk

import test_cases.test_image_based_optimization_TCGA.testing_config as config
import glimslib.utils.data_io as dio
from glimslib.visualisation import plotting as plott



# ==============================================================================
# PATH SETTINGS
# ==============================================================================
path_to_patient_image = config.path_patient_image
path_to_patient_seg = config.path_patient_seg
path_to_atlas_image = config.path_3d_atlas_reg_to_patient_image
path_to_atlas_seg = config.path_3d_atlas_reg_to_patient_seg

# == extract slice from images
label_img_type = sitk.sitkUInt8
img_slice = 87

# -- load patient image
image           = sitk.ReadImage(path_to_patient_image)
image_select    = image[:, :, img_slice]
image_select_np = sitk.GetArrayFromImage(image_select)
f_img = dio.image2fct2D(image_select)
f_img.rename("imgvalue", "label")

# -- load patient seg
image_label     = sitk.ReadImage(path_to_patient_seg)
image_label_select      = image_label[:, :, img_slice]
image_label_select.SetOrigin( image_select.GetOrigin() )
image_label_select.SetDirection( image_select.GetDirection() )
image_label_select_np   = sitk.GetArrayFromImage(image_label_select)
f_img_label     = dio.image2fct2D(image_label_select)
f_img_label.rename("label", "label")

# -- load atlas image
atlas_image    = sitk.ReadImage(path_to_atlas_image)
atlas_image_select      = atlas_image[:, :, img_slice]
atlas_image_select.SetOrigin( image_select.GetOrigin() )
atlas_image_select.SetDirection( image_select.GetDirection() )
atlas_image_select_np   = sitk.GetArrayFromImage(atlas_image_select)
f_atlas_image     = dio.image2fct2D(atlas_image_select)
f_atlas_image.rename("label", "label")

# -- load atlas seg
atlas_label    = sitk.ReadImage(path_to_atlas_seg)
atlas_label_select      = atlas_label[:, :, img_slice]
atlas_label_select.SetOrigin( image_select.GetOrigin() )
atlas_label_select.SetDirection( image_select.GetDirection() )
atlas_label_select_np   = sitk.GetArrayFromImage(atlas_label_select)
f_atlas_label     = dio.image2fct2D(atlas_label_select)
f_atlas_label.rename("label", "label")


#-- plot
plott.show_img_seg_f(image=image_select, segmentation=image_label_select, show=True,
                     path=os.path.join(config.path_00_data_extraction, 'patient_image_2d.png'))
plott.show_img_seg_f(function=f_img, show=True,
                     path=os.path.join(config.path_00_data_extraction, 'patient_image_2d_function.png'))

plott.show_img_seg_f(image=image_label_select, segmentation=image_label_select, show=True,
                     path=os.path.join(config.path_00_data_extraction, 'patient_seg_2d.png'))
plott.show_img_seg_f(function=f_img_label, show=True,
                     path=os.path.join(config.path_00_data_extraction, 'patient_seg_2d_function.png'))

plott.show_img_seg_f(image=atlas_image_select, segmentation=atlas_label_select, show=True,
                     path=os.path.join(config.path_00_data_extraction, 'atlas_image_2d.png'))
plott.show_img_seg_f(function=atlas_image_select, show=True,
                     path=os.path.join(config.path_00_data_extraction, 'atlas_image_2d_function.png'))

plott.show_img_seg_f(image=atlas_label_select, segmentation=atlas_label_select, show=True,
                     path=os.path.join(config.path_00_data_extraction, 'atlas_seg_2d.png'))
plott.show_img_seg_f(function=f_atlas_label, show=True,
                     path=os.path.join(config.path_00_data_extraction, 'atlas_seg_2d_function.png'))


#-- save 2D images
sitk.WriteImage(image_select, config.path_to_2d_patient_image)
sitk.WriteImage(image_label_select, config.path_to_2d_patient_seg)
sitk.WriteImage(atlas_image_select, config.path_to_2d_atlas_image)
sitk.WriteImage(atlas_label_select, config.path_to_2d_atlas_seg)

#-- save label function:
dio.save_function_mesh(f_img,config.path_to_2d_patient_imagefunction)
dio.save_function_mesh(f_img_label,config.path_to_2d_patient_labelfunction)
dio.save_function_mesh(f_atlas_image,config.path_to_2d_atlas_imagefunction)
dio.save_function_mesh(f_atlas_label,config.path_to_2d_atlas_labelfunction)

