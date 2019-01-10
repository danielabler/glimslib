"""
Extracts 2D data from 3D image and atlas and saves as hdf5.
"""
import os
import SimpleITK as sitk

import test_cases.test_image_based_optimisation.testing_config as config
import utils.data_io as dio
import visualisation.plotting as plott

path_to_atlas = os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')
path_to_image = os.path.join(config.test_data_dir, 'brain_atlas_image_t1_3d.mha')

img_slice = 87

# -- load brain atlas labels
image_label     = sitk.ReadImage(path_to_atlas)
image_label_select      = image_label[:, :, img_slice]
image_label_select_np   = sitk.GetArrayFromImage(image_label_select)
f_img_label     = dio.image2fct2D(image_label_select)
f_img_label.rename("label", "label")
# alternatively:
#labelfunction = dio.get_labelfunction_from_image(path_to_atlas, img_slice)

# -- load brain atlas image
image           = sitk.ReadImage(path_to_image)
image_select    = image[:, :, img_slice]
image_select_np = sitk.GetArrayFromImage(image_select)
image_select.SetOrigin( image_label_select.GetOrigin() )
image_select.SetDirection( image_label_select.GetDirection() )
f_img = dio.image2fct2D(image_select)
f_img.rename("imgvalue", "label")


#-- plot
plott.show_img_seg_f(image=image_select, segmentation=image_label_select, show=True,
                     path=os.path.join(config.output_path, 'image_label_from_sitk_image.png'))

plott.show_img_seg_f(function=f_img_label, show=True,
                     path=os.path.join(config.output_path, 'image_label_from_fenics_function.png'))

plott.show_img_seg_f(function=f_img, show=True,
                     path=os.path.join(config.output_path, 'image_from_fenics_function.png'))

#-- save 2D images
sitk.WriteImage(image_select, config.path_to_2d_image)
sitk.WriteImage(image_label_select, config.path_to_2d_labels)

#-- save label function:
dio.save_function_mesh(f_img,config.path_to_2d_imagefunction)
dio.save_function_mesh(f_img_label,config.path_to_2d_labelfunction)


