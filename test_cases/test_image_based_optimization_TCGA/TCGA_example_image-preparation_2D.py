import logging
import os
from fenics import *
import numpy as np
import SimpleITK as sitk

from glimslib import config
config.USE_ADJOINT = False
from glimslib.utils import file_utils as fu


# ==============================================================================
# PATH SETTINGS
# ==============================================================================
data_path   = os.path.join(config.test_data_dir, 'TCGA')
output_path = os.path.join(config.output_dir_testing, 'TCGA')
reg_path    = os.path.join(output_path, "registration_results_small_mask")

fu.ensure_dir_exists(output_path)

# ==============================================================================
# LOAD IMAGES TO DEFINE DOMAIN
# ==============================================================================
label_img_type = sitk.sitkUInt8
img_slice = 87

# -- load patient brain image
path_to_patient_brain       = os.path.join(data_path, 'TCGA-06-0190_2004-12-10_t1Gd.mha')
image_patient_brain         = sitk.ReadImage(path_to_patient_brain)
image_patient_brain_select  = image_patient_brain[:, :, img_slice]
#image_patient_brain_select_select_np = sitk.GetArrayFromImage(image_patient_brain_select)
f_patient_brain           = dfi.image2fct2D(image_patient_brain_select)
f_patient_brain.rename("T1", "label")

# -- load patient brain tumor segmentation
path_to_patient_seg       = os.path.join(data_path, 'TCGA-06-0190_2004-12-10_GlistrBoost_ManuallyCorrected_T1-label-5_T2-label-6.mha')
image_patient_seg         = sitk.Cast(sitk.ReadImage(path_to_patient_seg), label_img_type)
image_patient_seg_select  = image_patient_seg[:, :, img_slice]
#image_patient_brain_select_select_np = sitk.GetArrayFromImage(image_patient_brain_select)
f_patient_seg           = dfi.image2fct2D(image_patient_seg_select)
f_patient_seg.rename("label", "label")


# -- load patient brain tumor mask
#path_to_patient_mask      = os.path.join(data_path, 'TCGA-06-0190_2004-12-10_GlistrBoost_ManuallyCorrected_tumor-mask.mha')
path_to_patient_mask      = os.path.join(data_path, 'TCGA-06-0190_2004-12-10_GlistrBoost_ManuallyCorrected_mask_small.mha')
image_patient_mask         = sitk.Cast(sitk.ReadImage(path_to_patient_mask), label_img_type)
image_patient_mask_select  = image_patient_mask[:, :, img_slice]
#image_patient_brain_select_select_np = sitk.GetArrayFromImage(image_patient_brain_select)
f_patient_mask           = dfi.image2fct2D(image_patient_mask_select)
f_patient_mask.rename("label", "label")



#=== ATLAS -- AFFINE REGISTRATION -> from 3D AFFINE REGISTRATION
# -- load brain atlas image (result of affine registration of atlas to patient with tumor masked)
path_to_atlas_image_affine   = os.path.join(data_path, reg_path, 'SRI24_T1_RAI_RegAffine.mha')
image_atlas_image_affine     = sitk.ReadImage(path_to_atlas_image_affine)
image_atlas_image_affine_select    = image_atlas_image_affine[:, :, img_slice]
#image_atlas_image_select_np = sitk.GetArrayFromImage(image_atlas_image_select)
f_atlas_image_affine          = dfi.image2fct2D(image_atlas_image_affine_select)
f_atlas_image_affine.rename("T1", "label")
# -- load brain atlas labels (result of affine registration of atlas to patient with tumor masked)
path_to_atlas_labels_affine    = os.path.join(data_path, reg_path, 'atlas_reg_affine.mha')
image_atlas_labels_affine             = sitk.Cast(sitk.ReadImage(path_to_atlas_labels_affine), label_img_type)
image_atlas_labels_affine_select      = image_atlas_labels_affine[:, :, img_slice]
#image_label_select_np   = sitk.GetArrayFromImage(image_label_select)
f_atlas_labels_affine     = dfi.image2fct2D(image_atlas_labels_affine_select)
f_atlas_labels_affine.rename("label", "label")


#=== ATLAS -- DEFORMABLE REGISTRATION => 2D DEFORMABLE REGISTRATION FROM 3d AFFINE REGISTERED
# -- load brain atlas image (result of affine registration of atlas to patient with tumor masked, + DEF REG)
path_to_atlas_image_def   = os.path.join(data_path, reg_path, 'atlas_T1_reg_def_2D.mha')
image_atlas_image_def_select     = sitk.ReadImage(path_to_atlas_image_def)
#image_atlas_image_select_np = sitk.GetArrayFromImage(image_atlas_image_select)
f_atlas_image_def          = dfi.image2fct2D(image_atlas_image_def_select)
f_atlas_image_def.rename("T1", "label")

# -- load brain atlas labels (result of def registration of atlas to patient with tumor masked, + DEF REG)
path_to_atlas_labels_def    = os.path.join(data_path, reg_path, 'atlas_labels_reg_def_2D.mha')
image_atlas_labels_def_select             = sitk.Cast(sitk.ReadImage(path_to_atlas_labels_def), label_img_type)
#image_label_select_np   = sitk.GetArrayFromImage(image_label_select)
f_atlas_labels_def     = dfi.image2fct2D(image_atlas_labels_def_select)
f_atlas_labels_def.rename("label", "label")

# # -- load brain atlas labels & tumor seg (result of def registration of atlas to patient with tumor masked, + DEF REG)
path_to_atlas_labels_all    = os.path.join(data_path, reg_path, 'atlas_labels_reg_def_2D_all.mha')
image_atlas_labels_all             = sitk.Cast(sitk.ReadImage(path_to_atlas_labels_all), label_img_type)
image_atlas_labels_all_select = image_atlas_labels_all[:,:,0]
#image_label_select_np   = sitk.GetArrayFromImage(image_label_select)
f_atlas_labels_all     = dfi.image2fct2D(image_atlas_labels_all_select)
f_atlas_labels_all.rename("label", "label")


plotrange = [40, 200, -20, -210]
# plott.show_img_seg_f(image_patient_brain_select, image_patient_seg_select,
#                      contour=True, plot_range=plotrange,
#                      path=os.path.join(data_path, "patient-seg_on_patient-img_from-2D.png"))
#
# plott.show_img_seg_f(image_atlas_image_affine_select, image_atlas_labels_affine_select,
#                      contour=True, plot_range=plotrange,
#                      path=os.path.join(data_path, "atlas-labels_on_atlas-img_from-2D-affine.png"))
#
# plott.show_img_seg_f(image_patient_brain_select, image_atlas_labels_affine_select,
#                      contour=True, plot_range=plotrange,
#                      path=os.path.join(data_path, "atlas-labels_on_patient-img_from-2D-affine.png"))
#
# plott.show_img_seg_f(image_atlas_image_def_select, image_atlas_labels_def_select,
#                      contour=True, plot_range=plotrange,
#                      path=os.path.join(data_path, "atlas-labels_on_atlas-img_from-2D-def.png"))
# plott.show_img_seg_f(image_patient_brain_select, image_atlas_labels_def_select, contour=True, plot_range=plotrange,
#                      path=os.path.join(data_path, "atlas-labels_on_patient-img_from-2D-def.png"))
# plott.show_img_seg_f(image_patient_brain_select, image_atlas_labels_all_select,
#                      contour=True, plot_range=plotrange,
#                      path=os.path.join(data_path, "atlas-labels-all_on_patient-img_from-2D-def.png"))
#




#== write out selected 2D images to compute deformation based on 2D
# path_to_atlas_image_affine_2D   = os.path.join(data_path, reg_path, 'SRI24_T1_RAI_RegAffine_2D.mha')
# sitk.WriteImage(image_atlas_image_affine_select, path_to_atlas_image_affine_2D)
#
# path_to_atlas_labels_affine_2D   = os.path.join(data_path, reg_path, 'SRI24_labels_RAI_RegAffine_2D.mha')
# sitk.WriteImage(image_atlas_labels_affine_select, path_to_atlas_labels_affine_2D)
#
#
# path_to_patient_brain_2D        = os.path.join(data_path, reg_path, 'TCGA-06-0190_2004-12-10_t1Gd_2D.mha')
# sitk.WriteImage(image_patient_brain_select, path_to_patient_brain_2D)
#
# path_to_patient_mask_2D      = os.path.join(data_path, 'TCGA-06-0190_2004-12-10_GlistrBoost_ManuallyCorrected_tumor-mask_2D_small.mha')
# sitk.WriteImage(image_patient_mask_select, path_to_patient_mask_2D)
#
path_to_patient_seg_2D      = os.path.join(data_path, 'TCGA-06-0190_2004-12-10_GlistrBoost_ManuallyCorrected_seg_2D.mha')
sitk.WriteImage(image_patient_seg_select, path_to_patient_seg_2D)



#== Get Deformation Field from Warp Image

#path_to_warp_image    = os.path.join(data_path, reg_path, 'SRI24_T1_RAI_RegDeformable1Warp.nii.gz')
#== need inverse for application on points
# path_to_warp_image    = os.path.join(data_path, reg_path, 'SRI24_T1_RAI_RegDeformable1InverseWarp.nii.gz')
# image_warp           = sitk.ReadImage(path_to_warp_image)
# image_warp_select     = image_warp[:, :, img_slice]

path_to_warp_image    = os.path.join(data_path, reg_path, 'SRI24_T1_RAI_RegAffine_2D_RegDeformable_2D1InverseWarp.nii.gz')
image_warp_select           = sitk.ReadImage(path_to_warp_image)


def extract_pixel_component(image, dim=0):
    image_np = sitk.GetArrayFromImage(image)
    if image.GetDimension()==2:
        image_comp_np = image_np[:,:,dim]
    elif image.GetDimension()==3:
        image_comp_np = image_np[:, :, :, dim]
    image_comp = sitk.GetImageFromArray(image_comp_np)
    image_comp.SetOrigin(image.GetOrigin())
    image_comp.SetSpacing(image.GetSpacing())
    image_comp.SetDirection(image.GetDirection())
    return image_comp

def extract_abs_from_image(image):
    image_np = sitk.GetArrayFromImage(image)
    image_np = image_np[:,:,:2] # only take x and y component of deformation field
    image_comp_np_abs = np.linalg.norm(image_np, axis=2)
    image_comp = sitk.GetImageFromArray(image_comp_np_abs)
    image_comp.SetOrigin(image.GetOrigin())
    image_comp.SetSpacing(image.GetSpacing())
    image_comp.SetDirection(image.GetDirection())
    return image_comp

image_warp_select_abs = extract_abs_from_image(image_warp_select)
f_image_warp_abs     = dfi.image2fct2D(image_warp_select_abs)
f_image_warp_abs.rename("deformation_abs", "label")


image_warp_select_x = extract_pixel_component(image_warp_select,0)
f_image_warp_x     = dfi.image2fct2D(image_warp_select_x)
f_image_warp_x.rename("deformation_x", "label")

image_warp_select_y = extract_pixel_component(image_warp_select,1)
f_image_warp_y     = dfi.image2fct2D(image_warp_select_y)
f_image_warp_y.rename("deformation_y", "label")




range = [-15,15]
excludearound=None#[0,0.5]
alpha = 1
colormap = 'RdBu_r'
# ilsf.show_img_seg_f(image_patient_brain_select, image_atlas_labels_all_select, f_image_warp_abs, contour=True, showmesh=False, alpha_f=alpha,
#                                range_f=[0, 20], exclude_min_max=True, colormap=plt.cm.get_cmap('viridis', 20),
#                                label="displacement abs", show=True, plot_range=plotrange, exclude_around=excludearound,
#                                 path=os.path.join(data_path, "patient-brain_atlas-labels_deformation-abs_from-2D.png"))


# plott.show_img_seg_f(image_patient_brain_select, image_atlas_labels_all_select, f_image_warp_x, contour=True, showmesh=False, alpha_f=alpha,
#                      range_f=range, exclude_min_max=True, colormap=plt.cm.get_cmap(colormap, 20),
#                      label="displacement x", show=True, plot_range=plotrange, exclude_around=excludearound,
#                      path=os.path.join(data_path, "patient-brain_atlas-labels_deformation-x_from-2D.png"))
#
# plott.show_img_seg_f(image_patient_brain_select, image_atlas_labels_all_select, f_image_warp_y, contour=True, showmesh=False, alpha_f=alpha,
#                      range_f=range, exclude_min_max=True, colormap=plt.cm.get_cmap(colormap, 20),
#                      label="displacement y", show=True, plot_range=plotrange, exclude_around=excludearound,
#                      path=os.path.join(data_path, "patient-brain_atlas-labels_deformation-y_from-2D.png"))
#
#
#
# # ilsf.show_img_seg_f(image_atlas_image_affine_select, image_atlas_labels_all_select, f_image_warp_abs, contour=True, showmesh=False, alpha_f=alpha,
# #                                range_f=range, exclude_as_range=True, colormap=plt.cm.get_cmap('viridis', 20),
# #                                label="displacement abs", show=True)
#
# plott.show_img_seg_f(image_atlas_image_affine_select, image_atlas_labels_all_select, f_image_warp_x, contour=True, showmesh=False, alpha_f=alpha,
#                      range_f=range, exclude_min_max=True, colormap=plt.cm.get_cmap(colormap, 20),
#                      label="displacement x", show=True, plot_range=plotrange, exclude_around=excludearound,
#                      path=os.path.join(data_path, "atlas_image_atlas-labels_deformation-x_from-2D.png"))
#
# plott.show_img_seg_f(image_atlas_image_affine_select, image_atlas_labels_all_select, f_image_warp_y, contour=True, showmesh=False, alpha_f=alpha,
#                      range_f=range, exclude_min_max=True, colormap=plt.cm.get_cmap(colormap, 20),
#                      label="displacement y", show=True, plot_range=plotrange, exclude_around=excludearound,
#                      path=os.path.join(data_path, "atlas_image_atlas-labels_deformation-y_from-2D.png"))


#=== build displacement field

mesh  = f_image_warp_x.function_space().mesh()
V = f_image_warp_x.function_space()
Vvect = VectorFunctionSpace(mesh, "Lagrange", 1)
u = Function(Vvect)

u_x      = project(f_image_warp_x, V)
u_y      = project(f_image_warp_y, V)
assigner_x = FunctionAssigner(Vvect.sub(0), V)
assigner_x.assign(u.sub(0), u_x)
assigner_y = FunctionAssigner(Vvect.sub(1), V)
assigner_y.assign(u.sub(1), u_y)



plot_obj_img_aff = {'object'        : image_atlas_image_affine_select,
                'segmentation'  : image_atlas_labels_all_select,
                'label_alpha'   : 0.3,
                'contour'       : True,
                'cmap'          : 'Greys',
                'origin'        : 'upper',
}


plot_obj_img_def = {'object'        : image_atlas_image_def_select,
                'segmentation'  : image_atlas_labels_all_select,
                'label_alpha'   : 0.3,
                'contour'       : True,
                'cmap'          : 'Greys',
                'origin'        : 'upper',
}

plot_obj_disp_target   = {'object'        : u,
                 'alpha'         : 1,
                 'range_f'       : [0,20],
                 'exclude_below' : 1,
                 'cbar_label'    : 'displacement',
                 'n_cmap_levels' : 20,
                 #'color' : 'g',
                 #'exclude_min_max' : True,
                 'interpolate':True,
                 'n_interpolate' : 50,
                 'units' : 'xy',
                 'scale_units' :'xy',
                 'scale' :1,
                 #'edgecolor' : 'k',
                 #'linewidth' : 0.3
                           }


plott.plot([plot_obj_img_aff, plot_obj_disp_target], title='affine', plot_range=plotrange, dpi=400,
           save_path=os.path.join(data_path, "atlas_affine_with_segmentation.png"), show=DO_PLOT)
plott.plot([plot_obj_img_def, plot_obj_disp_target], title='deformed', plot_range=plotrange,dpi=400,
            save_path=os.path.join(data_path, "atlas_deformed_with_segmentation.png"), show=DO_PLOT)



#=== Try to Warp



# this seems to be necessary to actually move nodes
f_atlas_image_affine_new = project(f_atlas_image_affine, V)
f_atlas_image_affine_warped = feu.apply_deformation(f_atlas_image_affine_new, u)

f_atlas_labels_affine_new = project(f_atlas_labels_affine, V)
f_atlas_labels_affine_warped = feu.apply_deformation(f_atlas_labels_affine_new, u)


np.linalg.norm(f_atlas_image_affine.function_space().mesh().coordinates() - f_atlas_image_affine_warped.function_space().mesh().coordinates())

# plott.plot_plt(f_atlas_image_affine_warped, title='affine - warped')
# plott.plot_plt(f_atlas_image_def, title='deformably registered')
#
# plott.plot_plt(f_atlas_labels_def, title='deformably registered')
#
# plott.plot_plt(f_atlas_image_affine, title='affine')
#

plot_obj_atlas_orig  = {'object'        : f_atlas_image_affine,
                           'alpha'         : 1,
                           'cmap' : 'Greys_r'}


plot_obj_atlas_warped   = {'object'        : f_atlas_image_affine_warped,
                           'alpha'         : 1,
                           'cmap' : 'Greys_r'
                                    }

plot_obj_atlas_labels_orig   = {'object'        : f_atlas_labels_affine,
                           'alpha'         : 1,
                                    }

plot_obj_atlas_labels_warped   = {'object'        : f_atlas_labels_affine_warped,
                           'alpha'         : 1
                                    }

plot_obj_img_seg = {'object'        : image_atlas_labels_all_select,
                'alpha'   : 0.6,
                'origin'        : 'upper',
}

plott.plot([plot_obj_atlas_orig, plot_obj_img_seg, plot_obj_disp_target], title='atlas_orig_fenics',dpi=400,
           plot_range=plotrange, save_path=os.path.join(data_path, "atlas_orig_fenics_grey.png"), show=DO_PLOT)
plott.plot([plot_obj_atlas_warped, plot_obj_img_seg, plot_obj_disp_target], title='atlas_warped_fenics',dpi=400,
           plot_range=plotrange, save_path=os.path.join(data_path, "atlas_warped_fenics_grey.png"), show=DO_PLOT)
plott.plot([ plot_obj_atlas_labels_orig, plot_obj_disp_target], title='atlas_labels_orig_fenics', plot_range=plotrange,dpi=400,
           save_path=os.path.join(data_path, "atlas_orig_fenics_labels.png"), show=DO_PLOT)
plott.plot([ plot_obj_atlas_labels_warped, plot_obj_disp_target], title='atlas_labels_warped_fenics', plot_range=plotrange,dpi=400,
           save_path=os.path.join(data_path, "atlas_warped_fenics_labels.png"), show=DO_PLOT)
stop

#=== FOR SIMULATION
# 1) Domain == Atlas labelmap from affine registration to patient

path_to_atlas_deformed_file = os.path.join(data_path, reg_path, 'patient_domain_from_affine.xdmf')
dfi.save_labelfunction(path_to_atlas_deformed_file, f_atlas_labels_def)

# 2) start position == center of T1 segmentation




f_conc = f_patient_seg.copy(deepcopy=True)
f_conc_np = f_conc.vector().array()
#f_conc_np = float(f_conc_np)
f_conc_np[np.where(f_conc_np==5)]= 0.8# label 5 -> T1
f_conc_np[np.where(f_conc_np==6)]= 0.12# label 5 -> T1
np.unique(f_conc_np)
f_conc.vector().set_local( f_conc_np )




# -- load brain atlas labels & tumor seg (result of def registration of atlas to patient with tumor masked, + DEF REG)
path_to_atlas_labels_all_3D    = os.path.join(data_path, reg_path, 'TCGA-06-0190_Atlas_labels-combined.mha')
image_atlas_labels_all_3D             = sitk.Cast(sitk.ReadImage(path_to_atlas_labels_all_3D), label_img_type)
image_atlas_labels_all_3D_select      = image_atlas_labels_all_3D[:, :, img_slice]
#image_label_select_np   = sitk.GetArrayFromImage(image_label_select)
f_atlas_labels_all_3D     = dfi.image2fct2D(image_atlas_labels_all_3D_select)
f_atlas_labels_all_3D.rename("label", "label")




plott.show_img_seg_f(image_patient_brain_select, image_atlas_labels_all_3D_select, f_conc, contour=True, showmesh=False, alpha_f=alpha,
                     range_f=[0.001,1], exclude_as_range=True, colormap=plt.cm.get_cmap(colormap, 10),
                     label="relative tumor cell concentration from T1/T2 MRI", show=True, plot_range=plotrange, exclude_around=None,
                     path=os.path.join(data_path, "T1_conc_from-2D.png"))




#=== FOR ADJOINT OPTIMIZATION
# 1) Tumour Contours T1, T2 == tumour segmentation from
f_T1 = f_patient_seg.copy(deepcopy=True)
f_T1_np = f_T1.vector().array()
f_T1_np[np.where(f_T1_np==5)]= 1# label 5 -> T1
f_T1_np[np.where(f_T1_np>1)]= 0# label 5 -> T1
np.unique(f_T1_np)
f_T1.vector().set_local( f_T1_np )
plott.plot_plt(f_T1)

path_to_T1_file = os.path.join(data_path, reg_path, 'target_T1_threshold.xdmf')
dfi.save_function_to_xdmf(path_to_T1_file, f_T1, 'T1_threshold')





f_T2 = f_patient_seg.copy(deepcopy=True)
f_T2_np = f_T2.vector().array()
f_T2_np[np.where(f_T2_np>=5)]= 1# label 6 -> T2, we include T1
f_T2_np[np.where(f_T2_np>1)]= 0# label 6 -> T2
np.unique(f_T2_np)
f_T2.vector().set_local( f_T2_np )
plott.plot_plt(f_T2)

path_to_T2_file = os.path.join(data_path, reg_path, 'target_T2_threshold.xdmf')
dfi.save_function_to_xdmf(path_to_T2_file, f_T2, 'T2_threshold')


# 2) Displacement Field == inverse warp field from deformable registration of (affine registered) Atlas MR image to patient image

path_to_displacement_file = os.path.join(data_path, reg_path, 'target_displacement.xdmf')
dfi.save_function_to_xdmf(path_to_displacement_file, u, 'displacement')

