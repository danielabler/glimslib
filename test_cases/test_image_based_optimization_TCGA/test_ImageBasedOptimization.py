import os
from test_cases.test_image_based_optimization_TCGA.ImageBasedOptimization import ImageBasedOptimization

## PARAMS

params_sim = {
                "sim_time" : 50,
                "sim_time_step" : 1,
             }

params_fix  = { "E_GM" : 3000E-6,
                "E_WM" : 3000E-6,
                "E_CSF" :1000E-6,
                "E_VENT" :1000E-6,
                "nu_GM" : 0.45,
                "nu_WM" : 0.45,
                "nu_CSF" : 0.45,
                "nu_VENT" : 0.3
                }

params_target = {
                "D_WM" :     0.1,
                "D_GM" :     0.02,
                "rho_WM" :   0.1,
                "rho_GM" :   0.1,
                "coupling" : 0.15
                }

params_init = {
                "D_WM" :     0.1,
                "D_GM" :     0.1,
                "rho_WM" :   0.1,
                "rho_GM" :   0.1,
                "coupling" : 0.1
                }

bounds = [[0.01, 0.01, 0.01],
                  [2, 2, 3]]

opt_params = {'bounds' : bounds,
              #'method' : "L-BFGS-B",
              'method' : "TNC",
              'tol' : 1e-6,
              'options' : {'disp': True, 'gtol': 1e-6}}

seed_position = [148, -67]
image_slice   = 87


from glimslib import config


# image_slice   = 30
# base_dir = os.path.join(config.output_dir_testing, 'image_based_optimisation_IvyGap', 'W07', 'slice-%02d'%image_slice)
# os.makedirs(base_dir, exist_ok=True)
#
# data_path = os.path.join(config.test_data_dir, 'IvyGap', 'W07')
# path_patient_image = os.path.join(data_path, 'sub-W07_ses-1996-12-18_T1wPost_reg-rigid_skullstripped.mha')
# path_patient_seg = os.path.join(data_path, 'sub-W07_ses-1996-12-18_tumorseg_reg-rigid_seg-tumor_manual-T1c-T2.mha')
# path_atlas_image = os.path.join(config.test_data_dir, 'brain_atlas_image_t1_3d.mha')
# path_atlas_seg = os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')

image_slice   = 42
base_dir = os.path.join(config.output_dir_testing, 'image_based_optimisation_IvyGap', 'W11', 'slice-%02d'%image_slice)
os.makedirs(base_dir, exist_ok=True)

data_path = os.path.join(config.test_data_dir, 'IvyGap', 'W11')
path_patient_image = os.path.join(data_path, 'sub-W11_ses-1997-05-11_T1wPost_reg-rigid_skullstripped.mha')
path_patient_seg = os.path.join(data_path, 'sub-W11_ses-1997-05-11_tumorseg_reg-rigid_seg-tumor_manual-T1c-T2.mha')
path_atlas_image = os.path.join(config.test_data_dir, 'brain_atlas_image_t1_3d.mha')
path_atlas_seg = os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')


# data_path = os.path.join(config.test_data_dir, 'TCGA')
# path_patient_image = os.path.join(data_path, 'TCGA-06-0190_2004-12-10_t1Gd.mha')
# path_patient_seg = os.path.join(data_path, 'TCGA-06-0190_2004-12-10_GlistrBoost_ManuallyCorrected_T1-label-5_T2-label-6.mha')
# path_atlas_image = os.path.join(config.test_data_dir, 'brain_atlas_image_t1_3d.mha')
# path_atlas_seg = os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')




# #== 1)
# ibo1 = ImageBasedOptimization(base_dir,
#                              path_to_labels_patient_orig=path_patient_seg, path_to_image=path_patient_image,
#                              path_to_labels_atlas_orig=path_atlas_seg, path_to_image_atlas=path_atlas_image,
#                              image_z_slice=image_slice, plot=True)
# ibo1.create_patient_specific_reference_from_atlas()
# ibo1.extract_2d_domain(file_type='domain')
# ibo1.extract_2d_domain(file_type='patient')
# ibo1.extract_2d_domain(file_type='atlas')
#
# ibo1.reduce_2d_domain(file_type='domain')
# ibo1.reduce_2d_domain(file_type='patient')
# ibo1.reduce_2d_domain(file_type='atlas')
#
# ibo1.estimate_deformation_field_patient_reference()
# ibo1.reduce_2d_domain(file_type='patient_ref')
# ibo1.reduce_estimated_deformation_field()
# ibo1.create_thresholded_conc_fields_from_segmentation(T1_label=8, T2_label=9)
#
# ibo1.plot_all()
# #
#
# ibo2 = ImageBasedOptimization(base_dir)
# ibo2.reload_state()
# ibo2.init_forward_problem(seed_position=None, sim_params=params_sim,
#                          model_params_fixed=params_fix, model_params_varying=params_target, seed_from_com=True)
# ibo2.run_forward_sim()

ibo3 = ImageBasedOptimization(base_dir)
ibo3.reload_state()
ibo3.init_inverse_problem(seed_position=None, model_params_varying=params_init, sim_params=params_sim,
                             model_params_fixed=params_fix, path_to_domain=None, seed_from_com=True)
ibo3.run_inverse_problem_3params(opt_params)
#ibo3.run_inverse_problem_2params(opt_params)

#
# ibo4 = ImageBasedOptimization(base_dir)
# ibo4.init_optimized_problem()
# ibo4.run_optimized_sim()
