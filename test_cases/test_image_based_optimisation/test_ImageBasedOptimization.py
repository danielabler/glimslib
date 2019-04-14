import os
from test_cases.test_image_based_optimisation.ImageBasedOptimization import ImageBasedOptimization

## PARAMS

params_sim = {
                "sim_time" : 200,
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
                "D_WM" :     0.001,
                "D_GM" :     0.001,
                "rho_WM" :   0.001,
                "rho_GM" :   0.001,
                "coupling" : 0.001
                }

bounds = [[0.001, 0.001, 0.001, 0.001, 0.001],
                  [0.5, 0.5, 0.5, 0.5, 0.5]]

opt_params = {'bounds' : bounds,
              'method' : "L-BFGS-B",
              'tol' : 1e-6,
              'options' : {'disp': True, 'gtol': 1e-6}}

seed_position = [148, -67]
image_slice   = 87


import config
base_dir = os.path.join(config.output_dir_testing, 'image_based_optimisation', 'from_class', 'steps_200')
path_to_labels = os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')
path_to_image = os.path.join(config.test_data_dir, 'brain_atlas_image_t1_3d.mha')



#== 1)
ibo1 = ImageBasedOptimization(base_dir,
                             path_to_labels=path_to_labels, path_to_image=path_to_image,
                             image_z_slice=image_slice, plot=False)
ibo1.extract_2d_domain()
ibo1.reduce_2d_domain()
ibo1.init_forward_problem(seed_position=seed_position, sim_params=params_sim,
                         model_params_fixed=params_fix, model_params_varying=params_target)
ibo1.run_forward_sim()
ibo1.create_deformed_image()
ibo1.reconstruct_deformation_field()
ibo1.compare_displacement_field_simulated_registered()
ibo1.create_thresholded_conc_fields()

#== 2)
# ibo2 = ImageBasedOptimization(base_dir)
# ibo2.reload_state()
# ibo2.init_inverse_problem(seed_position=seed_position, sim_params=params_sim,
#                           model_params_varying=params_init)
# ibo2.run_inverse_problem()
# ibo2.init_optimized_problem()
# ibo2.run_optimized_sim()
