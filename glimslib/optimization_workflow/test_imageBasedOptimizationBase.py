import os
from unittest import TestCase

from glimslib import config
from glimslib.optimization_workflow.image_based_optimization import ImageBasedOptimizationBase


class TestImageBasedOptimizationBase(TestCase):

    def setUp(self):
        self.base_dir_2d = os.path.join(config.output_dir_testing, 'ImageBasedOptimizationBase_2d')
        self.base_dir_3d = os.path.join(config.output_dir_testing, 'ImageBasedOptimizationBase_3d')

        self.path_to_labels_atlas = os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')
        self.path_to_image_atlas = os.path.join(config.test_data_dir, 'brain_atlas_image_t1_3d.mha')
        self.image_slice = 87

        self.params_sim = {
            "sim_time": 10,
            "sim_time_step": 1,
        }

        self.params_fix = {"E_GM": 3000E-6,
                           "E_WM": 3000E-6,
                           "E_CSF": 1000E-6,
                           "E_VENT": 1000E-6,
                           "nu_GM": 0.45,
                           "nu_WM": 0.45,
                           "nu_CSF": 0.45,
                           "nu_VENT": 0.3
                           }

        self.params_target = {
            "D_WM": 0.1,
            "D_GM": 0.02,
            "rho_WM": 0.1,
            "rho_GM": 0.1,
            "coupling": 0.15
        }

        self.params_init = {
            "D_WM": 0.001,
            "D_GM": 0.001,
            "rho_WM": 0.001,
            "rho_GM": 0.001,
            "coupling": 0.001
        }

        self.bounds = [[0.001, 0.001, 0.001, 0.001, 0.001],
                       [0.5, 0.5, 0.5, 0.5, 0.5]]

        self.opt_params = {'bounds': self.bounds,
                           'method': "L-BFGS-B",
                           'tol': 1e-6,
                           'options': {'disp': True, 'gtol': 1e-6}}

        self.seed_position_2d = [148, -67]
        self.seed_position_3d = [118, -109, 72]

    def test_01_init_domain(self):
        self.ibo = ImageBasedOptimizationBase(self.base_dir_2d,
                                              path_to_labels_atlas=self.path_to_labels_atlas,
                                              path_to_image_atlas=self.path_to_image_atlas,
                                              image_z_slice=self.image_slice, plot=False)
        self.ibo2 = ImageBasedOptimizationBase(self.base_dir_2d)
        self.assertEqual(self.ibo.path_to_image_atlas_orig, self.ibo2.path_to_image_atlas_orig)
        self.assertEqual(self.ibo.dim, self.ibo2.dim)

    def test_02_reload_state(self):
        self.ibo = ImageBasedOptimizationBase(self.base_dir_2d)
        self.ibo.reload_state()

    def test_03a_mesh_domain_2d(self):
        self.ibo = ImageBasedOptimizationBase(self.base_dir_2d,
                                              path_to_labels_atlas=self.path_to_labels_atlas,
                                              path_to_image_atlas=self.path_to_image_atlas,
                                              image_z_slice=self.image_slice, plot=False)
        self.ibo.path_to_domain_image_3d = self.ibo.path_to_image_atlas_orig
        self.ibo.path_to_domain_labels_3d = self.ibo.path_to_labels_atlas_orig
        self.ibo.mesh_domain()

    def test_03b_mesh_domain_3d(self):
        self.ibo = ImageBasedOptimizationBase(self.base_dir_3d,
                                              path_to_labels_atlas=self.path_to_labels_atlas,
                                              path_to_image_atlas=self.path_to_image_atlas,
                                              image_z_slice=None, plot=False)
        self.ibo.path_to_domain_image_3d = self.ibo.path_to_image_atlas_orig
        self.ibo.path_to_domain_labels_3d = self.ibo.path_to_labels_atlas_orig
        self.ibo.mesh_domain()

    def test_04a_run_forward_sim_2d(self):
        self.ibo = ImageBasedOptimizationBase(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.init_forward_problem(seed_position=self.seed_position_2d,
                                      sim_params=self.params_sim,
                                      model_params_fixed=self.params_fix,
                                      model_params_varying=self.params_target)
        self.ibo.run_forward_sim(plot=True)

    # def test_04b_run_forward_sim_3d(self):
    #     self.ibo = ImageBasedOptimizationBase(self.base_dir_3d)
    #     self.ibo.reload_state()
    #     self.ibo.init_forward_problem(seed_position=self.seed_position_3d,
    #                                   sim_params=self.params_sim,
    #                                   model_params_fixed=self.params_fix,
    #                                   model_params_varying=self.params_target)
    #     self.ibo.run_forward_sim(plot=True)

    def test_05a_create_deformed_image_2d(self):
        self.ibo = ImageBasedOptimizationBase(self.base_dir_2d)
        self.ibo.reload_state()

        path_image_warped = self.ibo.data.create_image_path(processing=self.ibo.steps_sub_path_map['target_fields'],
                                                            datasource='sim', frame='deformed', content='T1')
        self.ibo._create_deformed_image(path_image_ref=self.ibo.path_to_domain_image_2d,
                                        output_path=self.ibo.path_target_fields,
                                        path_image_warped=path_image_warped)

    def test_reconstruct_deformation_field_2d(self):
        self.ibo = ImageBasedOptimizationBase(self.base_dir_2d)
        self.ibo.reload_state()
        path_image_warped = self.ibo.data.create_image_path(processing=self.ibo.steps_sub_path_map['target_fields'],
                                                            datasource='sim', frame='deformed', content='T1')
        path_to_warp_field = os.path.join(self.ibo.path_target_fields,
                                          'registered_image_deformed_to_reference_warp.nii.gz')

        self.ibo._reconstruct_deformation_field(path_to_reference_image=self.ibo.path_to_domain_image_2d,
                                                path_to_deformed_image=path_image_warped,
                                                path_to_warp_field=path_to_warp_field,
                                                path_to_reduced_domain=self.ibo.path_to_domain_meshfct_main, plot=True)

    def test_create_thresholded_conc_fields(self):
        self.ibo = ImageBasedOptimizationBase(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.create_thresholded_conc_fields(path_conc_field=self.ibo.path_forward_conc,
                                                path_to_reduced_domain=self.ibo.path_to_domain_meshfct_main,
                                                plot=True)

    def test_run_inverse_sim(self):
        self.fail()

    def test_run_optimized_sim(self):
        self.fail()
