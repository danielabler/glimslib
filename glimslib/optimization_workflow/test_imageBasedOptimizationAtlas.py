import os
from unittest import TestCase

from glimslib import config
from glimslib.optimization_workflow.image_based_optimization_atlas import ImageBasedOptimizationAtlas


class TestImageBasedOptimizationAtlas(TestCase):

    def setUp(self):
        self.base_dir_2d = os.path.join(config.output_dir_testing, 'ImageBasedOptimizationAtlas_2d')
        self.base_dir_3d = os.path.join(config.output_dir_testing, 'ImageBasedOptimizationAtlas_3d')

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

        self.bounds = [[0.001, 0.001, 0.001],
                       [0.5, 0.5, 0.5]]

        # large tolerance for quick tests
        self.opt_params = {'bounds': self.bounds,
                           'method': "L-BFGS-B",
                           'tol': 1,
                           'options': {'disp': True, 'gtol': 1}}

        self.seed_position_2d = [148, -67]
        self.seed_position_3d = [118, -109, 72]

    def test_01_init_domain(self):
        self.ibo = ImageBasedOptimizationAtlas(self.base_dir_2d,
                                               path_to_labels_atlas=self.path_to_labels_atlas,
                                               path_to_image_atlas=self.path_to_image_atlas,
                                               image_z_slice=self.image_slice, plot=False)
        self.ibo2 = ImageBasedOptimizationAtlas(self.base_dir_2d)
        self.assertEqual(self.ibo.path_to_image_atlas_orig, self.ibo2.path_to_image_atlas_orig)
        self.assertEqual(self.ibo.dim, self.ibo2.dim)

    def test_02a_prepare_domain_2d(self):
        self.ibo = ImageBasedOptimizationAtlas(self.base_dir_2d,
                                               path_to_labels_atlas=self.path_to_labels_atlas,
                                               path_to_image_atlas=self.path_to_image_atlas,
                                               image_z_slice=self.image_slice, plot=False)
        self.ibo.prepare_domain(plot=True)

    #
    # def test_02b_prepare_domain_3d(self):
    #     self.ibo = ImageBasedOptimizationAtlas(self.base_dir_3d,
    #                                            path_to_labels_atlas=self.path_to_labels_atlas,
    #                                            path_to_image_atlas=self.path_to_image_atlas,
    #                                            image_z_slice=None, plot=False)
    #     self.ibo.prepare_domain(plot=True)

    def test_03a_run_forward_sim_2d(self):
        self.ibo = ImageBasedOptimizationAtlas(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.init_forward_problem(seed_position=self.seed_position_2d,
                                      sim_params=self.params_sim,
                                      model_params_fixed=self.params_fix,
                                      model_params_varying=self.params_target)
        self.ibo.run_forward_sim(plot=True)

    # def test_03b_run_forward_sim_3d(self):
    #     self.ibo = ImageBasedOptimizationAtlas(self.base_dir_3d)
    #     self.ibo.reload_state()
    #     self.ibo.init_forward_problem(seed_position=self.seed_position_3d,
    #                                   sim_params=self.params_sim,
    #                                   model_params_fixed=self.params_fix,
    #                                   model_params_varying=self.params_target)
    #     self.ibo.run_forward_sim(plot=True)

    def test_04a_create_target_fields_2d(self):
        self.ibo = ImageBasedOptimizationAtlas(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.create_target_fields()

    def test_05a_plotting_2d(self):
        self.ibo = ImageBasedOptimizationAtlas(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.plot_all()

    def test_06a_run_inverse_problem_2d(self):
        self.ibo = ImageBasedOptimizationAtlas(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.init_inverse_problem(seed_position=self.seed_position_2d, sim_params=self.params_sim,
                                      model_params_varying=self.params_init, optimization_type=3)
        self.ibo.run_inverse_problem(self.opt_params)

    def test_07a_run_optimized_sim_2d(self):
        self.ibo = ImageBasedOptimizationAtlas(self.base_dir_2d)
        self.ibo.reload_state()
        print(self.ibo.params_inverse)
        self.ibo.init_optimized_problem()
        self.ibo.run_optimized_sim()

    def test_08_compare_original_optimized(self):
        self.ibo = ImageBasedOptimizationAtlas(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.compare_original_optimized(plot=True)

    def test_09_postprocess(self):
        self.ibo = ImageBasedOptimizationAtlas(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.post_process()
        self.ibo.write_analysis_summary()
