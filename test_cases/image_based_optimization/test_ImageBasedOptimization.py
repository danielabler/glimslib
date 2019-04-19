import os
import unittest
from test_cases.image_based_optimization.ImageBasedOptimization import ImageBasedOptimization
from glimslib import config


class Test_ImageBasedOptimization_atlas2d(unittest.TestCase):

    def setUp(self):
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

        self.seed_position = [148, -67]
        self.image_slice = 87

        self.base_dir = os.path.join(config.output_dir_testing, 'image_based_optimisation', 'testing', 'atlas_only_2d')
        self.path_to_labels_atlas = os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')
        self.path_to_image_atlas = os.path.join(config.test_data_dir, 'brain_atlas_image_t1_3d.mha')

    def test_01_init_domain(self):
        self.ibo = ImageBasedOptimization(self.base_dir,
                                          path_to_labels_atlas=self.path_to_labels_atlas,
                                          path_to_image_atlas=self.path_to_image_atlas,
                                          image_z_slice=self.image_slice, plot=False)

    def test_02_reload_state(self):
        self.ibo = ImageBasedOptimization(self.base_dir)
        self.ibo.reload_state()

    def test_03_prepare_domain(self):
        self.ibo = ImageBasedOptimization(self.base_dir)
        self.ibo.reload_state()
        self.ibo.prepare_domain(plot=True)

    def test_04_forward_sim(self):
        self.ibo = ImageBasedOptimization(self.base_dir)
        self.ibo.reload_state()
        self.ibo.init_forward_problem(seed_position=self.seed_position,
                                      sim_params=self.params_sim,
                                      model_params_fixed=self.params_fix,
                                      model_params_varying=self.params_target)
        self.ibo.run_forward_sim(plot=True)

    def test_05_get_target_fields(self):
        self.ibo = ImageBasedOptimization(self.base_dir)
        self.ibo.reload_state()
        self.ibo.get_target_fields(plot=True)




class Test_ImageBasedOptimization_patient2d(unittest.TestCase):

    def setUp(self):
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

        self.seed_position = [148, -67]
        self.image_slice = 87

        self.base_dir = os.path.join(config.output_dir_testing, 'image_based_optimisation', 'testing', 'patient_2d')
        self.path_to_labels_atlas = os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')
        self.path_to_image_atlas = os.path.join(config.test_data_dir, 'brain_atlas_image_t1_3d.mha')
        self.path_to_image_patient = os.path.join(config.test_data_dir, 'TCGA', 'TCGA-06-0190_2004-12-10_t1Gd.mha')
        self.path_to_labels_patient = os.path.join(config.test_data_dir, 'TCGA',
                                                   'TCGA-06-0190_2004-12-10_GlistrBoost_ManuallyCorrected_T1-label-5_T2-label-6.mha')


    def test_01_init_domain(self):
        self.ibo = ImageBasedOptimization(self.base_dir,
                                          path_to_labels_atlas=self.path_to_labels_atlas,
                                          path_to_image_atlas=self.path_to_image_atlas,
                                          path_to_image_patient=self.path_to_image_patient,
                                          path_to_labels_patient=self.path_to_labels_patient,
                                          image_z_slice=self.image_slice, plot=False)

    def test_02_reload_state(self):
        self.ibo = ImageBasedOptimization(self.base_dir)
        self.ibo.reload_state()

    def test_03_prepare_domain(self):
        self.ibo = ImageBasedOptimization(self.base_dir)
        self.ibo.reload_state()
        self.ibo.prepare_domain(plot=True)

    def test_05_get_target_fields(self):
        self.ibo = ImageBasedOptimization(self.base_dir)
        self.ibo.reload_state()
        self.ibo.get_target_fields(plot=True)
