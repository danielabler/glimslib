import os
from unittest import TestCase

from glimslib import config
from glimslib.optimization_workflow.image_based_optimization_patient import ImageBasedOptimizationPatient


class TestImageBasedOptimizationPatient(TestCase):

    def setUp(self):
        self.base_dir_2d = os.path.join(config.output_dir_testing, 'ImageBasedOptimizationPatient_2d')
        self.base_dir_3d = os.path.join(config.output_dir_testing, 'ImageBasedOptimizationPatient_3d')

        self.path_to_labels_atlas = os.path.join(config.test_data_dir, 'brain_atlas_image_3d.mha')
        self.path_to_image_atlas = os.path.join(config.test_data_dir, 'brain_atlas_image_t1_3d.mha')

        data_path = os.path.join(config.test_data_dir, 'TCGA')
        self.path_to_patient_image = os.path.join(data_path, 'TCGA-06-0190_2004-12-10_t1Gd.mha')
        self.path_to_patient_seg = os.path.join(data_path,
                                                'TCGA-06-0190_2004-12-10_GlistrBoost_ManuallyCorrected_T1-label-5_T2-label-6.mha')
        self.image_slice = 87


        data_path = os.path.join(config.test_data_dir, 'IvyGap', 'W07')
        self.path_to_patient_image = os.path.join(data_path, 'sub-W07_ses-1996-12-18_T1wPost_reg-rigid_skullstripped.mha')
        self.path_to_patient_seg = os.path.join(data_path,
                                                'sub-W07_ses-1996-12-18_tumorseg_reg-rigid_seg-tumor_manual-T1c-T2.mha')

        self.image_slice = 30


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

    def test_01_init_domain(self):
        self.ibo = ImageBasedOptimizationPatient(self.base_dir_2d,
                                                 path_to_labels_atlas=self.path_to_labels_atlas,
                                                 path_to_image_atlas=self.path_to_image_atlas,
                                                 path_to_labels_patient=self.path_to_patient_seg,
                                                 path_to_image_patient=self.path_to_patient_image,
                                                 image_z_slice=self.image_slice, plot=False)
        self.ibo2 = ImageBasedOptimizationPatient(self.base_dir_2d)
        self.ibo2.reload_state()
        self.assertEqual(self.ibo.path_to_image_atlas_orig, self.ibo2.path_to_image_atlas_orig)
        self.assertEqual(self.ibo.dim, self.ibo2.dim)

    def test_02a_prepare_domain_2d(self):
        self.ibo = ImageBasedOptimizationPatient(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.prepare_domain(plot=True)

    def test_02b_prepare_domain_3d(self):
        self.ibo = ImageBasedOptimizationPatient(self.base_dir_3d,
                                                 path_to_labels_atlas=self.path_to_labels_atlas,
                                                 path_to_image_atlas=self.path_to_image_atlas,
                                                 path_to_labels_patient=self.path_to_patient_seg,
                                                 path_to_image_patient=self.path_to_patient_image,
                                                 image_z_slice=None, plot=False)
        self.ibo.reload_state()
        self.ibo.prepare_domain(plot=True)

    def test_03a_create_target_fields_2d(self):
        self.ibo = ImageBasedOptimizationPatient(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.create_target_fields(T1_label=8, T2_label=9)

    def test_04a_plotting_2d(self):
        self.ibo = ImageBasedOptimizationPatient(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.plot_all()

    def test_06a_run_inverse_problem_2d(self):
        self.ibo = ImageBasedOptimizationPatient(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.init_inverse_problem(seed_position=None, sim_params=self.params_sim,
                                      model_params_varying=self.params_init,
                                      model_params_fixed=self.params_fix,
                                      seed_from_com=True,
                                      optimization_type=3
                                      )
        self.ibo.run_inverse_problem(self.opt_params)

    def test_07a_run_optimized_sim_2d(self):
        self.ibo = ImageBasedOptimizationPatient(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.init_optimized_problem()
        self.ibo.run_optimized_sim(plot=True)

    def test_09_postprocess(self):
        self.ibo = ImageBasedOptimizationPatient(self.base_dir_2d)
        self.ibo.reload_state()
        self.ibo.post_process()
        self.ibo.write_analysis_summary()