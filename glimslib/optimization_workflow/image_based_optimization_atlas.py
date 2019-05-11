import os

from glimslib import config

config.USE_ADJOINT = True

from glimslib import fenics_local as fenics
from glimslib.optimization_workflow.image_based_optimization import ImageBasedOptimizationBase
import glimslib.utils.data_io as dio
from glimslib.visualisation import plotting as plott


class ImageBasedOptimizationAtlas(ImageBasedOptimizationBase):

    def prepare_domain(self, plot=True):
        self.path_to_domain_image_3d = self.path_to_image_atlas_orig
        self.path_to_domain_labels_3d = self.path_to_labels_atlas_orig
        self.mesh_domain(plot=plot)

    def create_target_fields(self):
        # Deformation

        path_image_warped = self.data.create_image_path(processing=self.steps_sub_path_map['target_fields'],
                                                        datasource='sim', frame='deformed', content='T1')
        path_to_warp_field = os.path.join(self.path_target_fields,
                                          'registered_image_deformed_to_reference_warp.nii.gz')
        self._create_deformed_image(path_image_ref=self.path_to_domain_image_main,
                                    output_path=self.path_target_fields,
                                    path_image_warped=path_image_warped)

        self.path_forward_disp_reconstructed = self.data.create_fenics_path(
            processing=self.steps_sub_path_map['target_fields'],
            datasource='registration',
            content='disp', frame='reference')

        self._reconstruct_deformation_field(path_to_reference_image=self.path_to_domain_image_main,
                                            path_to_deformed_image=path_image_warped,
                                            path_to_warp_field=path_to_warp_field,
                                            path_to_reduced_domain=self.path_to_domain_meshfct_main, plot=True)
        # Concentration
        self.create_thresholded_conc_fields(path_conc_field=self.path_forward_conc,
                                            path_to_reduced_domain=None, plot=True)
        self._save_state()

    def compare_displacement_field_simulated_registered(self, plot=None):
        if plot is None:
            plot = self.plot

        disp_sim, mesh_sim, subdomains_sim, boundaries_sim = dio.load_function_mesh(self.path_forward_disp,
                                                                                    functionspace='vector')
        disp_est, mesh_est, subdomains_est, boundaries_est = dio.load_function_mesh(
            self.path_forward_disp_reconstructed,
            functionspace='vector')
        # -- chose simulation mesh as reference
        funspace_ref = disp_sim.function_space()
        # -- project/interpolate estimated displacement field over that mesh
        disp_est_ref = self.interpolate_non_matching(disp_est, funspace_ref)

        # compute errornorm
        error = fenics.errornorm(disp_sim, disp_est_ref)
        self.measures['errornorm_displacement_simulated_vs_registered'] = error

        # compute difference field
        disp_diff = fenics.project(disp_sim - disp_est_ref, funspace_ref, annotate=False)

        if plot:
            plott.show_img_seg_f(function=disp_sim, show=False,
                                 path=os.path.join(self.steps_path_map['plots'],
                                                   'displacement_field_from_simulation.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=disp_est_ref, show=False,
                                 path=os.path.join(self.steps_path_map['plots'],
                                                   'displacement_field_from_registration_ref_space.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=disp_diff, show=False,
                                 path=os.path.join(self.steps_path_map['plots'], 'displacement_field_difference.png'),
                                 dpi=300)
        self._save_state()

    def compare_original_optimized(self, plot=None):
        if plot is None:
            plot = self.plot

        output_path = self.steps_path_map['comparison']
        self.sim_forward.init_postprocess(output_path)
        self.sim_optimized.init_postprocess(output_path)
        # -- get solution at sim time from original forward model
        conc_orig_full = self.sim_forward.postprocess.get_solution_concentration(
            self.params_forward['sim_params']['sim_time']).copy()
        disp_orig_full = self.sim_forward.postprocess.get_solution_displacement(
            self.params_forward['sim_params']['sim_time']).copy()

        # -- get solution at sim time from optimized simulation
        conc_opt = self.sim_optimized.postprocess.get_solution_concentration(
            self.params_optimized['sim_params']['sim_time']).copy()
        disp_opt = self.sim_optimized.postprocess.get_solution_displacement(
            self.params_optimized['sim_params']['sim_time']).copy()

        # -- project original solution into domain of optimized solution
        # -- chose simulation mesh as reference
        funspace_disp_opt = self.sim_optimized.functionspace.get_functionspace(subspace_id=0)
        funspace_conc_opt = self.sim_optimized.functionspace.get_functionspace(subspace_id=1)
        conc_orig = self.interpolate_non_matching(conc_orig_full, funspace_conc_opt)
        disp_orig = self.interpolate_non_matching(disp_orig_full, funspace_disp_opt)

        # -- compute error norms
        error_conc = fenics.errornorm(conc_orig, conc_opt)
        error_disp = fenics.errornorm(disp_orig, disp_opt)

        self.measures['errornorm_displacement_forward_vs_optimized'] = error_disp
        self.measures['errornorm_concentration_forward_vs_optimized'] = error_conc

        if plot:
            plott.show_img_seg_f(function=conc_orig, path=os.path.join(output_path, 'conc_forward.png'))
            plott.show_img_seg_f(function=conc_opt, path=os.path.join(output_path, 'conc_opt.png'))
            conc_diff = fenics.project(conc_orig - conc_opt, funspace_conc_opt, annotate=False)
            plott.show_img_seg_f(function=conc_diff, path=os.path.join(output_path, 'conc_diff.png'))

            plott.show_img_seg_f(function=disp_orig, path=os.path.join(output_path, 'disp_forward.png'))
            plott.show_img_seg_f(function=disp_opt, path=os.path.join(output_path, 'disp_opt.png'))
            disp_diff = fenics.project(disp_orig - disp_opt, funspace_disp_opt, annotate=False)
            plott.show_img_seg_f(function=disp_diff, path=os.path.join(output_path, 'disp_diff.png'))

        self._save_state()

    def plot_all(self):
        super().plot_all()
        self.compare_displacement_field_simulated_registered(plot=True)

    def compute_param_rel_errors(self):
        param_rel_error_dict = {}
        for param in self.params_forward['model_params_varying'].keys():
            param_value_forward = self.params_forward['model_params_varying'][param]
            param_value_optimized = self.params_optimized['model_params_varying'][param]
            param_rel_error = (param_value_optimized - param_value_forward) / param_value_forward
            param_rel_error_dict['relative_error_' + param] = param_rel_error
        return param_rel_error_dict

    def write_analysis_summary(self, add_info_list=[]):
        add_info_list = [self.compute_param_rel_errors(),
                         self.flatten_params(self.params_forward, 'forward')]
        super().write_analysis_summary(add_info_list=add_info_list)

    def compute_com_all(self, conc_dict=None):
        conc_dict = {'forward': self.path_forward_conc}
        super().compute_com_all(conc_dict=conc_dict)

    def post_process(self):
        sim_list = ['forward', 'optimized']
        threshold_list = [self.conc_threshold_levels['T2'], self.conc_threshold_levels['T2']]
        super().post_process(sim_list=sim_list, threshold_list=threshold_list)
