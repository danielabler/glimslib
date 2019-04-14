import logging
import os
import pickle
from datetime import datetime
from itertools import product

import pandas as pd
import SimpleITK as sitk
from scipy.optimize import minimize as scipy_minimize

import config
config.USE_ADJOINT = True
import fenics_local as fenics

from simulation.simulation_tumor_growth_brain_quad import TumorGrowthBrain
from simulation.helpers.helper_classes import Boundary
import visualisation.plotting as plott
import utils.file_utils as fu
import utils.data_io as dio
import utils.vtk_utils as vtu
import utils.image_registration_utils as reg
import numpy as np

from ufl import tanh




if fenics.is_version("<2018.1.x"):
    fenics.set_log_level(fenics.INFO)
else:
    fenics.set_log_level(fenics.LogLevel.INFO)


def compute_com(fenics_scalar_field, domain=None):
    dim_geo = fenics_scalar_field.function_space().mesh().geometry().dim()
    com = []
    if domain is None:
        domain = fenics.dx
    volume = fenics.assemble(fenics_scalar_field * domain)
    for dim in range(dim_geo):
        coord_expr = fenics.Expression('x[%i]'%dim, degree=1)
        coord_int  = fenics.assemble(coord_expr * fenics_scalar_field * domain)
        if volume > 0:
            coord = coord_int / volume
        else:
            coord = np.nan
        com.append(coord)
    return com

def interpolate_non_matching(source_function, target_funspace):
    function_new = fenics.Function(target_funspace)
    fenics.LagrangeInterpolator.interpolate(function_new, source_function)
    return function_new

def thresh(f, thresh):
    smooth_f = 0.01
    f_thresh = 0.5 * (tanh((f - thresh) / smooth_f) + 1)
    return f_thresh

def eval_cb_post(j, a):
    values = [param.values()[0] for param in a]
    result = (j, *values)
    global opt_param_progress_post
    opt_param_progress_post.append(result)
    global opt_date_time
    opt_date_time.append((j, datetime.now()))
    print(result)

def derivative_cb_post(j, dj, m):
    param_values = [param.values()[0] for param in m]
    dj_values = [param.values()[0] for param in dj]
    global opt_dj_progress_post
    result = (j, *dj_values)
    opt_dj_progress_post.append(result)
    print(result)


def create_opt_progress_df(opt_param_list, opt_dj_list, param_name_list, datetime_list):
    columns_params = ['J', *param_name_list]
    columns_dJ = ['J', *['dJd%s'%param for param in param_name_list]]
    columns_datetime = ['J', 'datetime']
    # create data frames
    params_df = pd.DataFrame(opt_param_list)
    params_df.columns = columns_params
    datetime_df = pd.DataFrame(datetime_list)
    datetime_df.columns = columns_datetime
    opt_df_ = pd.merge(params_df, datetime_df, on='J', how='outer')
    try:
        dj_df = pd.DataFrame(opt_dj_list)
        dj_df.columns = columns_dJ
        opt_df = pd.merge(opt_df_, dj_df, on='J', how='outer')
    except:
        opt_df = opt_df_
    return opt_df

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def get_path_relative_to_subdir(abs_path, reference_subdir):
    split_path = splitall(abs_path)
    index = split_path.index(reference_subdir)
    path_relative = os.path.join(*split_path[index:])
    return path_relative



class ImageBasedOptimization():

    def __init__(self, base_dir, path_to_labels=None, path_to_image=None, image_z_slice=None, plot=False):
        # logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # -- stream
        self.logging_stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logging_stream_handler.setFormatter(formatter)
        self.logger.addHandler(self.logging_stream_handler)
        # -- file
        self.logging_file_handler = logging.FileHandler(os.path.join(base_dir,
                                 datetime.now().strftime("logger_ImagedBasedOpt_%Y-%m-%d_%H-%M-%S")+'.log' ))
        self.logging_file_handler.setFormatter(formatter)
        self.logger.addHandler(self.logging_file_handler)
        # -- add file handler to other loggers
        logger_names = ['FFC', 'UFL', 'dijitso', 'flufl', 'instant']
        for logger_name in logger_names:
            try:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.INFO)
                logger.addHandler(self.logging_file_handler)
            except:
                pass

        # paths
        self.base_dir = base_dir
        self.path_to_base_params = os.path.join(self.base_dir, "base_params.pkl")
        if (path_to_labels is None) or (path_to_image is None) or (image_z_slice is None):
            if os.path.exists(self.path_to_base_params):
                base_params = pickle.load(open(self.path_to_base_params, "rb"))
                self.path_to_labels = base_params["path_to_labels"]
                self.path_to_t1 = base_params["path_to_t1"]
                self.image_z_slice = base_params["image_z_slice"]
                self.plot = base_params["plot"]
            else:
                self.logger.error("Cannot initialize")
        else:
            self.path_to_labels = path_to_labels
            self.path_to_t1 = path_to_image
            self.image_z_slice = image_z_slice
            self.image_z_slice = image_z_slice
            self.plot = plot
            base_params = {"path_to_labels" : self.path_to_labels,
                           "path_to_t1" : self.path_to_t1,
                           "image_z_slice" : self.image_z_slice,
                           "plot" : self.plot}
            fu.ensure_dir_exists(self.base_dir)
            with open(self.path_to_base_params, 'wb') as handle:
                pickle.dump(base_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._init_defaults()
        self.logger.info("Initialized ImageBasedOptimization instance")

    # def __del__(self):
    #     self.logger.info("Deleting instance of ImageBasedOptimization")

    def _init_defaults(self):
        # domain preparation
        self.path_domain_prep = os.path.join(self.base_dir, '00_domain_preparation')
        self.path_domain_image_2d = os.path.join(self.path_domain_prep, 'domain_image_2d.mha')
        self.path_domain_labels_2d = os.path.join(self.path_domain_prep, 'domain_labels_2d.mha')
        self.path_domain_image_function_2d = os.path.join(self.path_domain_prep, 'domain_image_function_2d.h5')
        self.path_domain_labels_function_2d = os.path.join(self.path_domain_prep, 'domain_labels_function_2d.h5')
        self.path_domain_2d = os.path.join(self.path_domain_prep, 'domain_mesh_2d.h5')
        self.path_domain_2d_reduced = os.path.join(self.path_domain_prep, 'domain_mesh_2d_reduced.h5')
        fu.ensure_dir_exists(self.path_domain_prep)
        # forward simulation
        self.path_forward_sim = os.path.join(self.base_dir, '01_forward_simulation')
        self.path_forward_conc = os.path.join(self.path_forward_sim, 'forward_concentration_final.h5')
        self.path_forward_disp = os.path.join(self.path_forward_sim, 'forward_displacement_final.h5')
        self.path_run_params_forward = os.path.join(self.path_forward_sim, 'parameters_forward.pkl')
        fu.ensure_dir_exists(self.path_forward_sim)
        # manipulate simulation outputs for target fields
        self.path_target_fields = os.path.join(self.base_dir, '02_target_fields')
        self.path_target_conc   = os.path.join(self.path_target_fields, 'concentration')
        self.path_target_disp = os.path.join(self.path_target_fields, 'displacement')
        self.path_t1_2d_deformed = os.path.join(self.path_target_disp, 'T1_img_warped.nii')
        self.path_labels_2d_deformed = os.path.join(self.path_target_disp, 'labels_img_warped.nii')
        self.path_forward_disp_reconstructed = os.path.join(self.path_target_disp, 'displacement_from_image.h5')
        self.path_forward_disp_reduced = os.path.join(self.path_target_disp, 'forward_displacement_final_reduced.h5')
        self.path_forward_disp_reconstructed_reduced = os.path.join(self.path_target_disp, 'displacement_from_image_reduced.h5')
        self.path_forward_conc_threshold_02 = os.path.join(self.path_target_conc, 'concentration_thresholded_02.h5')
        self.path_forward_conc_threshold_08 = os.path.join(self.path_target_conc, 'concentration_thresholded_08.h5')
        self.path_forward_conc_threshold_02_reduced = os.path.join(self.path_target_conc, 'concentration_thresholded_02_reduced.h5')
        self.path_forward_conc_threshold_08_reduced = os.path.join(self.path_target_conc, 'concentration_thresholded_08_reduced.h5')
        fu.ensure_dir_exists(self.path_target_fields)
        # inverse simulaton
        self.path_inverse_sim = os.path.join(self.base_dir, '03_inverse_simulation')
        self.path_parameters_optimized = os.path.join(self.path_inverse_sim, 'optimized_parameters.pkl')
        self.path_optimization_progress_xls = os.path.join(self.path_inverse_sim, 'optimization_progress.xls')
        self.path_optimization_progress_pkl = os.path.join(self.path_inverse_sim, 'optimization_progress.pkl')
        self.path_run_params_inverse = os.path.join(self.path_inverse_sim, 'parameters_inverse.pkl')
        self.path_optimization_params = os.path.join(self.path_inverse_sim, 'parameters_optimization.pkl')
        fu.ensure_dir_exists(self.path_inverse_sim)
        # optimized simulation
        self.path_optimized_sim = os.path.join(self.base_dir, '04_optimized_simulation')
        self.path_optimized_conc_reduced = os.path.join(self.path_forward_sim, 'optimized_concentration_final_reduced.h5')
        self.path_optimized_disp_reduced = os.path.join(self.path_forward_sim, 'optimized_displacement_final_reduced.h5')
        self.path_run_params_optimized = os.path.join(self.path_optimized_sim, 'parameters_optimized.pkl')
        fu.ensure_dir_exists(self.path_optimized_sim)
        # comparison forward vs optimzied
        self.path_comparison = os.path.join(self.base_dir, '05_comparison_forward_optimized')
        self.path_comparison = os.path.join(self.base_dir, '05_comparison_forward_optimized')
        fu.ensure_dir_exists(self.path_comparison)
        # measures computed
        self.measures = {}
        self.path_to_measures = os.path.join(self.path_comparison, 'errornorms.pkl')
        # summary
        self.path_to_summary = os.path.join(self.base_dir, 'summary.pkl')

    def rebase(self, old_path):
        # TODO this is specific to use of this class with parametric study
        new_base = self.base_dir
        p_common = os.path.commonpath([new_base, old_path])
        if p_common != new_base:
            new_base_split = splitall(new_base)
            ref_subdir = new_base_split[-1]
            rel_path = get_path_relative_to_subdir(old_path, ref_subdir)
            new_base_reduced = os.path.join(*new_base_split[:-1])
            new_path = os.path.join(new_base_reduced, rel_path)
        else:  # nothing to do
            new_path = old_path
        return new_path

    def reload_model_sim(self, problem_type='forward'):
        self.logger.info("=== Reloading simulation '%s'."%problem_type)
        param_attr_name = "params_%s" % problem_type
        if hasattr(self, param_attr_name):
            if problem_type == 'forward':
                data_path = os.path.join(self.path_forward_sim, 'solution_timeseries.h5')
            elif problem_type == 'inverse':
                data_path = os.path.join(self.path_inverse_sim, 'solution_timeseries.h5')
            elif problem_type == 'optimized':
                data_path = os.path.join(self.path_optimized_sim, 'solution_timeseries.h5')
            else:
                self.logger.error("Non existing 'problem type'")

            if os.path.exists(data_path):
                params = getattr(self, param_attr_name)
                sim = self._init_problem(params['path_to_domain'], params['seed_position'], params['sim_params'],
                                         params['model_params_varying'], params['model_params_fixed'],
                                         problem_type=problem_type)
                sim.reload_from_hdf5(data_path)
                setattr(self, "sim_%s"%problem_type, sim)

        else:
            self.logger.warning("Parameter attribute for simulation '%s' does not exist"%problem_type)
            self.logger.warning("Cannot reload simulation")

    def reload_forward_sim(self):
        self.read_problem_run_params(problem_type='forward')
        self.reload_model_sim(problem_type='forward')

    def reload_inverse_sim(self):
        self.read_problem_run_params(problem_type='inverse')
        self.reload_model_sim(problem_type='inverse')
        # parameters of optimization process
        if os.path.exists(self.path_optimization_params):
            self.params_optimization = pickle.load(open(self.path_optimization_params, "rb"))
        # parameter results from optimization process
        if os.path.exists(self.path_parameters_optimized):
            self.model_params_optimized = pickle.load(open(self.path_parameters_optimized, "rb"))

    def reload_optimized_sim(self):
        self.read_problem_run_params(problem_type='optimized')
        self.reload_model_sim(problem_type='optimized')

    def reload_state(self):
        self.logger.info("")
        self.reload_forward_sim()
        self.reload_inverse_sim()
        self.reload_optimized_sim()
        if os.path.exists(self.path_to_measures):
            self.measures = pickle.load(open(self.path_to_measures, "rb"))


    def _init_problem(self, path_to_domain, seed_position, sim_params, model_params_varying, model_params_fixed,
                      problem_type='forward'):

        params_dict = {}
        if problem_type in ['forward', 'inverse', 'optimized']:
            params_dict['path_to_domain'] = path_to_domain
            params_dict['model_params_varying'] = model_params_varying
            params_dict['sim_params'] = sim_params
            params_dict['seed_position'] = seed_position
            params_dict['model_params_fixed'] = model_params_fixed
            setattr(self, "params_%s"%problem_type, params_dict)
        else:
            self.logger.error("Invalid problem_type '%s'!"%problem_type)

        #-- Initialisation
        mesh, subdomains, boundaries = dio.read_mesh_hdf5(path_to_domain)

        # initial values
        if len(seed_position)==2:
            u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))', degree=1,
                                              a=0.5, x0=seed_position[0], y0=seed_position[1])
        elif len(seed_position)==3:
            u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2)  - a*pow(x[2]-z0, 2))', degree=1,
                                              a=0.5, x0=seed_position[0], y0=seed_position[1], z0=seed_position[2])
        u_0_disp_expr = fenics.Constant((0.0, 0.0))
        ivs = {0: u_0_disp_expr, 1: u_0_conc_expr}

        tissue_id_name_map = {1: 'CSF',
                              3: 'WM',
                              2: 'GM',
                              4: 'Ventricles'}

        boundary = Boundary()
        boundary_dict = {'boundary_all': boundary}

        dirichlet_bcs = {'clamped_0': {'bc_value': fenics.Constant((0.0, 0.0)),
                                       'named_boundary': 'boundary_all',
                                       'subspace_id': 0}
                         }

        von_neuman_bcs = {}

        sim = TumorGrowthBrain(mesh)
        #logger_names = ['simulation', 'simulation_base', 'simulation.helpers.helper_classes', 'simulation.helpers']
        logger_names = ['simulation', 'simulation_base']
        for logger_name in logger_names:
            try:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.INFO)
                logger.addHandler(self.logging_file_handler)
            except:
                pass


        sim.setup_global_parameters(subdomains=subdomains,
                                     domain_names=tissue_id_name_map,
                                     boundaries=boundary_dict,
                                     dirichlet_bcs=dirichlet_bcs,
                                     von_neumann_bcs=von_neuman_bcs
                                     )

        sim.setup_model_parameters(iv_expression=ivs,
                                   **sim_params, **model_params_varying, **model_params_fixed)

        self.save_problem_run_params(problem_type)

        return sim

    def save_problem_run_params(self, problem_type):
        attributes = ['path_to_domain', 'model_params', 'sim_params', 'seed_position']
        if problem_type == 'forward':
            save_path = self.path_run_params_forward
        elif problem_type == 'inverse':
            save_path = self.path_run_params_inverse
        elif problem_type == 'optimized':
            save_path = self.path_run_params_optimized
        else:
            self.logger.error("Non existing 'problem type'")

        attribute_name = "params_%s"%problem_type
        if hasattr(self, attribute_name):
            params_dict = getattr(self, attribute_name)
            with open(save_path, 'wb') as handle:
                pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.logger.warning("Parameters for problem '%s' do not exist"%problem_type)

    def read_problem_run_params(self, problem_type):
        if problem_type == 'forward':
            save_path = self.path_run_params_forward
        elif problem_type == 'inverse':
            save_path = self.path_run_params_inverse
        elif problem_type == 'optimized':
            save_path = self.path_run_params_optimized
        else:
            self.logger.error("Non existing 'problem type'")

        if os.path.exists(save_path):
            param_dict = pickle.load(open(save_path, "rb"))
            # check path_to_domain
            new_dir = self.rebase(param_dict['path_to_domain'])
            param_dict['path_to_domain'] = new_dir
            attribute_name = "params_%s" % problem_type
            setattr(self, attribute_name, param_dict)
            self.logger.info("Read parameters for problem '%s'"%problem_type)
        else:
            self.logger.warning("Cannot read parameters for problem '%s'" % problem_type)
            self.logger.warning("File '%s' does not exist" % save_path)


    def init_forward_problem(self, seed_position, model_params_varying, sim_params,
                             model_params_fixed, path_to_domain=None):
        if path_to_domain is None: # here we may use full or reduced domain as default
            path_to_domain = self.path_domain_2d_reduced
        self.sim_forward = self._init_problem(path_to_domain, seed_position,
                                              sim_params, model_params_varying, model_params_fixed,
                                              problem_type='forward')

    def get_seed_from_com(self):
        funspace = self.sim_forward.functionspace.get_functionspace(subspace_id=1)
        domain = self.sim_forward.subdomains.dx
        path =  self.path_forward_conc_threshold_08_reduced
        conc = dio.read_function_hdf5('function', funspace, path)
        com = compute_com(conc, domain)
        return com

    def init_inverse_problem(self, seed_position, model_params_varying, sim_params,
                             model_params_fixed=None, path_to_domain=None, seed_from_com=False):
        if path_to_domain is None: # this should always be the reduced domain, unless specified otherwise
            path_to_domain = self.path_domain_2d_reduced
        if model_params_fixed is None:
            model_params_fixed = self.params_forward['model_params_fixed']
        if seed_position is None:
            if seed_from_com:
                self.logger.info("Initialising seed for inverse problem from COM-position")
                seed_position = self.get_seed_from_com()
            else:
                self.logger.info("Using same seed for inverse problem as for forward problem")
                seed_position = self.params_forward['seed_position']
        if sim_params is None:
            sim_params = self.params_forward['sim_params']
        self.sim_inverse = self._init_problem(path_to_domain, seed_position,
                                              sim_params, model_params_varying, model_params_fixed,
                                              problem_type='inverse')


    def init_optimized_problem(self):
        model_params_varying = self.params_inverse['model_params_varying'].copy()
        self.logger.info(model_params_varying)
        model_params_varying.update(self.model_params_optimized)
        self.logger.info(model_params_varying)
        self.sim_optimized = self._init_problem(self.params_inverse['path_to_domain'],
                                                self.params_inverse['seed_position'],
                                                self.params_inverse['sim_params'],
                                                model_params_varying,
                                                self.params_inverse['model_params_fixed'],
                                                problem_type='optimized')


    def run_forward_sim(self, plot=None):
        if plot is None:
            plot = self.plot
        fu.ensure_dir_exists(self.path_forward_sim)
        self.sim_forward.run(save_method=None, plot=False, output_dir=self.path_forward_sim, clear_all=True)
        # save results
        disp_target_0, conc_target_0 = fenics.split(self.sim_forward.solution.copy())
        conc_target = self.sim_forward.functionspace.project_over_space(conc_target_0, subspace_id=1)
        disp_target = self.sim_forward.functionspace.project_over_space(disp_target_0, subspace_id=0)
        # save functions on reduced domain
        dio.save_function_mesh(conc_target, self.path_forward_conc, labelfunction=None,
                               subdomains=self.sim_forward.subdomains.subdomains)
        dio.save_function_mesh(disp_target, self.path_forward_disp, labelfunction=None,
                               subdomains=self.sim_forward.subdomains.subdomains)

        # Save as VTK & Plot
        self.sim_forward.init_postprocess(self.path_forward_sim)
        selection = [1, self.params_forward['sim_params']['sim_time']]
        self.sim_forward.postprocess.save_all(save_method='vtk', clear_all=False, selection=selection)

        if fenics.is_version("<2018.1.x") and plot:
            self.sim_forward.postprocess.plot_all(deformed=False, selection=selection,
                                     output_dir=os.path.join(self.path_forward_sim, 'plots'))
            self.sim_forward.postprocess.plot_all(deformed=True, selection=selection,
                                     output_dir=os.path.join(self.path_forward_sim, 'plots'))
            self.sim_forward.postprocess.plot_for_pub(deformed=True, selection=selection,
                                         output_dir=os.path.join(self.path_forward_sim, 'plots_for_pub'))


    def extract_2d_domain(self, plot=None):
        """
        Extracts 2D data from 3D image and atlas and saves as hdf5.
        """
        if plot is None:
            plot = self.plot

        # -- load brain atlas labels
        image_label = sitk.ReadImage(self.path_to_labels)
        image_label_select = image_label[:, :, self.image_z_slice]
        f_img_label = dio.image2fct2D(image_label_select)
        f_img_label.rename("label", "label")

        # -- load brain atlas image
        image = sitk.ReadImage(self.path_to_t1)
        image_select = image[:, :, self.image_z_slice]
        image_select.SetOrigin(image_label_select.GetOrigin())       # same origin as label image
        image_select.SetDirection(image_label_select.GetDirection()) # same orientation as label image
        f_img = dio.image2fct2D(image_select)
        f_img.rename("imgvalue", "label")

        # -- plot
        if plot:
            plott.show_img_seg_f(image=image_select, segmentation=image_label_select, show=True,
                                 path=os.path.join(self.path_domain_prep, 'image_label_from_sitk_image.png'))

            plott.show_img_seg_f(function=f_img_label, show=True,
                                 path=os.path.join(self.path_domain_prep, 'image_label_from_fenics_function.png'))

            plott.show_img_seg_f(function=f_img, show=True,
                                 path=os.path.join(self.path_domain_prep, 'image_from_fenics_function.png'))

        # == save
        fu.ensure_dir_exists(self.path_domain_prep)

        # -- save 2D images
        sitk.WriteImage(image_select, self.path_domain_image_2d)
        sitk.WriteImage(image_label_select, self.path_domain_labels_2d)

        # -- save label function:
        dio.save_function_mesh(f_img, self.path_domain_image_function_2d)
        dio.save_function_mesh(f_img_label, self.path_domain_labels_function_2d)

        # ======= save as sim domain
        mesh = f_img_label.function_space().mesh()

        tissue_id_name_map = {0: 'outside',
                              1: 'CSF',
                              3: 'WM',
                              2: 'GM',
                              4: 'Ventricles'}

        from simulation.helpers.helper_classes import SubDomains
        subdomains = SubDomains(mesh)
        subdomains.setup_subdomains(label_function=f_img_label)
        subdomains._setup_boundaries_from_subdomains(tissue_id_name_map=tissue_id_name_map)

        dio.save_mesh_hdf5(mesh, self.path_domain_2d, subdomains=subdomains.subdomains, boundaries=None)


    def reduce_2d_domain(self):

        mesh, subdomains, boundaries = dio.read_mesh_hdf5(self.path_domain_2d)

        # -- reduce domain size
        mesh_thr, subdomains_thr = dio.remove_mesh_subdomain(mesh, subdomains, lower_thr=1, upper_thr=4,
                                                             temp_dir=self.path_target_disp)
        dio.save_mesh_hdf5(mesh_thr, self.path_domain_2d_reduced, subdomains=subdomains_thr)

    def create_deformed_image(self):
        output_path = self.path_target_disp
        # 1) Load reference image
        T1 = sitk.ReadImage(self.path_domain_image_2d)
        size_T1_2d = T1.GetSize()

        # 2) Convert simulation result to labelmap with predefined resolution
        resolution = (*size_T1_2d, 1)
        name_sim_vtu = dio.create_file_name('all', self.params_forward['sim_params']["sim_time"])
        path_to_sim_vtu = os.path.join(self.path_forward_sim, 'merged', name_sim_vtu)
        sim_vtu = vtu.read_vtk_data(path_to_sim_vtu)
        # convert vtu to vti
        sim_vti = vtu.resample_to_image(sim_vtu, resolution)
        path_to_sim_vti = os.path.join(output_path, 'simulation_as_image.vti')
        vtu.write_vtk_data(sim_vti, path_to_sim_vti)

        if fenics.is_version("<2018.1.x"):
            # convert vti to normal image
            label_img = vtu.convert_vti_to_img(sim_vti, array_name='label_map', RGB=False)
            path_label_img = os.path.join(output_path, 'label_img_orig.nii')
            sitk.WriteImage(label_img, path_label_img)

        # 3) Create deformed labelmap
        sim_vtu_warped = vtu.warpVTU(sim_vtu, 'point', 'displacement')

        sim_vti_warped = vtu.resample_to_image(sim_vtu_warped, resolution)
        path_to_sim_vti_warped = os.path.join(output_path, 'simulation_as_image_warped.vti')
        vtu.write_vtk_data(sim_vti_warped, path_to_sim_vti_warped)

        if fenics.is_version("<2018.1.x"):
            label_img_warped = vtu.convert_vti_to_img(sim_vti_warped, array_name='label_map', RGB=False)
            sitk.WriteImage(label_img_warped, self.path_labels_2d_deformed)

        # 4) Extract displacement field
        disp_img_RGB = vtu.convert_vti_to_img(sim_vti, array_name='displacement', RGB=True)
        path_disp_img = os.path.join(output_path, 'displacement_img.nii')
        sitk.WriteImage(disp_img_RGB, path_disp_img)

        disp_img_RGB_inv = vtu.convert_vti_to_img(sim_vti, array_name='displacement', RGB=True, invert_values=True)
        path_disp_img_inv = os.path.join(output_path, 'displacement_img_inv.nii')
        sitk.WriteImage(disp_img_RGB_inv, path_disp_img_inv)

        # 5) Apply deformation field to warp image
        os.environ.copy()

        # -- Warp atlas T1 image:
        # - 1) atlas T1 by deformation field
        output_T1_warped = self.path_t1_2d_deformed

        reg.ants_apply_transforms(input_img=self.path_domain_image_2d, output_file=output_T1_warped,
                                  reference_img=self.path_domain_image_2d,
                                  transforms=[path_disp_img_inv], dim=2)

        if fenics.is_version("<2018.1.x"):
            # - 2) resample label map to T1
            output_label_resampled = os.path.join(output_path, 'label_img_resampledToT1.nii')
            reg.ants_apply_transforms(input_img=path_label_img, output_file=output_label_resampled,
                                      reference_img=self.path_domain_image_2d,
                                      transforms=[], dim=2)

            # - 3) resample label map to T1
            output_label_resampled_warped = os.path.join(output_path, 'label_img_resampledToT1_warped.nii')
            reg.ants_apply_transforms(input_img=path_label_img, output_file=output_label_resampled_warped,
                                      reference_img=self.path_domain_image_2d,
                                      transforms=[path_disp_img_inv], dim=2)

    def reconstruct_deformation_field(self, plot=None):
        if plot is None:
            plot = self.plot
        #-- registration to obtain displacement field
        path_to_reference_image = self.path_domain_image_2d
        path_to_deformed_image = self.path_t1_2d_deformed

        path_to_warped_image = os.path.join(self.path_target_disp, 'registered_image_deformed_to_reference')
        path_to_warp_field = os.path.join(self.path_target_disp, 'registered_image_deformed_to_reference_warp.nii.gz')

        reg.register_ants(path_to_reference_image, path_to_deformed_image, path_to_warped_image,
                          path_to_transform=path_to_warp_field, registration_type='Syn',
                          image_ext='nii', fixed_mask=None, moving_mask=None, verbose=1, dim=2)

        #-- read registration, convert to fenics function, save
        image_warp = sitk.ReadImage(path_to_warp_field)

        self.logger.info("== Transforming image to fenics function ... this is very slow...")
        f_img = dio.create_fenics_function_from_image(image_warp)
        dio.save_function_mesh(f_img, self.path_forward_disp_reconstructed)

        if plot:
            plott.show_img_seg_f(function=f_img, show=False,
                                 path=os.path.join(self.path_target_disp, 'displacement_from_registration_fenics.png'),
                                 dpi=300)



    def compare_displacement_field_simulated_registered(self, plot=None):
        if plot is None:
            plot = self.plot
        disp_sim, mesh_sim, subdomains_sim, boundaries_sim = dio.load_function_mesh(self.path_forward_disp,
                                                                                    functionspace='vector')
        disp_est, mesh_est, subdomains_est, boundaries_est = dio.load_function_mesh(self.path_forward_disp_reconstructed,
                                                                                    functionspace='vector')
        # -- chose simulation mesh as reference
        funspace_ref = disp_sim.function_space()
        # -- project/interpolate estimated displacement field over that mesh
        disp_est_ref = interpolate_non_matching(disp_est, funspace_ref)

        # compute errornorm
        error = fenics.errornorm(disp_sim, disp_est_ref)
        self.measures['errornorm_displacement_simulated_vs_registered'] = error

        # compute difference field
        disp_diff = fenics.project(disp_sim - disp_est_ref, funspace_ref, annotate=False)

        # -- map to reduced domain
        mesh_reduced, subdomains_reduced, boundaries_reduced = dio.read_mesh_hdf5(self.path_domain_2d_reduced)
        funspace_disp_reduced = fenics.VectorFunctionSpace(mesh_reduced, 'Lagrange', 1)
        disp_sim_reduced = interpolate_non_matching(disp_sim, funspace_disp_reduced)
        disp_est_reduced = interpolate_non_matching(disp_est, funspace_disp_reduced)

        error_reduced = fenics.errornorm(disp_sim_reduced, disp_est_reduced)
        self.measures['errornorm_displacement_simulated_vs_registered_reduced_domain'] = error_reduced

        disp_diff_reduced = fenics.project(disp_sim_reduced - disp_est_reduced, funspace_disp_reduced, annotate=False)

        dio.save_function_mesh(disp_sim_reduced, self.path_forward_disp_reduced, subdomains=subdomains_reduced)

        dio.save_function_mesh(disp_est_reduced, self.path_forward_disp_reconstructed_reduced, subdomains=subdomains_reduced)

        with open(self.path_to_measures, 'wb') as handle:
            pickle.dump(self.measures, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if plot:
            plott.show_img_seg_f(function=disp_sim, show=False,
                                 path=os.path.join(self.path_target_disp, 'displacement_field_from_simulation.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=disp_est_ref, show=False,
                                 path=os.path.join(self.path_target_disp, 'displacement_field_from_registration_ref_space.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=disp_sim_reduced, show=False,
                                 path=os.path.join(self.path_target_disp,
                                                   'displacement_field_from_simulation_reduced_domain.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=disp_est_reduced, show=False,
                                 path=os.path.join(self.path_target_disp,
                                                   'displacement_field_from_registration_reduced_domain.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=disp_diff, show=False,
                                 path=os.path.join(self.path_target_disp, 'displacement_field_difference.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=disp_diff_reduced, show=False,
                                 path=os.path.join(self.path_target_disp,
                                                   'displacement_field_difference_reduced_domain.png'),
                                 dpi=300)


    def create_thresholded_conc_fields(self, plot=None):
        if plot is None:
            plot = self.plot
        conc_sim, mesh_sim, subdomains_sim, boundaries_sim = dio.load_function_mesh(self.path_forward_conc,
                                                                                    functionspace='function', degree=2)

        conc_thresh_02 = fenics.project(thresh(conc_sim, 0.2), conc_sim.function_space(), annotate=False)
        dio.save_function_mesh(conc_thresh_02, self.path_forward_conc_threshold_02)
        conc_thresh_08 = fenics.project(thresh(conc_sim, 0.8), conc_sim.function_space(), annotate=False)
        dio.save_function_mesh(conc_thresh_08, self.path_forward_conc_threshold_08)

        if plot:
            plott.show_img_seg_f(function=conc_sim, show=False,
                                 path=os.path.join(self.path_target_conc, 'concentration_field_from_simulation.png'),
                                 dpi=300)

            plott.show_img_seg_f(function=conc_thresh_02, show=False,
                                 path=os.path.join(self.path_target_conc, 'thresholded_concentration_forward_02.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=conc_thresh_08, show=False,
                                 path=os.path.join(self.path_target_conc, 'thresholded_concentration_forward_08.png'),
                                 dpi=300)

        # reduced domain
        mesh_reduced, subdomains_reduced, boundaries_reduced = dio.read_mesh_hdf5(self.path_domain_2d_reduced)
        funspace_conc_reduced = fenics.FunctionSpace(mesh_reduced, 'Lagrange', 2)
        conc_sim_reduced = interpolate_non_matching(conc_sim, funspace_conc_reduced)

        conc_thresh_02_reduced = fenics.project(thresh(conc_sim_reduced, 0.2), conc_sim_reduced.function_space(), annotate=False)
        dio.save_function_mesh(conc_thresh_02_reduced, self.path_forward_conc_threshold_02_reduced)
        conc_thresh_08_reduced = fenics.project(thresh(conc_sim_reduced, 0.8), conc_sim_reduced.function_space(), annotate=False)
        dio.save_function_mesh(conc_thresh_08_reduced, self.path_forward_conc_threshold_08_reduced)

        if plot:
            plott.show_img_seg_f(function=conc_sim_reduced, show=False,
                                 path=os.path.join(self.path_target_conc,
                                                   'concentration_field_from_simulation_reduced_domain.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=conc_thresh_02_reduced, show=False,
                                 path=os.path.join(self.path_target_conc, 'thresholded_concentration_forward_02_reduced.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=conc_thresh_08_reduced, show=False,
                                 path=os.path.join(self.path_target_conc, 'thresholded_concentration_forward_08_reduced.png'),
                                 dpi=300)

    def custom_optimizer(self, J, m_global, dJ, H, bounds, **kwargs):
        self.logger.info("-- Starting optimization")
        try:
            opt_res = scipy_minimize(J, m_global, bounds=bounds, **kwargs)
            self.logger.info("-- Finished Optimization")
            for name, item in opt_res.items():
                self.logger.info("  - %s: %s"%(name, item))
                if not name in ['hess_inv']:
                    self.measures["optimization_%s"%name] = item
            return np.array(opt_res["x"])
        except Exception as e:
            self.logger.error("Error in optimization:")
            self.logger.error(e)

    def run_inverse_problem_n_params(self, params_init_values, params_names, solver_function,
                                     opt_params={}, **kwargs):
        params_init = [fenics.Constant(param) for param in params_init_values]
        # first run
        u = solver_function(params_init, **kwargs)

        # simulated fields
        disp_opt, conc_opt = fenics.split(u)
        disp_opt_proj = self.sim_inverse.functionspace.project_over_space(disp_opt, subspace_id=0)
        conc_opt_proj_02 = self.sim_inverse.functionspace.project_over_space(thresh(conc_opt, 0.2), subspace_id=1)
        conc_opt_proj_08 = self.sim_inverse.functionspace.project_over_space(thresh(conc_opt, 0.8), subspace_id=1)

        # target fields
        conc_target_thr_02 = dio.read_function_hdf5('function',
                                                    self.sim_inverse.functionspace.get_functionspace(subspace_id=1),
                                                    self.path_forward_conc_threshold_02_reduced)
        conc_target_thr_08 = dio.read_function_hdf5('function',
                                                    self.sim_inverse.functionspace.get_functionspace(subspace_id=1),
                                                    self.path_forward_conc_threshold_08_reduced)
        disp_target = dio.read_function_hdf5('function',
                                             self.sim_inverse.functionspace.get_functionspace(subspace_id=0),
                                             self.path_forward_disp_reconstructed_reduced)

        # optimization functional
        function_expr = fenics.inner(conc_opt_proj_02 - conc_target_thr_02,
                                     conc_opt_proj_02 - conc_target_thr_02) * self.sim_inverse.subdomains.dx \
                        + fenics.inner(conc_opt_proj_08 - conc_target_thr_08,
                                       conc_opt_proj_08 - conc_target_thr_08) * self.sim_inverse.subdomains.dx \
                        + fenics.inner(disp_opt_proj - disp_target,
                                       disp_opt_proj - disp_target) * self.sim_inverse.subdomains.dx

        if fenics.is_version("<2018.1.x"):
            J = fenics.Functional(function_expr)
        else:
            J = fenics.assemble(function_expr)

        controls = [fenics.Control(param) for param in params_init]
        rf = fenics.ReducedFunctional(J, controls, eval_cb_post=eval_cb_post, derivative_cb_post=derivative_cb_post)

        # optimization
        # -- for keeping progress info
        global opt_param_progress_post
        global opt_dj_progress_post
        global opt_date_time
        opt_param_progress_post = []
        opt_dj_progress_post = []
        opt_date_time = []

        # -- optimization parameters
        bounds_min = [0.005 for param in params_init]
        bounds_max = [0.5 for param in params_init]
        bounds_int = [bounds_min, bounds_max]
        params = {'bounds': bounds_int,
                  'method': "L-BFGS-B",
                  'tol': 1e-6,
                  'options': {'disp': True, 'gtol': 1e-6}}

        params.update(opt_params)
        self.params_optimization = params

        # -- save optimization parameters
        self.logger.info("Writing optimization parameters to '%s'" % self.path_optimization_params)
        with open(self.path_optimization_params, 'wb') as handle:
            pickle.dump(self.params_optimization, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # -- optimization
        self.logger.info("== Start Optimization")
        # m_opt = fenics.minimize(rf, **params)
        rf_np = fenics.ReducedFunctionalNumPy(rf)
        m_opt = fenics.optimization.minimize_custom(rf_np, algorithm=self.custom_optimizer, **params)

        # -- extract optimized parameters
        params_dict = {}
        self.logger.info("Optimized parameters:")
        for var, name in zip(m_opt, params_names):
            params_dict[name] = var.values()[0]
            self.logger.info("  - %s = %f"%(name, var.values()[0]))
        self.model_params_optimized = params_dict

        self.logger.info("Writing optimized simulation parameters to '%s'"%self.path_parameters_optimized)
        # -- save optimized parameters
        with open(self.path_parameters_optimized, 'wb') as handle:
            pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(opt_param_progress_post)
        print(opt_dj_progress_post)
        print(params_names)
        print(opt_date_time)
        opt_df = create_opt_progress_df(opt_param_progress_post, opt_dj_progress_post,
                                        params_names, datetime_list=opt_date_time)

        self.optimization_progress = opt_df
        self.logger.info(opt_df)
        opt_df.to_excel(self.path_optimization_progress_xls)
        opt_df.to_pickle(self.path_optimization_progress_pkl)

        if fenics.is_version(">2017.2.x"):
            self.sim_inverse.tape.visualise()

    def run_inverse_problem_2params(self, opt_params={}):
        params_names = ["D_WM", "rho_WM"]
        params_init_values = [self.params_inverse["model_params_varying"][name] for name in params_names]
        self.run_inverse_problem_n_params(params_init_values, params_names,
                                          solver_function=self.sim_inverse.run_for_adjoint_2params,
                                          opt_params=opt_params, output_dir=self.path_inverse_sim)

    def run_inverse_problem_3params(self, opt_params={}):
        params_names = ["D_WM", "rho_WM", "coupling"]
        params_init_values = [self.params_inverse["model_params_varying"][name] for name in params_names]
        self.run_inverse_problem_n_params(params_init_values, params_names,
                                          solver_function=self.sim_inverse.run_for_adjoint_3params,
                                          opt_params=opt_params, output_dir=self.path_inverse_sim)

    def run_inverse_problem_4params(self, opt_params={}):
        params_names = ["D_WM", "D_GM", "rho_WM", "coupling"]
        params_init_values = [self.params_inverse["model_params_varying"][name] for name in params_names]
        self.run_inverse_problem_n_params(params_init_values, params_names,
                                          solver_function=self.sim_inverse.run_for_adjoint_4params,
                                          opt_params=opt_params, output_dir=self.path_inverse_sim)

    def run_inverse_problem_5params(self, opt_params={}):
        params_names = ["D_WM", "D_GM", "rho_WM", "rho_GM", "coupling"]
        params_init_values = [self.params_inverse["model_params_varying"][name] for name in params_names]
        self.run_inverse_problem_n_params(params_init_values, params_names,
                                          solver_function=self.sim_inverse.run_for_adjoint,
                                          opt_params=opt_params, output_dir=self.path_inverse_sim)


    def run_optimized_sim(self, plot=None):
        if plot is None:
            plot = self.plot
        fu.ensure_dir_exists(self.path_optimized_sim)
        self.sim_optimized.run(save_method=None, plot=False, output_dir=self.path_optimized_sim, clear_all=True)
        # save results
        disp_target_0, conc_target_0 = fenics.split(self.sim_optimized.solution)
        conc_target = self.sim_optimized.functionspace.project_over_space(conc_target_0, subspace_id=1)
        disp_target = self.sim_optimized.functionspace.project_over_space(disp_target_0, subspace_id=0)
        # save functions on reduced domain
        dio.save_function_mesh(conc_target, self.path_optimized_conc_reduced, labelfunction=None,
                               subdomains=self.sim_optimized.subdomains.subdomains)
        dio.save_function_mesh(disp_target, self.path_optimized_disp_reduced, labelfunction=None,
                               subdomains=self.sim_optimized.subdomains.subdomains)

        # Save as VTK & Plot
        self.sim_optimized.init_postprocess(self.path_optimized_sim)
        selection = [1, self.params_inverse['sim_params']['sim_time']]
        self.sim_optimized.postprocess.save_all(save_method='vtk', clear_all=False, selection=selection)

        if fenics.is_version("<2018.1.x") and plot:
            self.sim_optimized.postprocess.plot_all(deformed=False, selection=selection,
                                                  output_dir=os.path.join(self.path_optimized_sim, 'plots'))
            self.sim_optimized.postprocess.plot_all(deformed=True, selection=selection,
                                                  output_dir=os.path.join(self.path_optimized_sim, 'plots'))
            self.sim_optimized.postprocess.plot_for_pub(deformed=True, selection=selection,
                                                      output_dir=os.path.join(self.path_optimized_sim, 'plots_for_pub'))


    def compare_original_optimized(self, plot=None):
        if plot is None:
            plot = self.plot

        output_path = self.path_comparison
        self.sim_forward.init_postprocess(output_path)
        self.sim_optimized.init_postprocess(output_path)
        #-- get solution at sim time from original forward model
        conc_orig_full = self.sim_forward.postprocess.get_solution_concentration(
                                    self.params_forward['sim_params']['sim_time']).copy()
        disp_orig_full = self.sim_forward.postprocess.get_solution_displacement(
                                    self.params_forward['sim_params']['sim_time']).copy()

        #-- get solution at sim time from optimized simulation
        conc_opt = self.sim_optimized.postprocess.get_solution_concentration(
                                    self.params_optimized['sim_params']['sim_time']).copy()
        disp_opt = self.sim_optimized.postprocess.get_solution_displacement(
                                    self.params_optimized['sim_params']['sim_time']).copy()

        #-- project original solution into domain of optimized solution
        # -- chose simulation mesh as reference
        funspace_disp_opt = self.sim_optimized.functionspace.get_functionspace(subspace_id=0)
        funspace_conc_opt = self.sim_optimized.functionspace.get_functionspace(subspace_id=1)
        conc_orig = interpolate_non_matching(conc_orig_full, funspace_conc_opt)
        disp_orig = interpolate_non_matching(disp_orig_full, funspace_disp_opt)

        #-- compute error norms
        error_conc = fenics.errornorm(conc_orig, conc_opt)
        error_disp = fenics.errornorm(disp_orig, disp_opt)

        self.measures['errornorm_displacement_forward_vs_optimized'] = error_disp
        self.measures['errornorm_concentration_forward_vs_optimized'] = error_conc

        with open(self.path_to_measures, 'wb') as handle:
            pickle.dump(self.measures, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if plot:
            plott.show_img_seg_f(function=conc_orig, path=os.path.join(output_path, 'conc_forward.png'))
            plott.show_img_seg_f(function=conc_opt, path=os.path.join(output_path, 'conc_opt.png'))
            conc_diff = fenics.project(conc_orig - conc_opt, funspace_conc_opt, annotate=False)
            plott.show_img_seg_f(function=conc_diff, path=os.path.join(output_path, 'conc_diff.png'))

            plott.show_img_seg_f(function=disp_orig, path=os.path.join(output_path, 'disp_forward.png'))
            plott.show_img_seg_f(function=disp_opt, path=os.path.join(output_path, 'disp_opt.png'))
            disp_diff = fenics.project(disp_orig - disp_opt, funspace_disp_opt, annotate=False)
            plott.show_img_seg_f(function=disp_diff, path=os.path.join(output_path, 'disp_diff.png'))

    def flatten_params(self, params_dict, prefix):
        params_flat = {}
        for param_type in ['model_params_fixed', 'model_params_varying', 'sim_params']:
            for name, value in params_dict[param_type].items():
                new_name = prefix + '_' + name
                params_flat[new_name] = value
        for name in ['path_to_domain']:
            new_name = prefix + '_' + name
            params_flat[new_name] = params_dict[name]
        params_flat[prefix + '_seed_position_' + 'x'] = params_dict['seed_position'][0]
        params_flat[prefix + '_seed_position_' + 'y'] = params_dict['seed_position'][1]
        return params_flat

    def compute_param_rel_errors(self):
        param_rel_error_dict = {}
        for param in self.params_forward['model_params_varying'].keys():
            param_value_forward = self.params_forward['model_params_varying'][param]
            param_value_optimized = self.params_optimized['model_params_varying'][param]
            param_rel_error = (param_value_optimized - param_value_forward)/param_value_forward
            param_rel_error_dict['relative_error_'+param] = param_rel_error
        return param_rel_error_dict

    def write_analysis_summary(self):
        summary={}
        summary.update(self.flatten_params(self.params_forward, 'forward'))
        summary.update(self.flatten_params(self.params_inverse, 'inverse'))
        summary.update(self.flatten_params(self.params_optimized, 'optimized'))
        summary.update(self.measures)
        summary.update(self.compute_param_rel_errors())
        summary['optimization_method'] = self.params_optimization['method']
        summary['optimization_tol'] = self.params_optimization['tol']
        opt_df = pd.read_pickle(self.path_optimization_progress_pkl)
        J_start = opt_df.iloc[0].J
        J_end = opt_df.iloc[-1].J
        time_delta = opt_df.iloc[-1].datetime - opt_df.iloc[0].datetime
        summary['objective_function_start'] = J_start
        summary['objective_function_end'] = J_end
        summary['total_time_optimization_seconds'] = time_delta.total_seconds()
        summary['number_iterations_optimization'] = opt_df.shape[0]
        with open(self.path_to_summary, 'wb') as handle:
            pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_volume(self, conc_fun, domain):
        vol = fenics.assemble(conc_fun*domain)
        return vol

    def post_process(self):
        self.compute_volume_thresholded()
        self.compute_com_all()
        # compute volumes / coms for each time step
        for measure in ['volume', 'com']:
            results_df = pd.DataFrame()
            for problem_type, threshold in product(*[['forward', 'optimized'], [0.2, 0.8]]):
                self.logger.info("Trying to compute '%s' for '%s' simulation with concentration threshold '%.02f'"%(measure, problem_type, threshold))
                self.logger.info("-- This computation is performed in the reference configuration, no deformation! --")
                results_tmp = self.compute_from_conc_for_each_time_step(threshold=threshold, problem_type=problem_type, computation=measure)
                print(results_tmp)
                if not results_tmp is None:
                    new_col_names = [ "_".join([problem_type, measure, str(threshold), name.lower()]) if name!='sim_time_step'
                                     else name.lower() for name in results_tmp.columns]
                    results_tmp.columns = new_col_names
                    if results_df.empty:
                        results_df = results_tmp
                    else:
                        results_df = pd.merge(results_df, results_tmp, how='left', on='sim_time_step')
            save_path = os.path.join(self.base_dir, measure + '.pkl')
            results_df.to_pickle(save_path)
            save_path = os.path.join(self.base_dir, measure + '.xls')
            results_df.to_excel(save_path)


    def compute_volume_thresholded(self):
        vol_dict = {'volume_threshold_02': self.path_forward_conc_threshold_02_reduced,
                      'volume_threshold_08': self.path_forward_conc_threshold_08_reduced}
        if hasattr(self, 'sim_inverse'):
            sim = self.sim_inverse
            for name, path in vol_dict.items():
                if os.path.exists(path):
                    self.logger.info("Computing Volume for '%s'" % (name))
                    conc = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=1),path)
                    vol = self.compute_volume(conc, sim.subdomains.dx)
                    self.measures[name] = vol
                else:
                    self.logger.warning("Cannot compute volume. '%s' does not exist" % path)

            with open(self.path_to_measures, 'wb') as handle:
                pickle.dump(self.measures, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            self.logger.warning("Cannot compute volume. Instance 'sim_inverse' does not exist")


    def compute_com_all(self):
        field_dict = {  'forward_threshold_02' : self.path_forward_conc_threshold_02_reduced,
                        'forward_threshold_08' : self.path_forward_conc_threshold_08_reduced,
                        'forward': self.path_forward_conc,
                        'inverse': self.path_optimized_conc_reduced}
        if hasattr(self, 'sim_forward'):
            for name, path in field_dict.items():
                if name.startswith('forward'):
                    funspace = self.sim_forward.functionspace.get_functionspace(subspace_id=1)
                    domain = self.sim_forward.subdomains.dx
                else:
                    funspace = self.sim_inverse.functionspace.get_functionspace(subspace_id=1)
                    domain = self.sim_inverse.subdomains.dx
                if os.path.exists(path):
                    self.logger.info("Computing COM for '%s'"%(name))
                    conc = dio.read_function_hdf5('function', funspace, path)
                    com = compute_com(conc, domain)
                    for i, coord in enumerate(com):
                        self.measures['com_%i_'%i + name] = coord
                else:
                    self.logger.warning("Cannot compute COM. '%s' does not exist"%path)

            with open(self.path_to_measures, 'wb') as handle:
                pickle.dump(self.measures, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            self.logger.warning("Cannot compute com. Instance 'sim_inverse' does not exist")


    def compute_from_conc_for_each_time_step(self, threshold=0.2, problem_type='forward', computation='volume'):
        if problem_type == 'forward':
            sim_name = 'sim_forward'
            base_path =  self.path_forward_sim
        elif problem_type == 'inverse':
            sim_name = 'sim_inverse'
            base_path = self.path_inverse_sim
        elif problem_type == 'optimized':
            sim_name = 'sim_optimized'
            base_path = self.path_optimized_sim
        else:
            self.logger.error("Non existing 'problem type'")

        if hasattr(self, sim_name):
            sim = getattr(self, sim_name)

            results = pd.DataFrame()
            rec_steps = sim.results.get_recording_steps()
            self.logger.info("Recording steps: %s"%rec_steps)
            for step in rec_steps:
                if step % 10 == 0:
                    self.logger.info("Computing '%s' for '%s' problem, step %03d" % (computation, problem_type, step))
                conc = sim.results.get_solution_function(subspace_id=1, recording_step=step)
                q = fenics.conditional(fenics.ge(conc, threshold), 1, 0)
                q_fun = sim.functionspace.project_over_space(q, subspace_id=1)
                # results
                results_dict = {}
                results_dict['sim_time_step'] = step
                if computation == 'volume':
                    results_dict['all'] = self.compute_volume(q_fun, sim.subdomains.dx)
                elif computation == 'com':
                    com = compute_com(q_fun, sim.subdomains.dx)
                    for i, coord in enumerate(com):
                        results_dict["%s_%i" % ('all', i)] = coord
                else:
                    self.logger.warning("Cannot compute '%s' -- underfined"%computation)
                # for all subdomains
                for subdomain_id, subdomain_name in sim.subdomains.tissue_id_name_map.items():
                    if computation == 'volume':
                        results_dict[subdomain_name] = self.compute_volume(q_fun, sim.subdomains.dx(subdomain_id))
                    elif computation == 'com':
                        com = compute_com(q_fun, sim.subdomains.dx(subdomain_id))
                        for i, coord in enumerate(com):
                            results_dict["%s_%i" % (subdomain_name, i)] = coord
                    else:
                        self.logger.warning("Cannot compute '%s' -- underfined" % computation)

                results = results.append(results_dict, ignore_index=True)

            #new_col_names = [ "_".join([problem_type, 'volume', str(threshold), name.lower()]) if name!='sim_time_step'
            #                  else name.lower() for name in volumes.columns]

            results.columns = [ name.lower() for name in results.columns]

            save_name = "_".join([computation, str(threshold)])+'.pkl'
            save_path = os.path.join(base_path, save_name)
            results.to_pickle(save_path)
            self.logger.info("Saving '%s' dataframe to '%s'"%(computation, save_path))
            save_name = "_".join([computation, str(threshold)])+'.xls'
            save_path = os.path.join(base_path, save_name)
            results.to_excel(save_path)
            self.logger.info("Saving '%s' dataframe to '%s'"%(computation, save_path))
            return results

        else:
            self.logger.warning("Cannot compute '%s' for '%s'. No such simulation instance names '%s'."%(computation, problem_type, sim_name))




