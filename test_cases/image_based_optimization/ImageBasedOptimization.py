import logging
import os
import pickle
from datetime import datetime
from itertools import product

import pandas as pd
import SimpleITK as sitk
from scipy.optimize import minimize as scipy_minimize

from glimslib import config
config.USE_ADJOINT = True
from glimslib import fenics_local as fenics
from glimslib.visualisation import plotting as plott, helpers as vh

from glimslib.simulation.simulation_tumor_growth_brain_quad import TumorGrowthBrain
from glimslib.simulation_helpers.helper_classes import Boundary, SubDomains
import glimslib.utils.file_utils as fu
import glimslib.utils.data_io as dio
import glimslib.utils.vtk_utils as vtu
import glimslib.utils.image_registration_utils as reg
import numpy as np

from ufl import tanh



if fenics.is_version("<2018.1.x"):
    fenics.set_log_level(fenics.DEBUG)
else:
    fenics.set_log_level(fenics.LogLevel.DEBUG)


class ImageBasedOptimization():

    def __init__(self, base_dir,
                 path_to_labels_patient=None, path_to_image_patient=None,
                 path_to_labels_atlas=None, path_to_image_atlas=None,
                 image_z_slice=None, plot=False):
        # logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # -- stream
        fu.ensure_dir_exists(base_dir)
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
                logger.setLevel(logging.DEBUG)
                logger.addHandler(self.logging_file_handler)
            except:
                pass

        # paths
        self.base_dir = base_dir
        self.path_to_base_params = os.path.join(self.base_dir, "base_params.pkl")
        # 2d vs 3d
        if image_z_slice is None:
            self.dim=3
        else:
            self.dim=2

        if (path_to_labels_patient is None) or (path_to_image_patient is None):
            self.mode='atlas_only'
        else:
            self.mode = 'atlas_patient'

        if (path_to_labels_atlas is None) or (path_to_image_atlas is None):
            if os.path.exists(self.path_to_base_params):
                with open(self.path_to_base_params, "rb") as f:
                    base_params = pickle.load(f)
                self.path_to_labels_patient_orig = base_params["path_to_labels_patient_orig"]
                self.path_to_image_patient_orig = base_params["path_to_image_patient_orig"]
                self.image_z_slice = base_params["image_z_slice"]
                self.path_to_labels_atlas_orig = base_params["path_to_labels_atlas_orig"]
                self.path_to_image_atlas_orig = base_params["path_to_image_atlas_orig"]
                self.plot = base_params["plot"]
                self.dim = base_params["dim"]
                self.mode = base_params["mode"]
            else:
                self.logger.error("Cannot initialize")
        else:
            self.path_to_labels_patient_orig = path_to_labels_patient
            self.path_to_image_patient_orig = path_to_image_patient
            self.path_to_labels_atlas_orig = path_to_labels_atlas
            self.path_to_image_atlas_orig = path_to_image_atlas
            self.image_z_slice = image_z_slice
            self.plot = plot
            base_params = {"path_to_labels_patient_orig" : self.path_to_labels_patient_orig,
                           "path_to_image_patient_orig" : self.path_to_image_patient_orig,
                           "image_z_slice" : self.image_z_slice,
                           "path_to_labels_atlas_orig" : self.path_to_labels_atlas_orig,
                           "path_to_image_atlas_orig": self.path_to_image_atlas_orig,
                           "plot" : self.plot,
                           "dim" : self.dim,
                           "mode" : self.mode}
            fu.ensure_dir_exists(self.base_dir)
            with open(self.path_to_base_params, 'wb') as handle:
                pickle.dump(base_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self._init_defaults()
        self.logger.info("Initialized ImageBasedOptimization instance")

    def _init_defaults(self):
        #== Processing Directories
        self.steps_sub_path_map = {'temp' : '00_temp',
                                  'domain_preparation' : '01_domain_preparation',
                                  'forward_sim' : '02_forward_simulation',
                                  'target_fields' : '03_target_fields',
                                  'inverse_sim' : '02_inverse_simulation',
                                  'optimized_sim' : '02_optimized_simulation'}
        self.steps_path_map = {}
        for key, subpath in self.steps_sub_path_map.items():
            path = os.path.join(self.base_dir, subpath)
            self.steps_path_map[key] = path
            fu.ensure_dir_exists(path)

        #== File Paths
        self.extension_img='nii'
        #-- Registration: Patient-sepcific Reference
        self.path_3d_atlas_reg_to_patient_image = \
                                    self._create_file_path(category='domain_preparation', dim=3, source_type='atlas',
                                                            info_type='image', data_type='image', domain_type='full',
                                                            registered='to_patient')
        self.path_3d_atlas_reg_to_patient_trafo = \
                                    self._create_file_path(category='domain_preparation', dim=3, source_type='atlas',
                                                            info_type='trafo', data_type='trafo', domain_type=None,
                                                            registered='to_patient')
        self.path_3d_atlas_reg_to_patient_seg =  \
                                    self._create_file_path(category='domain_preparation', dim=3, source_type='atlas',
                                                           info_type='labels', data_type='image', domain_type='full',
                                                           registered='to_patient')
        #-- Domain Extraction: 2d domain from 3d
        self.path_domain_image = self._create_file_path(category='domain_preparation', dim=self.dim, source_type='domain',
                                                        info_type='image', data_type='image', domain_type='full')
        self.path_domain_labels = self._create_file_path(category='domain_preparation', dim=self.dim, source_type='domain',
                                                         info_type='labels', data_type='image', domain_type='full')
        self.path_domain_image_fct = self._create_file_path(category='domain_preparation', dim=self.dim, source_type='domain',
                                                            info_type='image', data_type='fenics', domain_type='full')
        self.path_domain_labels_fct = self._create_file_path(category='domain_preparation', dim=self.dim, source_type='domain',
                                                             info_type='labels', data_type='fenics', domain_type='full')
        self.path_domain_mesh = self._create_file_path(category='domain_preparation', dim=self.dim, source_type='domain',
                                                       info_type='mesh', data_type='fenics', domain_type='full')
        self.path_domain_mesh_red = self._create_file_path(category='domain_preparation', dim=self.dim, source_type='domain',
                                                           info_type='mesh', data_type='fenics', domain_type='reduced')
        #-- Forward Sim:
        self.path_run_params_forward = self._create_file_path(category='forward_sim', dim=None, source_type='forward_sim',
                                                            info_type='parameters', data_type='pickle', domain_type=None)
        self.path_forward_conc = self._create_file_path(category='forward_sim', dim=self.dim, source_type='forward_sim',
                                                            info_type='concentration', data_type='fenics', domain_type=None)
        self.path_forward_disp = self._create_file_path(category='forward_sim', dim=self.dim, source_type='forward_sim',
                                                            info_type='displacement', data_type='fenics', domain_type=None)
        #-- Target Fields:
        self.path_target_disp = self._create_file_path(category='target_fields', dim=self.dim, source_type='target',
                                                            info_type='displacement', data_type='fenics', domain_type=None)
        self.path_target_conc = self._create_file_path(category='target_fields', dim=self.dim, source_type='target',
                                                       info_type='concentration', data_type='fenics', domain_type=None)


        self.path_run_params_inverse = self._create_file_path(category='inverse_sim', dim=None, source_type='inverse_sim',
                                                              info_type='parameters', data_type='pickle', domain_type=None)

        self.path_run_params_optimized = self._create_file_path(category='optimized_sim', dim=None, source_type='optimized_sim',
                                                              info_type='parameters', data_type='pickle', domain_type=None)



    def _create_file_path(self, category='domain_preparation',
                          dim=None, source_type='atlas',
                          info_type='image', data_type='image', domain_type='full',
                          registered=None, with_ext=True):
        """
        :param dim: 2, 3, None
        :param source_type: patient, atlas, domain, forward_sim
        :param info_type: labels, image, mesh, trafo, parameters, concentration, displacement
        :param data_type: image, fenics, trafo, pickle
        :param domain_type: reduced, full, None
        :param registered: to_patient, None
        :return:
        """
        base_path = self.steps_path_map[category]
        fu.ensure_dir_exists(base_path)
        if source_type:
            file_name = "%s_%s_%s"%(source_type, info_type, data_type)
        else:
            file_name = "%s_%s" % (info_type, data_type)
        if dim:
            file_name = "%s_%dd" % (file_name, dim)
        if domain_type:
            file_name = "%s_%s" % (file_name, domain_type)
        if registered:
            file_name = "%s_registered_%s" % (file_name, registered)

        if data_type=='image':
            file_ext  = self.extension_img
        elif data_type=='fenics':
            file_ext  = "h5"
        elif data_type=='trafo':
            file_ext = 'mat'
        elif data_type=='pickle':
            file_ext = 'pkl'
        if with_ext:
            file_name = file_name + '.' + file_ext
        file_path = os.path.join(base_path, file_name)
        return file_path


    def _reload_model_sim(self, problem_type='forward'):
        self.logger.info("=== Reloading simulation '%s'."%problem_type)
        param_attr_name = "params_%s" % problem_type
        if hasattr(self, param_attr_name):
            path_map_name = problem_type + '_sim'
            try:
                data_path = os.path.join(self.steps_path_map[path_map_name], 'solution_timeseries.h5')
            except:
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


    def _reload_forward_sim(self):
        self._read_problem_run_params(problem_type='forward')
        self._reload_model_sim(problem_type='forward')


    def _reload_inverse_sim(self):
        self._read_problem_run_params(problem_type='inverse')
        self._reload_model_sim(problem_type='inverse')
        # parameters of optimization process
        if os.path.exists(self.path_optimization_params):
            with open(self.path_optimization_params, "rb") as f:
                self.params_optimization  = pickle.load(f)
        # parameter results from optimization process
        if os.path.exists(self.path_parameters_optimized):
            with open(self.path_parameters_optimized, "rb") as f:
                self.model_params_optimized  = pickle.load(f)

    def _reload_optimized_sim(self):
        self._read_problem_run_params(problem_type='optimized')
        self._reload_model_sim(problem_type='optimized')

    def reload_state(self):
        self.logger.info("")
        self._reload_forward_sim()
        #self._reload_inverse_sim()
        #self._reload_optimized_sim()
        # if os.path.exists(self.path_to_measures):
        #     with open(self.path_to_measures, "rb") as f:
        #         self.measures  = pickle.load(f)


    def prepare_domain(self, plot=None):
        if (self.path_to_image_patient_orig is not None) and (self.path_to_labels_patient_orig is not None):
            self._create_patient_specific_reference_from_atlas()
            self.path_to_domain_image_3d = self.path_3d_atlas_reg_to_patient_image
            self.path_to_domain_labels_3d = self.path_3d_atlas_reg_to_patient_seg
        else:
            self.path_to_domain_image_3d = self.path_to_image_atlas_orig
            self.path_to_domain_labels_3d = self.path_to_labels_atlas_orig
        if self.dim==2:
            self._extract_2d_domain(plot=plot)
            self._reduce_2d_domain()
        elif self.dim==3:
             pass


    def _create_patient_specific_reference_from_atlas(self):
        """
        Affine registration to match original (3d) atlas to original (3d) patient image:
        """
        if os.path.exists(self.path_to_image_atlas_orig):
            reg.register_ants(fixed_img=self.path_to_image_patient_orig,
                              moving_img=self.path_to_image_atlas_orig,
                              output_prefix='.'.join(self.path_3d_atlas_reg_to_patient_image.split('.')[:-1]),
                              path_to_transform=self.path_3d_atlas_reg_to_patient_trafo,
                              registration_type='Affine',
                              image_ext=self.path_3d_atlas_reg_to_patient_image.split('.')[-1],
                              fixed_mask=None, moving_mask=None, verbose=1, dim=3)

        if os.path.exists(self.path_to_labels_atlas_orig) and os.path.exists(self.path_3d_atlas_reg_to_patient_trafo):
            reg.ants_apply_transforms(input_img=self.path_to_labels_atlas_orig,
                                      reference_img=self.path_to_image_atlas_orig,
                                      output_file=self.path_3d_atlas_reg_to_patient_seg,
                                      transforms=[self.path_3d_atlas_reg_to_patient_trafo],
                                      dim=3, interpolation='GenericLabel')

    def _extract_2d_domain(self, plot=None):
        """
        Extracts 2D data from 3D image and atlas and saves as hdf5.
        """
        if plot is None:
            plot = self.plot

        path_img_3d = self.path_to_domain_image_3d
        path_labels_3d = self.path_to_domain_labels_3d
        path_img_2d = self.path_domain_image
        path_labels_2d = self.path_domain_labels
        path_img_fct = self.path_domain_image_fct
        path_labels_fct = self.path_domain_labels_fct
        path_meshfct = self.path_domain_mesh

        # -- load patient labels
        image_label = sitk.Cast(sitk.ReadImage(path_labels_3d), sitk.sitkUInt8)
        image_label_select = image_label[:, :, self.image_z_slice]
        f_img_label = dio.image2fct2D(image_label_select)
        f_img_label.rename("label", "label")

        # -- load patient image
        image = sitk.ReadImage(path_img_3d)
        image_select = image[:, :, self.image_z_slice]
        image_select.SetOrigin(image_label_select.GetOrigin())  # same origin as label image
        image_select.SetDirection(image_label_select.GetDirection())  # same orientation as label image
        f_img = dio.image2fct2D(image_select)
        f_img.rename("imgvalue", "label")

        # -- plot
        path_domain_prep = self.steps_path_map['domain_preparation']
        if plot:
            plott.show_img_seg_f(image=image_select, segmentation=image_label_select, show=True,
                                 path=os.path.join(path_domain_prep, 'domain', 'label_from_sitk_image.png'))

            plott.show_img_seg_f(function=f_img_label, show=True,
                                 path=os.path.join(path_domain_prep, 'domain', 'label_from_fenics_function.png'))

            plott.show_img_seg_f(function=f_img, show=True,
                                 path=os.path.join(path_domain_prep, 'domain', 'from_fenics_function.png'))

        # == save
        # -- save 2D images
        sitk.WriteImage(image_select, path_img_2d)
        sitk.WriteImage(image_label_select, path_labels_2d)

        # -- save label function:
        dio.save_function_mesh(f_img, path_img_fct)
        dio.save_function_mesh(f_img_label, path_labels_fct)

        # ======= save as sim domain
        mesh = f_img_label.function_space().mesh()

        tissue_id_name_map = {0: 'outside',
                              1: 'CSF',
                              3: 'WM',
                              2: 'GM',
                              4: 'Ventricles'}

        subdomains = SubDomains(mesh)
        subdomains.setup_subdomains(label_function=f_img_label)
        subdomains._setup_boundaries_from_subdomains(tissue_id_name_map=tissue_id_name_map)

        dio.save_mesh_hdf5(mesh, path_meshfct, subdomains=subdomains.subdomains, boundaries=None)

    def _reduce_2d_domain(self):
        mesh, subdomains, boundaries = dio.read_mesh_hdf5(self.path_domain_mesh)
        # -- reduce domain size
        mesh_thr, subdomains_thr = dio.remove_mesh_subdomain(mesh, subdomains, lower_thr=1, upper_thr=4,
                                                             temp_dir=self.steps_path_map['temp'])
        dio.save_mesh_hdf5(mesh_thr, self.path_domain_mesh_red, subdomains=subdomains_thr)

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
        if len(seed_position)==2 and self.dim==2:
            u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))', degree=1,
                                              a=0.5, x0=seed_position[0], y0=seed_position[1])
        elif len(seed_position)==3 and self.dim==3:
            u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2)  - a*pow(x[2]-z0, 2))', degree=1,
                                              a=0.5, x0=seed_position[0], y0=seed_position[1], z0=seed_position[2])
        else:
            self.logger.error("Seed coordinates not compatible with spatial dimensions")

        u_0_disp_expr = fenics.Constant(np.zeros(self.dim))
        ivs = {0: u_0_disp_expr, 1: u_0_conc_expr}

        tissue_id_name_map = {1: 'CSF',
                              3: 'WM',
                              2: 'GM',
                              4: 'Ventricles'}

        boundary = Boundary()
        boundary_dict = {'boundary_all': boundary}
        dirichlet_bcs = {'clamped_0': {'bc_value': fenics.Constant(np.zeros(self.dim)),
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
        self._save_problem_run_params(problem_type)
        return sim

    def _save_problem_run_params(self, problem_type):
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

    def _rebase(self, old_path):
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


    def _read_problem_run_params(self, problem_type):
        if problem_type == 'forward':
            save_path = self.path_run_params_forward
        elif problem_type == 'inverse':
            save_path = self.path_run_params_inverse
        elif problem_type == 'optimized':
            save_path = self.path_run_params_optimized
        else:
            self.logger.error("Non existing 'problem type'")

        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                param_dict = pickle.load(f)
            # check path_to_domain
            new_dir = self._rebase(param_dict['path_to_domain'])
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
            if self.dim==2:
                path_to_domain = self.path_domain_mesh_red
            elif self.dim==3:
                pass
        self.sim_forward = self._init_problem(path_to_domain, seed_position,
                                              sim_params, model_params_varying, model_params_fixed,
                                              problem_type='forward')

    def run_forward_sim(self, plot=None):
        if plot is None:
            plot = self.plot
        path_forward_sim = self.steps_path_map['forward_sim']
        self.sim_forward.run(save_method=None, plot=False, output_dir=path_forward_sim, clear_all=True)
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
        self.sim_forward.init_postprocess(path_forward_sim)
        selection = [1, self.params_forward['sim_params']['sim_time']]
        self.sim_forward.postprocess.save_all(save_method='vtk', clear_all=False, selection=selection)
        if fenics.is_version("<2018.1.x") and plot and self.dim==2:
            self.sim_forward.postprocess.plot_all(deformed=False, selection=selection,
                                     output_dir=os.path.join(path_forward_sim, 'plots'))
            self.sim_forward.postprocess.plot_all(deformed=True, selection=selection,
                                     output_dir=os.path.join(path_forward_sim, 'plots'))
            self.sim_forward.postprocess.plot_for_pub(deformed=True, selection=selection,
                                         output_dir=os.path.join(path_forward_sim, 'plots_for_pub'))


    def get_target_fields(self, plot=None):
        if (self.path_to_image_patient_orig is not None) and (self.path_to_labels_patient_orig is not None):
            pass
        else:
            self._create_deformed_image()


    def _create_deformed_image(self):
        path_forward_sim = self.steps_path_map['forward_sim']
        output_path = self.path_target_disp
        # 1) Load reference image
        T1 = sitk.ReadImage(self.path_domain_image)
        size_T1 = T1.GetSize()

        # 2) Convert simulation result to labelmap with predefined resolution
        if self.dim==2:
            resolution = (*size_T1, 1)
        else:
            resolution = size_T1
        name_sim_vtu = dio.create_file_name('all', self.params_forward['sim_params']["sim_time"])
        path_to_sim_vtu = os.path.join(path_forward_sim, 'merged', name_sim_vtu)
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
            path_labels_def = os.path.join(output_path, 'labels_img_warped.nii')
            sitk.WriteImage(label_img_warped, path_labels_def)

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
        output_T1_warped = os.path.join(output_path, 'T1_img_warped.nii')

        reg.ants_apply_transforms(input_img=self.path_domain_image, output_file=output_T1_warped,
                                  reference_img=self.path_domain_image,
                                  transforms=[path_disp_img_inv], dim=self.dim)

        if fenics.is_version("<2018.1.x"):
            # - 2) resample label map to T1
            output_label_resampled = os.path.join(output_path, 'label_img_resampledToT1.nii')
            reg.ants_apply_transforms(input_img=path_label_img, output_file=output_label_resampled,
                                      reference_img=self.path_domain_image,
                                      transforms=[], dim=self.dim)

            # - 3) resample label map to T1
            output_label_resampled_warped = os.path.join(output_path, 'label_img_resampledToT1_warped.nii')
            reg.ants_apply_transforms(input_img=path_label_img, output_file=output_label_resampled_warped,
                                      reference_img=self.path_domain_image,
                                      transforms=[path_disp_img_inv], dim=self.dim)