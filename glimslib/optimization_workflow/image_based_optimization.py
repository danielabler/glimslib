import logging
import os
import pickle
from abc import abstractmethod
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.optimize import minimize as scipy_minimize

from glimslib import config

config.USE_ADJOINT = True

from glimslib import fenics_local as fenics
from glimslib.optimization_workflow.path_io import PathIO
from glimslib.utils import file_utils as fu
import glimslib.utils.data_io as dio
import glimslib.utils.meshing as meshing
import glimslib.utils.vtk_utils as vtu
from glimslib.visualisation import plotting as plott, helpers as vh
import glimslib.utils.image_registration_utils as reg
from glimslib.simulation_helpers.helper_classes import SubDomains, Boundary
from glimslib.simulation.simulation_tumor_growth_brain_quad import TumorGrowthBrain
from ufl import tanh

if fenics.is_version("<2018.1.x"):
    fenics.set_log_level(fenics.INFO)
else:
    fenics.set_log_level(fenics.LogLevel.INFO)


class ImageBasedOptimizationBase:

    def __init__(self, base_dir,
                 path_to_labels_atlas=None, path_to_image_atlas=None,
                 image_z_slice=None, plot=False):
        # paths
        self.base_dir = base_dir
        self.data = PathIO(self.base_dir)
        self.path_to_base_params = self.data.create_params_path(processing=None)
        self._setup_paths()
        # init steps
        self._setup_loggers()
        if path_to_image_atlas and path_to_labels_atlas:
            self.path_to_image_atlas_orig = path_to_image_atlas
            self.path_to_labels_atlas_orig = path_to_labels_atlas
            self.image_z_slice = image_z_slice
            self.plot = plot
            self.conc_threshold_levels = {'T2': 0.12,
                                          'T1': 0.80}
            if image_z_slice is None:
                self.dim = 3
            else:
                self.dim = 2

            self.measures = {}
            self._save_state()
        else:
            self._load_state()

    @abstractmethod
    def prepare_domain(self):
        pass

    @abstractmethod
    def create_target_fields(self):
        pass

    def _setup_paths(self):
        # == Processing Directories
        self.steps_sub_path_map = {'temp': '00_temp',
                                   'plots': '00_plots',
                                   'domain_prep': '01_domain_preparation',
                                   'forward_sim': '02_forward_simulation',
                                   'target_fields': '03_target_fields',
                                   'inverse_sim': '02_inverse_simulation',
                                   'optimized_sim': '02_optimized_simulation',
                                   'summary': 'summary',
                                   'comparison': 'comparison'}
        self.steps_path_map = {}
        for key, subpath in self.steps_sub_path_map.items():
            path = os.path.join(self.base_dir, subpath)
            self.steps_path_map[key] = path
            fu.ensure_dir_exists(path)

        # image paths
        self.path_to_domain_image = self.data.create_image_path(processing=self.steps_sub_path_map['domain_prep'],
                                                                datasource='domain',
                                                                domain='full', frame='reference', datatype='image',
                                                                content='T1', extension='mha')
        self.path_to_domain_labels = self.data.create_image_path(processing=self.steps_sub_path_map['domain_prep'],
                                                                 datasource='domain',
                                                                 domain='full', frame='reference', datatype='image',
                                                                 content='T1', extension='mha')
        self.path_to_domain_image_3d = self.data.create_image_path(processing=self.steps_sub_path_map['domain_prep'],
                                                                   datasource='domain',
                                                                   domain='full', frame='reference', datatype='image',
                                                                   content='T1', extension='mha', dim=3)
        self.path_to_domain_labels_3d = self.data.create_image_path(processing=self.steps_sub_path_map['domain_prep'],
                                                                    datasource='domain',
                                                                    domain='full', frame='reference', datatype='image',
                                                                    content='T1', extension='mha', dim=3)

        # sim paths
        self.path_run_params_forward = self.data.create_params_path(processing=self.steps_sub_path_map['forward_sim'])
        self.path_run_params_inverse = self.data.create_params_path(processing=self.steps_sub_path_map['inverse_sim'])
        self.path_run_params_optimized = self.data.create_params_path(
            processing=self.steps_sub_path_map['optimized_sim'])

        self.path_target_fields = self.data.create_path(processing=self.steps_sub_path_map['target_fields'])
        self.path_domain_prep = self.data.create_path(processing=self.steps_sub_path_map['domain_prep'])

        self.path_to_measures = self.data.create_params_path(processing=self.steps_sub_path_map['summary'],
                                                             content='measures')
        self.path_to_summary = self.data.create_params_path(processing=self.steps_sub_path_map['summary'],
                                                            content='summary')

        self.path_optimization_params = self.data.create_params_path(processing=self.steps_sub_path_map['inverse_sim'],
                                                                     datasource='optimization_params')
        self.path_parameters_optimized = self.data.create_params_path(processing=self.steps_sub_path_map['inverse_sim'],
                                                                      datasource='parameters_optimized')

    def _setup_loggers(self):
        # logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # -- stream
        fu.ensure_dir_exists(self.base_dir)
        self.logging_stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logging_stream_handler.setFormatter(formatter)
        self.logger.addHandler(self.logging_stream_handler)
        # -- file
        self.logging_file_handler = logging.FileHandler(os.path.join(self.base_dir,
                                                                     datetime.now().strftime(
                                                                         "logger_ImagedBasedOpt_%Y-%m-%d_%H-%M-%S") + '.log'))
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

    def _save_state(self):
        # save base params
        base_params = {"path_to_labels_atlas_orig": self.path_to_labels_atlas_orig,
                       "path_to_image_atlas_orig": self.path_to_image_atlas_orig,
                       "image_z_slice": self.image_z_slice,
                       "plot": self.plot,
                       "dim": self.dim,
                       "conc_threshold_levels": self.conc_threshold_levels
                       }
        # save all paths
        path_dict = {}
        for attr in dir(self):
            if attr.startswith('path'):
                path_dict[attr] = getattr(self, attr)
        base_params.update(path_dict)
        fu.ensure_dir_exists(self.base_dir)
        with open(self.path_to_base_params, 'wb') as handle:
            pickle.dump(base_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.path_to_measures, 'wb') as handle:
            pickle.dump(self.measures, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_state(self):
        # load base params
        if os.path.exists(self.path_to_base_params):
            with open(self.path_to_base_params, "rb") as f:
                base_params = pickle.load(f)
            for name, value in base_params.items():
                setattr(self, name, value)
        if os.path.exists(self.path_to_measures):
            self.measures = pickle.load(open(self.path_to_measures, "rb"))
        else:
            self.logger.warning("Cannot initialize ")

    # Functions for Domain Preparation

    def _extract_2d_domain(self,
                           path_img_3d, path_labels_3d,
                           path_img_2d, path_labels_2d,
                           plot_path,
                           path_img_fct=None, path_labels_fct=None,
                           path_meshfct=None,
                           plot=None):
        """
        Extracts 2D data from 3D image and atlas and saves as hdf5.
        """
        if plot is None:
            plot = self.plot

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
        if plot:
            plott.show_img_seg_f(image=image_select, segmentation=image_label_select, show=True,
                                 path=os.path.join(plot_path, 'image_from_sitk_image.png'))

            plott.show_img_seg_f(function=f_img_label, show=True,
                                 path=os.path.join(plot_path, 'label_from_fenics_function.png'))

            plott.show_img_seg_f(function=f_img, show=True,
                                 path=os.path.join(plot_path, 'image_fenics_function.png'))

        # == save
        # -- save 2D images
        sitk.WriteImage(image_select, path_img_2d)
        sitk.WriteImage(image_label_select, path_labels_2d)

        # -- save label function:
        if path_img_fct and path_labels_fct:
            dio.save_function_mesh(f_img, path_img_fct)
            dio.save_function_mesh(f_img_label, path_labels_fct)

        # ======= save as sim domain
        if path_meshfct:
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

    @staticmethod
    def _reduce_2d_domain(path_domain_mesh, path_domain_mesh_red, tmp_path):
        mesh, subdomains, boundaries = dio.read_mesh_hdf5(path_domain_mesh)
        # -- reduce domain size
        mesh_thr, subdomains_thr = dio.remove_mesh_subdomain(mesh, subdomains, lower_thr=1, upper_thr=4,
                                                             temp_dir=tmp_path)
        dio.save_mesh_hdf5(mesh_thr, path_domain_mesh_red, subdomains=subdomains_thr)

    @staticmethod
    def _mesh_3d_domain(path_to_image, path_to_mesh, tissues_dict=None):
        if tissues_dict is None:
            tissues_dict = {'gray_matter': {'domain_id': 2, 'cell_size': 2},
                            'global': {"cell_radius_edge_ratio": 2.1,
                                       "cell_size": 5,
                                       "facet_angle": 30.0,
                                       "facet_distance": 2,
                                       "facet_size": 2}
                            }
        mash_dir = os.path.dirname(path_to_mesh)
        path_to_xml_file = os.path.join(mash_dir, "mesh_config.xml")

        meshing.create_mesh_xml(path_to_image_in=path_to_image,
                                path_to_mesh_out=path_to_mesh,
                                tissues_dict=tissues_dict,
                                path_to_xml_file=path_to_xml_file)

        meshing.mesh_image(path_to_meshtool_bin=config.path_to_meshtool_bin,
                           path_to_meshtool_xsd=config.path_to_meshtool_xsd,
                           path_to_config_file=path_to_xml_file)

    def mesh_domain(self, plot=None):
        """
        Creates fenics mesh function from 3d image in
        - 2d: by extracting 2d slice from 3d image and converting 2d regular image grid to fenics mesh function
        - 3d: by meshing 3D image using meshtool and converting VTU to fenics mesh function

        input: self.path_to_domain_image_3d, self.path_to_domain_labels_3d
        output: self.path_to_domain_meshfct
        """

        self.path_to_domain_meshfct = self.data.create_fenics_path(processing=self.steps_sub_path_map['domain_prep'],
                                                                   datasource='domain',
                                                                   domain='full', frame='reference', datatype='fenics',
                                                                   content='mesh')

        self.path_to_domain_meshfct_red = self.data.create_fenics_path(
            processing=self.steps_sub_path_map['domain_prep'],
            datasource='domain',
            domain='reduced', frame='reference',
            datatype='fenics', content='mesh')

        if self.dim == 2:
            self.path_to_domain_image_2d = self.data.create_image_path(
                processing=self.steps_sub_path_map['domain_prep'], datasource='domain',
                domain='full', frame='reference', datatype='image', content='T1', extension='mha', dim=2)
            self.path_to_domain_labels_2d = self.data.create_image_path(
                processing=self.steps_sub_path_map['domain_prep'], datasource='domain',
                domain='full', frame='reference', datatype='image', content='labels', extension='mha', dim=2)

            self.path_to_domain_image_2d_fct = self.data.create_fenics_path(
                processing=self.steps_sub_path_map['domain_prep'], datasource='domain',
                domain='full', frame='reference', datatype='fenics', content='T1')
            self.path_to_domain_labels_2d_fct = self.data.create_fenics_path(
                processing=self.steps_sub_path_map['domain_prep'], datasource='domain',
                domain='full', frame='reference', datatype='fenics', content='labels')

            self._extract_2d_domain(path_img_3d=self.path_to_domain_image_3d,
                                    path_labels_3d=self.path_to_domain_labels_3d,
                                    path_img_2d=self.path_to_domain_image_2d,
                                    path_labels_2d=self.path_to_domain_labels_2d,
                                    path_img_fct=self.path_to_domain_image_2d_fct,
                                    path_labels_fct=self.path_to_domain_labels_2d_fct,
                                    path_meshfct=self.path_to_domain_meshfct,
                                    plot_path=self.data.create_path(processing=self.steps_sub_path_map['domain_prep']),
                                    plot=plot)

            self._reduce_2d_domain(path_domain_mesh=self.path_to_domain_meshfct,
                                   path_domain_mesh_red=self.path_to_domain_meshfct_red,
                                   tmp_path=self.data.create_path(processing='tmp'))

            self.path_to_domain_meshfct_main = self.path_to_domain_meshfct_red
            self.path_to_domain_image_main = self.path_to_domain_image_2d
            self.path_to_domain_labels_main = self.path_to_domain_labels_2d


        elif self.dim == 3:
            self.path_to_domain_mesh_vtu = self.data.create_path(processing=self.steps_sub_path_map['domain_prep'],
                                                                 datasource='domain',
                                                                 datatype='vtu', content='mesh', extension='vtu')

            self._mesh_3d_domain(path_to_image=self.path_to_domain_labels_3d,
                                 path_to_mesh=self.path_to_domain_mesh_vtu,
                                 tissues_dict=None)
            # convert to fenics
            mesh_fenics, subdomains = dio.read_vtk_convert_to_fenics(self.path_to_domain_mesh_vtu)
            dio.save_mesh_hdf5(mesh_fenics, self.path_to_domain_meshfct,
                               subdomains=subdomains)

            self.path_to_domain_meshfct_main = self.path_to_domain_meshfct
            self.path_to_domain_image_main = self.path_to_domain_image_3d
            self.path_to_domain_labels_main = self.path_to_domain_labels_3d

        self.logger.info("path_to_domain_meshfct_main: %s" % self.path_to_domain_meshfct_main)
        self.logger.info("path_to_domain_image_main:   %s" % self.path_to_domain_image_main)
        self.logger.info("path_to_domain_labels_main:  %s" % self.path_to_domain_labels_main)
        self._save_state()

    def _init_problem(self, path_to_domain, seed_position, sim_params, model_params_varying, model_params_fixed,
                      problem_type='forward', save_params=True, **kwargs):
        params_dict = {}
        if problem_type in ['forward', 'inverse', 'optimized']:
            params_dict['path_to_domain'] = path_to_domain
            params_dict['model_params_varying'] = model_params_varying
            params_dict['sim_params'] = sim_params
            params_dict['seed_position'] = seed_position
            params_dict['model_params_fixed'] = model_params_fixed
            for key, value in kwargs.items():
                self.logger.info("Adding additional parameters '%s' with value '%s'"%(key, str(value)))
                params_dict[key] = value
            self.logger.info("Parameters for '%s' problem: %s"%(problem_type, params_dict))
            setattr(self, "params_%s" % problem_type, params_dict.copy())
        else:
            self.logger.error("Invalid problem_type '%s'!" % problem_type)
        # -- Initialisation
        mesh, subdomains, boundaries = dio.read_mesh_hdf5(path_to_domain)
        # initial values
        if len(seed_position) == 2 and self.dim == 2:
            u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))', degree=1,
                                              a=0.5, x0=seed_position[0], y0=seed_position[1])
        elif len(seed_position) == 3 and self.dim == 3:
            u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2)  - a*pow(x[2]-z0, 2))',
                                              degree=1,
                                              a=0.5, x0=seed_position[0], y0=seed_position[1], z0=seed_position[2])
        else:
            u_0_conc_expr = fenics.Constant(0)
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
        # logger_names = ['simulation', 'simulation_base', 'simulation.helpers.helper_classes', 'simulation.helpers']
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
        if save_params:
            self._save_problem_run_params(problem_type)
        return sim

    def _save_problem_run_params(self, problem_type):
        if problem_type == 'forward':
            save_path = self.path_run_params_forward
        elif problem_type == 'inverse':
            save_path = self.path_run_params_inverse
        elif problem_type == 'optimized':
            save_path = self.path_run_params_optimized
        else:
            self.logger.error("Non existing 'problem type'")

        attribute_name = "params_%s" % problem_type
        if hasattr(self, attribute_name):
            params_dict = getattr(self, attribute_name)
            with open(save_path, 'wb') as handle:
                pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.logger.warning("Parameters for problem '%s' do not exist" % problem_type)

    def _rebase(self, old_path):
        # TODO this is specific to use of this class with parametric study
        new_base = self.base_dir
        p_common = os.path.commonpath([new_base, old_path])
        if p_common != new_base:
            new_base_split = self.splitall(new_base)
            ref_subdir = new_base_split[-1]
            rel_path = self.get_path_relative_to_subdir(old_path, ref_subdir)
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
            self.logger.info("Loading parameter file for '%s' problem: %s"%(problem_type, save_path))
            with open(save_path, "rb") as f:
                param_dict = pickle.load(f)
            # check path_to_domain
            new_dir = self._rebase(param_dict['path_to_domain'])
            param_dict['path_to_domain'] = new_dir
            attribute_name = "params_%s" % problem_type
            setattr(self, attribute_name, param_dict.copy())
            self.logger.info("Read parameters for problem '%s': %s" % (problem_type, param_dict))
        else:
            self.logger.warning("Cannot read parameters for problem '%s'" % problem_type)
            self.logger.warning("File '%s' does not exist" % save_path)

    def init_forward_problem(self, seed_position, model_params_varying, sim_params,
                             model_params_fixed, path_to_domain=None):
        self.path_forward_sim = self.data.create_path(processing=self.steps_sub_path_map['forward_sim'])
        if path_to_domain is None:  # here we may use full or reduced domain as default
            path_to_domain = self.path_to_domain_meshfct_main
        self.sim_forward = self._init_problem(path_to_domain, seed_position,
                                              sim_params, model_params_varying, model_params_fixed,
                                              problem_type='forward')

    def init_inverse_problem(self, seed_position, model_params_varying, sim_params,
                             model_params_fixed=None, path_to_domain=None, seed_from_com=False,
                             optimization_type=5):
        self.path_inverse_sim = self.data.create_path(processing=self.steps_sub_path_map['inverse_sim'])
        if path_to_domain is None:  # this should always be the reduced domain, unless specified otherwise
            path_to_domain = self.path_to_domain_meshfct_main
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
        self.sim_inverse = self._init_problem(path_to_domain,
                                              seed_position,
                                              sim_params,
                                              model_params_varying,
                                              model_params_fixed,
                                              problem_type='inverse',
                                              optimization_type=optimization_type)

    def init_optimized_problem(self):
        self.path_optimized_sim = self.data.create_path(processing=self.steps_sub_path_map['optimized_sim'])
        model_params_varying = self.params_inverse['model_params_varying'].copy()
        self.logger.info("Params Varying (init): %s"% model_params_varying)
        model_params_varying.update(self.model_params_optimized)
        self.logger.info("Params Varying (updates with opt results): %s"% model_params_varying)
        self.sim_optimized = self._init_problem(self.params_inverse['path_to_domain'],
                                                self.params_inverse['seed_position'],
                                                self.params_inverse['sim_params'],
                                                model_params_varying,
                                                self.params_inverse['model_params_fixed'],
                                                problem_type='optimized',
                                                optimization_type=self.params_inverse['optimization_type'])

    def run_forward_sim(self, plot=None):
        if plot is None:
            plot = self.plot
        self.sim_forward.run(save_method=None, plot=False, output_dir=self.path_forward_sim, clear_all=True)
        # save results
        disp_target_0, conc_target_0 = fenics.split(self.sim_forward.solution.copy())
        conc_target = self.sim_forward.functionspace.project_over_space(conc_target_0, subspace_id=1)
        disp_target = self.sim_forward.functionspace.project_over_space(disp_target_0, subspace_id=0)
        # save functions on reduced domain
        self.path_forward_conc = self.data.create_path(processing=self.steps_sub_path_map['forward_sim'],
                                                       datasource='simulation', domain=None,
                                                       frame='reference', datatype='fenics',
                                                       content='conc', extension='h5')
        self.path_forward_disp = self.data.create_path(processing=self.steps_sub_path_map['forward_sim'],
                                                       datasource='simulation', domain=None,
                                                       frame='reference', datatype='fenics',
                                                       content='disp', extension='h5')

        dio.save_function_mesh(conc_target, self.path_forward_conc, labelfunction=None,
                               subdomains=self.sim_forward.subdomains.subdomains)
        dio.save_function_mesh(disp_target, self.path_forward_disp, labelfunction=None,
                               subdomains=self.sim_forward.subdomains.subdomains)
        # Save as VTK & Plot
        self.sim_forward.init_postprocess(self.path_forward_sim)
        selection = [1, self.params_forward['sim_params']['sim_time']]
        self.sim_forward.postprocess.save_all(save_method='vtk', clear_all=False, selection=selection)
        if fenics.is_version("<2018.1.x") and plot and self.dim == 2:
            self.sim_forward.postprocess.plot_all(deformed=False, selection=selection,
                                                  output_dir=os.path.join(self.path_forward_sim, 'plots'))
            self.sim_forward.postprocess.plot_all(deformed=True, selection=selection,
                                                  output_dir=os.path.join(self.path_forward_sim, 'plots'))
            self.sim_forward.postprocess.plot_for_pub(deformed=True, selection=selection,
                                                      output_dir=os.path.join(self.path_forward_sim, 'plots_for_pub'))
        self._save_state()


    def run_optimized_sim(self, plot=None):
        if plot is None:
            plot = self.plot
        fu.ensure_dir_exists(self.path_optimized_sim)
        # run standard forward simulation -- inccomplete parametervalues when optimization on < 5 params
        # self.sim_optimized.run(save_method=None, plot=False, output_dir=self.path_optimized_sim, clear_all=True)

        # run using adjoint call
        params_names, solver_function = self.map_optimization_type(self.sim_optimized,
                                                                   self.params_optimized['optimization_type'])
        params_init_values = [self.model_params_optimized[name] for name in params_names]
        solver_function(params_init_values, output_dir=self.path_optimized_sim)

        # save results
        disp_target_0, conc_target_0 = fenics.split(self.sim_optimized.solution)
        conc_target = self.sim_optimized.functionspace.project_over_space(conc_target_0, subspace_id=1)
        disp_target = self.sim_optimized.functionspace.project_over_space(disp_target_0, subspace_id=0)
        # save functions on reduced domain
        self.path_optimized_conc = self.data.create_path(processing=self.steps_sub_path_map['optimized_sim'],
                                                         datasource='simulation', domain=None,
                                                         frame='reference', datatype='fenics',
                                                         content='conc', extension='h5')
        self.path_optimized_disp = self.data.create_path(processing=self.steps_sub_path_map['optimized_sim'],
                                                         datasource='simulation', domain=None,
                                                         frame='reference', datatype='fenics',
                                                         content='disp', extension='h5')

        dio.save_function_mesh(conc_target, self.path_optimized_conc, labelfunction=None,
                               subdomains=self.sim_optimized.subdomains.subdomains)
        dio.save_function_mesh(disp_target, self.path_optimized_disp, labelfunction=None,
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
                                                        output_dir=os.path.join(self.path_optimized_sim,
                                                                                'plots_for_pub'))
        self._save_state()

    def eval_cb_post(self, j, a):
        values = [param.values()[0] for param in a]
        result = (j, *values)
        self.opt_param_progress_post.append(result)
        self.opt_date_time.append((j, datetime.now()))
        self.logger.info("optimization eval: %s" % (''.join(str(e) for e in result)))

    def derivative_cb_post(self, j, dj, m):
        dj_values = [param.values()[0] for param in dj]
        result = (j, *dj_values)
        self.opt_dj_progress_post.append(result)
        self.logger.info("optimization derivative: %s" % (''.join(str(e) for e in result)))

    @staticmethod
    def create_opt_progress_df(opt_param_list, opt_dj_list, param_name_list, datetime_list):
        columns_params = ['J', *param_name_list]
        columns_dJ = ['J', *['dJd%s' % param for param in param_name_list]]
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

    def custom_optimizer(self, J, m_global, dJ, H, bounds, **kwargs):
        self.logger.info("-- Starting optimization")
        try:
            opt_res = scipy_minimize(J, m_global, bounds=bounds, **kwargs)
            self.logger.info("-- Finished Optimization")
            for name, item in opt_res.items():
                self.logger.info("  - %s: %s" % (name, item))
                if name not in ['hess_inv']:
                    self.measures["optimization_%s" % name] = item
            return np.array(opt_res["x"])
        except Exception as e:
            self.logger.error("Error in optimization:")
            self.logger.error(e)

    def run_inverse_problem_n_params(self, params_init_values, params_names, solver_function,
                                     opt_params=None, **kwargs):
        params_init = [fenics.Constant(param) for param in params_init_values]
        # first run
        u = solver_function(params_init, **kwargs)

        # simulated fields
        disp_opt, conc_opt = fenics.split(u)
        disp_opt_proj = self.sim_inverse.functionspace.project_over_space(disp_opt, subspace_id=0)
        conc_opt_proj_T2 = self.sim_inverse.functionspace.project_over_space(self.thresh(conc_opt,
                                                                                         self.conc_threshold_levels[
                                                                                             'T2']), subspace_id=1)
        conc_opt_proj_T1 = self.sim_inverse.functionspace.project_over_space(self.thresh(conc_opt,
                                                                                         self.conc_threshold_levels[
                                                                                             'T1']), subspace_id=1)

        # target fields
        conc_target_thr_T2 = dio.read_function_hdf5('function',
                                                    self.sim_inverse.functionspace.get_functionspace(subspace_id=1),
                                                    self.path_conc_T2)
        conc_target_thr_T1 = dio.read_function_hdf5('function',
                                                    self.sim_inverse.functionspace.get_functionspace(subspace_id=1),
                                                    self.path_conc_T1)
        disp_target = dio.read_function_hdf5('function',
                                             self.sim_inverse.functionspace.get_functionspace(subspace_id=0),
                                             self.path_displacement_reconstructed)

        # optimization functional
        function_expr = fenics.inner(conc_opt_proj_T2 - conc_target_thr_T2,
                                     conc_opt_proj_T2 - conc_target_thr_T2) * self.sim_inverse.subdomains.dx \
                        + fenics.inner(conc_opt_proj_T1 - conc_target_thr_T1,
                                       conc_opt_proj_T1 - conc_target_thr_T1) * self.sim_inverse.subdomains.dx \
                        + fenics.inner(disp_opt_proj - disp_target,
                                       disp_opt_proj - disp_target) * self.sim_inverse.subdomains.dx

        if fenics.is_version("<2018.1.x"):
            J = fenics.Functional(function_expr)
        else:
            J = fenics.assemble(function_expr)

        controls = [fenics.Control(param) for param in params_init]

        # optimization
        # -- for keeping progress info
        self.opt_param_progress_post = []
        self.opt_dj_progress_post = []
        self.opt_date_time = []

        rf = fenics.ReducedFunctional(J, controls, eval_cb_post=self.eval_cb_post,
                                      derivative_cb_post=self.derivative_cb_post)

        # -- optimization parameters
        bounds_min = [0.005 for param in params_init]
        bounds_max = [0.5 for param in params_init]
        bounds_int = [bounds_min, bounds_max]
        params = {'bounds': bounds_int,
                  'method': "L-BFGS-B",
                  'tol': 1e-6,
                  'options': {'disp': True, 'gtol': 1e-6}}
        if opt_params:
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
            self.logger.info("  - %s = %f" % (name, var.values()[0]))
        self.model_params_optimized = params_dict

        self.logger.info("Writing optimized simulation parameters to '%s'" % self.path_parameters_optimized)
        # -- save optimized parameters
        with open(self.path_parameters_optimized, 'wb') as handle:
            pickle.dump(params_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        opt_df = self.create_opt_progress_df(self.opt_param_progress_post,
                                             self.opt_dj_progress_post,
                                             params_names,
                                             datetime_list=self.opt_date_time)

        self.path_optimization_progress_pkl = self.data.create_params_path(
            processing=self.steps_sub_path_map['inverse_sim'],
            datasource='optimization_progress')
        self.path_optimization_progress_xls = self.data.create_params_path(
            processing=self.steps_sub_path_map['inverse_sim'],
            datasource='optimization_progress', extension='xls')
        self.optimization_progress = opt_df
        self.logger.info(opt_df)
        opt_df.to_excel(self.path_optimization_progress_xls)
        opt_df.to_pickle(self.path_optimization_progress_pkl)

        if fenics.is_version(">2017.2.x"):
            self.sim_inverse.tape.visualise()

        self._save_state()


    def map_optimization_type(self, sim_instance, optimization_type):
        if optimization_type==2:
            params_names = ["D_WM", "rho_WM"]
            solver_function = sim_instance.run_for_adjoint_2params
        elif optimization_type==3:
            params_names = ["D_WM", "rho_WM", "coupling"]
            solver_function = sim_instance.run_for_adjoint_3params
        elif optimization_type==4:
            params_names = ["D_WM", "D_GM", "rho_WM", "coupling"]
            solver_function = sim_instance.run_for_adjoint_4params
        elif optimization_type==5:
            params_names = ["D_WM", "D_GM", "rho_WM", "rho_GM", "coupling"]
            solver_function = sim_instance.run_for_adjoint_5params
        return params_names, solver_function


    def run_inverse_problem(self, opt_params=None):
        optimization_type = self.params_inverse['optimization_type']
        self.logger.info("Running inverse optimization for optimization type '%s'"%str(optimization_type))
        params_names, solver_function = self.map_optimization_type(self.sim_inverse, optimization_type)
        params_init_values = [self.params_inverse["model_params_varying"][name] for name in params_names]
        self.run_inverse_problem_n_params(params_init_values, params_names,
                                          solver_function=solver_function,
                                          opt_params=opt_params, output_dir=self.path_inverse_sim)


    def run_inverse_problem_2params(self, opt_params=None):
        params_names = ["D_WM", "rho_WM"]
        params_init_values = [self.params_inverse["model_params_varying"][name] for name in params_names]
        self.run_inverse_problem_n_params(params_init_values, params_names,
                                          solver_function=self.sim_inverse.run_for_adjoint_2params,
                                          opt_params=opt_params, output_dir=self.path_inverse_sim)

    def run_inverse_problem_3params(self, opt_params=None):
        params_names = ["D_WM", "rho_WM", "coupling"]
        params_init_values = [self.params_inverse["model_params_varying"][name] for name in params_names]
        self.run_inverse_problem_n_params(params_init_values, params_names,
                                          solver_function=self.sim_inverse.run_for_adjoint_3params,
                                          opt_params=opt_params, output_dir=self.path_inverse_sim)

    def run_inverse_problem_4params(self, opt_params=None):
        params_names = ["D_WM", "D_GM", "rho_WM", "coupling"]
        params_init_values = [self.params_inverse["model_params_varying"][name] for name in params_names]
        self.run_inverse_problem_n_params(params_init_values, params_names,
                                          solver_function=self.sim_inverse.run_for_adjoint_4params,
                                          opt_params=opt_params, output_dir=self.path_inverse_sim)

    def run_inverse_problem_5params(self, opt_params=None):
        params_names = ["D_WM", "D_GM", "rho_WM", "rho_GM", "coupling"]
        params_init_values = [self.params_inverse["model_params_varying"][name] for name in params_names]
        self.run_inverse_problem_n_params(params_init_values, params_names,
                                          solver_function=self.sim_inverse.run_for_adjoint,
                                          opt_params=opt_params, output_dir=self.path_inverse_sim)

    def _reload_model_sim(self, problem_type='forward'):
        self.logger.info("=== Reloading simulation '%s'." % problem_type)
        param_attr_name = "params_%s" % problem_type
        if hasattr(self, param_attr_name):
            path_map_name = problem_type + '_sim'
            try:
                data_path = os.path.join(self.steps_path_map[path_map_name], 'solution_timeseries.h5')
                if os.path.exists(data_path):
                    params = getattr(self, param_attr_name)
                    sim = self._init_problem(problem_type=problem_type, save_params=False, **params)
                    sim.reload_from_hdf5(data_path)
                    setattr(self, "sim_%s" % problem_type, sim)
            except:
                self.logger.error("Non existing 'problem type'")
        else:
            self.logger.warning("Parameter attribute for simulation '%s' does not exist" % problem_type)
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
                self.params_optimization = pickle.load(f)
        else:
            self.logger.warning("Cannot load 'params_optimization', %s does not exist" % self.path_optimization_params)
        # parameter results from optimization process
        if os.path.exists(self.path_parameters_optimized):
            with open(self.path_parameters_optimized, "rb") as f:
                self.model_params_optimized = pickle.load(f)
        else:
            self.logger.warning(
                "Cannot load 'model_params_optimized', %s does not exist" % self.path_parameters_optimized)

    def _reload_optimized_sim(self):
        self._read_problem_run_params(problem_type='optimized')
        self._reload_model_sim(problem_type='optimized')

    def reload_state(self):
        self.logger.info("")
        self._reload_forward_sim()
        self._reload_inverse_sim()
        self._reload_optimized_sim()
        if os.path.exists(self.path_to_measures):
            with open(self.path_to_measures, "rb") as f:
                self.measures = pickle.load(f)

    def _create_deformed_image(self, path_image_ref, output_path, path_image_warped):
        # 1) Load reference image
        ref_image = sitk.ReadImage(path_image_ref)
        ref_image_size = ref_image.GetSize()

        # 2) Convert simulation result to labelmap with predefined resolution
        if self.dim == 2:
            resolution = (*ref_image_size, 1)
        else:
            resolution = ref_image_size

        print('resolution: ', resolution)
        # get vtu file
        name_sim_vtu = dio.create_file_name('all', self.params_forward['sim_params']["sim_time"])
        path_to_sim_vtu = os.path.join(self.steps_path_map['forward_sim'], 'merged', name_sim_vtu)
        sim_vtu = vtu.read_vtk_data(path_to_sim_vtu)
        print("simulation file name: %s" % path_to_sim_vtu)
        print("sim vtu", sim_vtu)
        # convert vtu to vti
        sim_vti = vtu.resample_to_image(sim_vtu, resolution)
        path_to_sim_vti = os.path.join(output_path, 'simulation_as_image.vti')
        vtu.write_vtk_data(sim_vti, path_to_sim_vti)
        # convert vti to normal image
        label_img = vtu.convert_vti_to_img(sim_vti, array_name='label_map', RGB=False)
        path_label_img = os.path.join(output_path, 'label_img_orig.nii')
        sitk.WriteImage(label_img, path_label_img)

        # 3) Create deformed labelmap
        sim_vtu_warped = vtu.warpVTU(sim_vtu, 'point', 'displacement')
        sim_vti_warped = vtu.resample_to_image(sim_vtu_warped, resolution)
        path_to_sim_vti_warped = os.path.join(output_path, 'simulation_as_image_warped.vti')
        vtu.write_vtk_data(sim_vti_warped, path_to_sim_vti_warped)

        path_labels_warped_from_vti = os.path.join(output_path, 'labels_warped_from_vti.nii')
        label_img_warped = vtu.convert_vti_to_img(sim_vti_warped, array_name='label_map', RGB=False)
        sitk.WriteImage(label_img_warped, path_labels_warped_from_vti)

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
        reg.ants_apply_transforms(input_img=path_image_ref, output_file=path_image_warped,
                                  reference_img=path_image_ref,
                                  transforms=[path_disp_img_inv], dim=self.dim)

        # - 2) resample label map to T1
        output_label_resampled = os.path.join(output_path, 'label_img_resampledToT1.nii')
        reg.ants_apply_transforms(input_img=path_label_img, output_file=output_label_resampled,
                                  reference_img=path_image_ref,
                                  transforms=[], dim=self.dim)

        # - 3) resample label map to T1
        output_label_resampled_warped = os.path.join(output_path, 'label_img_resampledToT1_warped.nii')
        reg.ants_apply_transforms(input_img=path_label_img, output_file=output_label_resampled_warped,
                                  reference_img=path_image_ref,
                                  transforms=[path_disp_img_inv], dim=self.dim)

    def _reconstruct_deformation_field(self, path_to_reference_image, path_to_deformed_image, path_to_warp_field,
                                       path_to_reduced_domain=None, plot=None):
        self.logger.info("path_to_reference_image:              %s" % path_to_reference_image)
        self.logger.info("path_to_deformed_image:               %s" % path_to_deformed_image)
        self.logger.info("path_to_warp_field:                   %s" % path_to_warp_field)
        self.logger.info("path_to_reduced_domain:               %s" % path_to_reduced_domain)

        self.path_displacement_reconstructed = self.data.create_fenics_path(
            processing=self.steps_sub_path_map['target_fields'],
            datasource='registration',
            content='disp', frame='reference')
        if plot is None:
            plot = self.plot
        # -- registration to obtain displacement field
        self.path_to_warped_image_deformed_to_ref = os.path.join(self.path_target_fields,
                                                                 'registered_image_deformed_to_reference')
        reg.register_ants(path_to_reference_image, path_to_deformed_image, self.path_to_warped_image_deformed_to_ref,
                          path_to_transform=path_to_warp_field, registration_type='Syn',
                          image_ext='nii', fixed_mask=None, moving_mask=None, verbose=1, dim=self.dim)

        # -- read registration, convert to fenics function, save
        image_warp = sitk.ReadImage(path_to_warp_field)
        self.logger.info("== Transforming image to fenics function ... this is very slow...")
        f_img = dio.create_fenics_function_from_image(image_warp)

        if path_to_reduced_domain:
            f_img = self._map_field_into_reduced_domain(f_img, path_to_reduced_domain, degree=1, function_type='vector')

        dio.save_function_mesh(f_img, self.path_displacement_reconstructed)
        if plot:
            plott.show_img_seg_f(function=f_img, show=False,
                                 path=os.path.join(self.path_target_fields,
                                                   'displacement_from_registration_fenics.png'),
                                 dpi=300)

        self._save_state()

    def _warp_deformed_to_reference(self, path_to_deformed_img, path_to_img_ref_frame,
                                    path_to_deformed_labels, path_to_labels_ref_frame,
                                    path_to_warp_field,
                                    path_to_img_ref_frame_fct, path_to_labels_ref_frame_fct,
                                    path_to_meshfct_ref_frame,
                                    plot=True):

        self.logger.info("path_to_deformed_img:               %s" % path_to_deformed_img)
        self.logger.info("path_to_img_ref_frame:              %s" % path_to_img_ref_frame)
        self.logger.info("path_to_deformed_labels:            %s" % path_to_deformed_labels)
        self.logger.info("path_to_labels_ref_frame:           %s" % path_to_labels_ref_frame)
        self.logger.info("path_to_warp_field:                 %s" % path_to_warp_field)
        self.logger.info("path_to_img_ref_frame_fct:          %s" % path_to_img_ref_frame_fct)
        self.logger.info("path_to_labels_ref_frame_fct:       %s" % path_to_labels_ref_frame_fct)
        self.logger.info("path_to_meshfct_ref_frame:          %s" % path_to_meshfct_ref_frame)

        # warp patient data to reference
        reg.ants_apply_transforms(input_img=path_to_deformed_img,
                                  output_file=path_to_img_ref_frame,
                                  reference_img=path_to_deformed_img,
                                  transforms=[path_to_warp_field], dim=self.dim)

        reg.ants_apply_transforms(input_img=path_to_deformed_labels,
                                  output_file=path_to_labels_ref_frame,
                                  reference_img=path_to_deformed_labels,
                                  transforms=[path_to_warp_field], dim=self.dim,
                                  interpolation='GenericLabel')

        # create fields
        # -- load patient labels
        image_label = sitk.Cast(sitk.ReadImage(path_to_labels_ref_frame), sitk.sitkUInt8)
        if self.dim == 2:
            f_img_label = dio.image2fct2D(image_label)
        else:
            self.logger.error("cannot convert 3d image to fenics function")
        f_img_label.rename("label", "label")

        # -- load patient image
        image = sitk.ReadImage(path_to_img_ref_frame)
        if self.dim == 2:
            f_img = dio.image2fct2D(image)
        else:
            self.logger.error("cannot convert 3d image to fenics function")
        f_img.rename("imgvalue", "label")

        # -- plot
        if plot:
            plott.show_img_seg_f(image=image, segmentation=image_label, show=True,
                                 path=os.path.join(self.path_target_fields, 'label_from_sitk_image_in_ref_frame.png'))

            plott.show_img_seg_f(function=f_img_label, show=True,
                                 path=os.path.join(self.path_target_fields,
                                                   'label_from_fenics_function_in_ref_frame.png'))

            plott.show_img_seg_f(function=f_img, show=True,
                                 path=os.path.join(self.path_target_fields, 'from_fenics_function_in_ref_frame.png'))

        # == save
        # -- save label function:
        dio.save_function_mesh(f_img, path_to_img_ref_frame_fct)
        dio.save_function_mesh(f_img_label, path_to_labels_ref_frame_fct)

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

        dio.save_mesh_hdf5(mesh, path_to_meshfct_ref_frame, subdomains=subdomains.subdomains, boundaries=None)

    def create_thresholded_conc_field(self, path_to_conc_in, threshold,
                                      path_to_reduced_domain=None, path_to_conc_out=None):
        self.logger.info("Creating thresholded concentration field:")
        self.logger.info(" - input field: '%s'" % path_to_conc_in)
        self.logger.info(" - threshold: '%f'" % threshold)
        self.logger.info(" - domain reduced: '%s'" % path_to_reduced_domain)
        self.logger.info(" - iutput field: '%s'" % path_to_conc_out)

        conc_sim, mesh_sim, subdomains_sim, boundaries_sim = dio.load_function_mesh(path_to_conc_in,
                                                                                    functionspace='function', degree=2)
        conc_thresh = fenics.project(self.thresh(conc_sim, threshold), conc_sim.function_space(), annotate=False)

        f_conc = conc_thresh.copy(deepcopy=True)
        f_conc_np = f_conc.vector().get_local()
        f_conc_np[f_conc_np < 0.5] = 0
        f_conc.vector().set_local(f_conc_np)

        if path_to_reduced_domain:
            f_conc = self._map_field_into_reduced_domain(f_conc, path_to_reduced_domain, degree=2)
        if path_to_conc_out:
            dio.save_function_mesh(f_conc, path_to_conc_out)
        return f_conc

    def _map_field_into_reduced_domain(self, field_in, path_to_reduced_domain, function_type='function', degree=2):

        self.logger.info("Mapping field into reduced domain:")
        self.logger.info(" - field: '%s'" % field_in)
        self.logger.info(" - path to reduced domain: '%s'" % path_to_reduced_domain)
        mesh_reduced, subdomains_reduced, boundaries_reduced = dio.read_mesh_hdf5(path_to_reduced_domain)
        if function_type == 'function':
            funspace_reduced = fenics.FunctionSpace(mesh_reduced, 'Lagrange', degree)
        elif function_type == 'vector':
            funspace_reduced = fenics.VectorFunctionSpace(mesh_reduced, 'Lagrange', degree)
        else:
            self.logger.error("Don't understand function_type %s'" % function_type)
        field_reduced = self.interpolate_non_matching(field_in, funspace_reduced)
        return field_reduced

    def create_thresholded_conc_fields(self, path_conc_field, path_to_reduced_domain=None, plot=None):

        self.logger.info("Creating thresholded concentration fields:")
        self.logger.info(" - input field: '%s'" % path_conc_field)
        self.logger.info(" - domain reduced: '%s'" % path_to_reduced_domain)

        if plot is None:
            plot = self.plot

        self.path_conc_T2 = self.data.create_fenics_path(processing=self.steps_sub_path_map['target_fields'],
                                                         datasource='domain',
                                                         datatype='fenics',
                                                         content='conc_T2')

        self.path_conc_T1 = self.data.create_fenics_path(processing=self.steps_sub_path_map['target_fields'],
                                                         datasource='domain',
                                                         datatype='fenics',
                                                         content='conc_T1')

        conc_thresh_T2 = self.create_thresholded_conc_field(path_to_conc_in=path_conc_field,
                                                            threshold=self.conc_threshold_levels['T2'],
                                                            path_to_conc_out=self.path_conc_T2,
                                                            path_to_reduced_domain=path_to_reduced_domain)

        conc_thresh_T1 = self.create_thresholded_conc_field(path_to_conc_in=path_conc_field,
                                                            threshold=self.conc_threshold_levels['T1'],
                                                            path_to_conc_out=self.path_conc_T1,
                                                            path_to_reduced_domain=path_to_reduced_domain)

        if plot:
            plott.show_img_seg_f(function=conc_thresh_T2, show=False,
                                 path=os.path.join(self.path_target_fields, 'thresholded_concentration_forward_T2.png'),
                                 dpi=300)
            plott.show_img_seg_f(function=conc_thresh_T1, show=False,
                                 path=os.path.join(self.path_target_fields, 'thresholded_concentration_forward_T1.png'),
                                 dpi=300)

        self._save_state()

    def create_conc_fields_from_segmentation(self, path_to_label_function, path_to_conc_field_out,
                                             plot=None, T1_label=5, T2_label=6):
        self.logger.info("Creating concentration fields from segmentation:")
        self.logger.info(" - label function: '%s'" % path_to_label_function)
        self.logger.info(" - extracted concentration field: '%s'" % path_to_conc_field_out)
        if plot is None:
            plot = self.plot

        # reduce domain
        f_label_patient, mesh, subdomains, boundaries = dio.load_function_mesh(path_to_label_function,
                                                                               functionspace='function', degree=1)
        V_quad = fenics.FunctionSpace(mesh, "Lagrange", 2)
        f_label_quad = fenics.interpolate(f_label_patient, V_quad)

        f_conc = f_label_quad.copy(deepcopy=True)

        f_conc_np = f_conc.vector().get_local()
        # f_conc_np = float(f_conc_np)
        f_conc_np[np.isclose(f_conc_np, T1_label, 0.1)] = self.conc_threshold_levels['T1'] + 0.2  # label 5 -> T1
        f_conc_np[np.isclose(f_conc_np, T2_label, 0.1)] = self.conc_threshold_levels['T2'] + 0.2
        print(np.unique(f_conc_np))
        f_conc.vector().set_local(f_conc_np)

        dio.save_function_mesh(f_conc, path_to_conc_field_out)

        if plot:
            plott.show_img_seg_f(function=f_conc, show=False,
                                 path=os.path.join(self.path_target_fields,
                                                   'concentration_field_from_segmentation.png'),
                                 dpi=300)

    def get_seed_from_com(self):
        target_conc_01, mesh, subdomains, boundaries = dio.load_function_mesh(self.path_conc_T2,
                                                                              functionspace='function', degree=1)
        com = self.compute_com(target_conc_01, domain=None)
        return com

    def plot_domain_with_seed(self, seed_from='com'):
        # domain with seed
        if seed_from == 'com':
            seed_position = self.get_seed_from_com()
        elif seed_from == 'seed':
            seed_position = self.sim_forward.params['seed_position']
        else:
            self.logger.error("'seed_from' must have value 'com' or 'seed'; current value: '%s'" % seed_from)

        domainn_image = sitk.ReadImage(self.path_to_domain_image_main)
        atlas_labels = sitk.ReadImage(self.path_to_domain_labels_main)

        target_conc_08, mesh, subdomains, boundaries = dio.load_function_mesh(self.path_conc_T1,
                                                                              functionspace='function', degree=2)
        if len(seed_position) == 2:
            u_0_conc_expr = fenics.Expression('exp(-a*pow(x[0]-x0, 2) - a*pow(x[1]-y0, 2))', degree=1,
                                              a=0.5, x0=seed_position[0], y0=seed_position[1])
            plott.show_img_seg_f(image=domainn_image, segmentation=atlas_labels,
                                 function=fenics.project(u_0_conc_expr, target_conc_08.ufl_function_space()),
                                 showmesh=False, alpha_f=1,
                                 range_f=[0.001, 1.01], exclude_min_max=True, colormap='viridis',
                                 path=os.path.join(self.steps_path_map['plots'], 'sim_domain_with_seed.png'))

    def plot_domain_target_fields_conc(self):
        # from domain with target fields
        target_conc_T2, mesh, subdomains, boundaries = dio.load_function_mesh(self.path_conc_T2,
                                                                              functionspace='function', degree=2)
        target_conc_T1, mesh, subdomains, boundaries = dio.load_function_mesh(self.path_conc_T1,
                                                                              functionspace='function', degree=2)

        mesh, subdomains, boundaries = dio.read_mesh_hdf5(self.path_to_domain_meshfct_main)
        subdomains_fct = vh.convert_meshfunction_to_function(mesh, subdomains)

        plot_obj = {'object': target_conc_T2,
                    'alpha': 0.5}
        plott.show_img_seg_f(function=subdomains_fct, add_plot_object_post=plot_obj,
                             path=os.path.join(self.steps_path_map['plots'], 'sim_domain_with_target_conc_T2.png'))
        plot_obj = {'object': target_conc_T1,
                    'alpha': 0.5}
        plott.show_img_seg_f(function=subdomains_fct, add_plot_object_post=plot_obj,
                             path=os.path.join(self.steps_path_map['plots'], 'sim_domain_with_target_conc_T1.png'))

    def plot_domain_target_fields_disp(self):
        domain_image = sitk.ReadImage(self.path_to_domain_image_main)
        atlas_labels = sitk.ReadImage(self.path_to_domain_labels_main)

        target_disp, _mesh, _subdomains, _boundaries = dio.load_function_mesh(
            self.path_displacement_reconstructed,
            functionspace='vector')
        plott.show_img_seg_f(image=domain_image, segmentation=atlas_labels, function=target_disp,
                             showmesh=False, alpha_f=1, exclude_min_max=True,
                             path=os.path.join(self.steps_path_map['plots'], 'sim_domain_with_target_disp.png'))

        target_disp_x = fenics.dot(target_disp, fenics.Expression(('1.0', '0.0'), degree=1))
        target_disp_y = fenics.dot(target_disp, fenics.Expression(('0.0', '1.0'), degree=1))

        V = fenics.FunctionSpace(_mesh, 'Lagrange', 1)
        plott.show_img_seg_f(image=domain_image, segmentation=atlas_labels,
                             function=fenics.project(target_disp_x, V),
                             showmesh=False, alpha_f=0.8, exclude_min_max=True,
                             path=os.path.join(self.steps_path_map['plots'], 'sim_domain_with_target_disp_x.png'))
        plott.show_img_seg_f(image=domain_image, segmentation=atlas_labels,
                             function=fenics.project(target_disp_y, V),
                             showmesh=False, alpha_f=0.8, exclude_min_max=True,
                             path=os.path.join(self.steps_path_map['plots'], 'sim_domain_with_target_disp_y.png'))

    def plot_all(self):
        self.plot_domain_target_fields_conc()
        self.plot_domain_target_fields_disp()

    def write_analysis_summary(self, add_info_list=[]):
        summary = {}
        #summary.update(self.flatten_params(self.params_forward, 'forward'))
        summary.update(self.flatten_params(self.params_inverse, 'inverse'))
        summary.update(self.flatten_params(self.params_optimized, 'optimized'))
        summary.update(self.measures)
        for item in add_info_list:
            summary.update(item)
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

    def compute_volume_thresholded(self):
        vol_dict = {'volume_threshold_T2_target': self.path_conc_T2,
                    'volume_threshold_T1_target': self.path_conc_T1}
        if hasattr(self, 'sim_inverse'):
            sim = self.sim_inverse
            for name, path in vol_dict.items():
                if os.path.exists(path):
                    self.logger.info("Computing Volume for '%s'" % (name))
                    conc = dio.read_function_hdf5('function', sim.functionspace.get_functionspace(subspace_id=1), path)
                    vol = self.compute_volume(conc, sim.subdomains.dx)
                    self.measures[name] = vol
                else:
                    self.logger.warning("Cannot compute volume. '%s' does not exist" % path)
            self._save_state()
        else:
            self.logger.warning("Cannot compute volume. Instance 'sim_inverse' does not exist")

    def compute_com_all(self, conc_dict=None):
        field_dict = {'threshold_T2_target': self.path_conc_T2,
                      'threshold_T1_target': self.path_conc_T1,
                      'inverse': self.path_optimized_conc}
        if conc_dict is not None:
            field_dict.update(conc_dict)
        if hasattr(self, 'sim_forward'):
            for name, path in field_dict.items():
                if name.startswith('forward') and hasattr(self, 'sim_forward'):
                    funspace = self.sim_forward.functionspace.get_functionspace(subspace_id=1)
                    domain = self.sim_forward.subdomains.dx
                else:
                    funspace = self.sim_inverse.functionspace.get_functionspace(subspace_id=1)
                    domain = self.sim_inverse.subdomains.dx
                if os.path.exists(path):
                    self.logger.info("Computing COM for '%s'" % (name))
                    conc = dio.read_function_hdf5('function', funspace, path)
                    com = self.compute_com(conc, domain)
                    for i, coord in enumerate(com):
                        self.measures['com_%i_' % i + name] = coord
                else:
                    self.logger.warning("Cannot compute COM. '%s' does not exist" % path)
            self._save_state()
        else:
            self.logger.warning("Cannot compute com. Instance 'sim_inverse' does not exist")

    def post_process(self, sim_list, threshold_list):
        self.compute_volume_thresholded()
        self.compute_com_all()
        # compute volumes / coms for each time step
        for measure in ['volume', 'com']:
            results_df = pd.DataFrame()
            for problem_type, threshold in product(
                    *[sim_list, threshold_list]):
                self.logger.info("Trying to compute '%s' for '%s' simulation with concentration threshold '%.02f'" % (
                    measure, problem_type, threshold))
                self.logger.info("-- This computation is performed in the reference configuration, no deformation! --")
                results_tmp = self.compute_from_conc_for_each_time_step(threshold=threshold, problem_type=problem_type,
                                                                        computation=measure)
                print(results_tmp)
                if results_tmp is not None:
                    new_col_names = [
                        "_".join([problem_type, measure, str(threshold), name.lower()]) if name != 'sim_time_step'
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

    def compute_from_conc_for_each_time_step(self, threshold=None, problem_type='forward', computation='volume'):
        if not threshold:
            threshold = self.conc_threshold_levels['T2']

        if problem_type == 'forward':
            sim_name = 'sim_forward'
            base_path = self.path_forward_sim
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
            self.logger.info("Recording steps: %s" % rec_steps)
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
                    com = self.compute_com(q_fun, sim.subdomains.dx)
                    for i, coord in enumerate(com):
                        results_dict["%s_%i" % ('all', i)] = coord
                else:
                    self.logger.warning("Cannot compute '%s' -- underfined" % computation)
                # for all subdomains
                for subdomain_id, subdomain_name in sim.subdomains.tissue_id_name_map.items():
                    if computation == 'volume':
                        results_dict[subdomain_name] = self.compute_volume(q_fun, sim.subdomains.dx(subdomain_id))
                    elif computation == 'com':
                        com = self.compute_com(q_fun, sim.subdomains.dx(subdomain_id))
                        for i, coord in enumerate(com):
                            results_dict["%s_%i" % (subdomain_name, i)] = coord
                    else:
                        self.logger.warning("Cannot compute '%s' -- underfined" % computation)

                results = results.append(results_dict, ignore_index=True)

            # new_col_names = [ "_".join([problem_type, 'volume', str(threshold), name.lower()]) if name!='sim_time_step'
            #                  else name.lower() for name in volumes.columns]

            results.columns = [name.lower() for name in results.columns]

            save_name = "_".join([computation, str(threshold)]) + '.pkl'
            save_path = os.path.join(base_path, save_name)
            results.to_pickle(save_path)
            self.logger.info("Saving '%s' dataframe to '%s'" % (computation, save_path))
            save_name = "_".join([computation, str(threshold)]) + '.xls'
            save_path = os.path.join(base_path, save_name)
            results.to_excel(save_path)
            self.logger.info("Saving '%s' dataframe to '%s'" % (computation, save_path))
            return results

        else:
            self.logger.warning("Cannot compute '%s' for '%s'. No such simulation instance names '%s'." % (
                computation, problem_type, sim_name))

    @staticmethod
    def thresh(f, thresh):
        smooth_f = 0.01
        f_thresh = 0.5 * (tanh((f - thresh) / smooth_f) + 1)
        return f_thresh

    @staticmethod
    def interpolate_non_matching(source_function, target_funspace):
        function_new = fenics.Function(target_funspace)
        fenics.LagrangeInterpolator.interpolate(function_new, source_function)
        return function_new

    @staticmethod
    def compute_com(fenics_scalar_field, domain=None):
        dim_geo = fenics_scalar_field.function_space().mesh().geometry().dim()
        com = []
        if domain is None:
            domain = fenics.dx
        volume = fenics.assemble(fenics_scalar_field * domain)
        for dim in range(dim_geo):
            coord_expr = fenics.Expression('x[%i]' % dim, degree=1)
            coord_int = fenics.assemble(coord_expr * fenics_scalar_field * domain)
            if volume > 0:
                coord = coord_int / volume
            else:
                coord = np.nan
            com.append(coord)
        return com

    @staticmethod
    def splitall(path):
        allparts = []
        while 1:
            parts = os.path.split(path)
            if parts[0] == path:  # sentinel for absolute paths
                allparts.insert(0, parts[0])
                break
            elif parts[1] == path:  # sentinel for relative paths
                allparts.insert(0, parts[1])
                break
            else:
                path = parts[0]
                allparts.insert(0, parts[1])
        return allparts

    @staticmethod
    def get_path_relative_to_subdir(abs_path, reference_subdir):
        split_path = super().splitall(abs_path)
        index = split_path.index(reference_subdir)
        path_relative = os.path.join(*split_path[index:])
        return path_relative

    @staticmethod
    def flatten_params(params_dict, prefix):
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

    @staticmethod
    def compute_volume(conc_fun, domain):
        vol = fenics.assemble(conc_fun * domain)
        return vol
