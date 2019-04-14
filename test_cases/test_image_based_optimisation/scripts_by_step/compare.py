import logging
import os

import config
config.USE_ADJOINT = True
import test_cases.test_image_based_optimisation.testing_config as test_config

from simulation.simulation_tumor_growth_brain_quad import TumorGrowthBrain
from simulation.helpers.helper_classes import Boundary
import fenics_local as fenics
import utils.file_utils as fu
import utils.data_io as dio
import visualisation.plotting as plott

path_to_hdf5_mesh = os.path.join(test_config.test_data_dir,'brain_atlas_mesh_2d_reduced_domain.h5')
mesh, subdomains, boundaries = dio.read_mesh_hdf5(path_to_hdf5_mesh)

funspace_disp = fenics.VectorFunctionSpace(mesh, 'Lagrange', 1)
funspace_conc = fenics.FunctionSpace(mesh, 'Lagrange', test_config.degree)

output_path = os.path.join(test_config.output_path, 'comparison')

# direct from sim
output_path_1_red = test_config.path_01_forward_simulation_red

path_to_sim_conc_01 = os.path.join(output_path_1_red, 'concentration_simulated_direct.h5')
conc_sim_01 = dio.read_function_hdf5('function', funspace_conc,path_to_sim_conc_01)

plott.show_img_seg_f(function=conc_sim_01, show=True,
                     path=os.path.join(output_path, 'conc_01.png'),
                     dpi=300)

path_to_sim_disp_01 = os.path.join(output_path_1_red, 'displacement_simulated_direct.h5')
disp_sim_01 = dio.read_function_hdf5('function', funspace_disp,path_to_sim_disp_01)

plott.show_img_seg_f(function=disp_sim_01, show=True,
                     path=os.path.join(output_path, 'sim_01.png'),
                     dpi=300)

# reloaded in 01b
path_to_sim_conc_01b = os.path.join(output_path_1_red, 'concentration_simulated.h5')
conc_sim_01b = dio.read_function_hdf5('function', funspace_conc,path_to_sim_conc_01b)

plott.show_img_seg_f(function=conc_sim_01b, show=True,
                     path=os.path.join(output_path, 'conc_01b.png'),
                     dpi=300)


path_to_sim_disp_01b = os.path.join(output_path_1_red, 'displacement_simulated.h5')
disp_sim_01b = dio.read_function_hdf5('function', funspace_disp,path_to_sim_disp_01b)


plott.show_img_seg_f(function=disp_sim_01b, show=True,
                     path=os.path.join(output_path, 'disp_01b.png'),
                     dpi=300)

# from 3
output_path_3 = test_config.path_03_registration

path_to_sim_conc_03 = os.path.join(output_path_3, 'concentration_from_simulation_reduced.h5')
conc_sim_03 = dio.read_function_hdf5('function', funspace_conc, path_to_sim_conc_03)


plott.show_img_seg_f(function=conc_sim_03, show=True,
                     path=os.path.join(output_path, 'conc_03.png'),
                     dpi=300)


path_to_sim_disp_03 = os.path.join(output_path_3, 'displacement_from_simulation_reduced.h5')
disp_sim_03 = dio.read_function_hdf5('function', funspace_disp, path_to_sim_disp_03)

plott.show_img_seg_f(function=disp_sim_03, show=True,
                     path=os.path.join(output_path, 'disp_03.png'),
                     dpi=300)

path_to_est_disp_03 = os.path.join(output_path_3, 'displacement_from_registration_reduced.h5')
disp_est_03 = dio.read_function_hdf5('function', funspace_disp, path_to_est_disp_03)


plott.show_img_seg_f(function=disp_est_03, show=True,
                     path=os.path.join(output_path, 'disp_est_03.png'),
                     dpi=300)

print(fenics.errornorm(conc_sim_01, conc_sim_01b))
print(fenics.errornorm(conc_sim_01, conc_sim_03))
print(fenics.errornorm(conc_sim_01b, conc_sim_03))

print(fenics.errornorm(disp_sim_01, disp_sim_01b))
print(fenics.errornorm(disp_sim_01, disp_sim_03))
print(fenics.errornorm(disp_sim_01b, disp_sim_03))
print(fenics.errornorm(disp_sim_03, disp_est_03))

conc_diff_01_03 = fenics.project(conc_sim_01-conc_sim_03, funspace_conc)
plott.show_img_seg_f(function=conc_diff_01_03, show=True,
                     path=os.path.join(output_path, 'conc_diff_01_03.png'),
                     dpi=300)

disp_diff_01_03 = fenics.project(disp_sim_01-disp_sim_03, funspace_disp)
plott.show_img_seg_f(function=disp_diff_01_03, show=True,
                     path=os.path.join(output_path, 'disp_diff_01_03.png'),
                     dpi=300)



for threshold_value in test_config.thresholds:
    path_to_threshold_sim_h5 = os.path.join(output_path_3,
                                            'thresholded_concentration_simulation_%03d.h5' % (threshold_value * 100))
    path_to_threshold_reduced_h5 = os.path.join(output_path_3, 'thresholded_concentration_simulation_reduced_%03d.h5'%(threshold_value*100))
    conc_sim_thresh = dio.read_function_hdf5('function', funspace_conc, path_to_threshold_sim_h5)
    conc_sim_thresh_red = dio.read_function_hdf5('function', funspace_conc, path_to_threshold_reduced_h5)

    conc_diff = fenics.project(conc_sim_thresh-conc_sim_thresh_red, funspace_conc)
    path_thresh_diff = os.path.join(output_path, 'conc_thresh_diff_%03d.png'%(threshold_value*100))
    plott.show_img_seg_f(function=conc_diff, show=True,
                         path=path_thresh_diff, dpi=300)