import os
from config import *

from fenics_local import __version__ as version


if 'SIMTIME' in os.environ:
    sim_time = int(os.environ['SIMTIME'])
else:
    sim_time = 50

if 'SIMOUTPATH' in os.environ:
    output_path = os.environ['SIMOUTPATH']
else:
    output_path = os.path.join(output_dir_testing, 'image_based_optimisation', version, 'steps-%03d'%sim_time)

if 'DEGREE' in os.environ:
    degree_txt = os.environ['DEGREE']
    output_path = os.path.join(output_path, degree_txt)
    if degree_txt=='quadratic':
        degree = 2
    elif degree_txt=='linear':
        degree = 1
    else:
        degree = 1
else:
    degree = 1

#output_path='/opt/project/output/test_cases/image_based_optimisation/2017.2.0/2019-03-07/steps-050/quadratic'


path_00_data_extraction = os.path.join(output_path, '00_data_extraction')
path_01_forward_simulation = os.path.join(output_path, '01_forward_simulation')
path_01_forward_simulation_red = os.path.join(output_path, '01_forward_simulation_red')

path_02_deformed_image = os.path.join(output_path, '02_deformed_image')
path_03_registration = os.path.join(output_path, '03_registration')
path_04_optimization_from_image = os.path.join(output_path, '04_optimization_from_image')
path_04_optimization_from_sim = os.path.join(output_path, '04_optimization_from_simulation')
path_04_optimization_from_sim_reduced = os.path.join(output_path, '04_optimization_from_simulation_reduced')
path_04_optimization_from_sim_reduced_no_save = os.path.join(output_path, '04_optimization_from_simulation_reduced_no_save')
path_04_optimization_from_sim_reduced_with_save = os.path.join(output_path, '04_optimization_from_simulation_reduced_with_save')

path_05_forward_simulation_optimized_from_image = os.path.join(output_path, '05_forward_simulation_optimized_from_image')
path_05_forward_simulation_optimized_from_sim = os.path.join(output_path, '05_forward_simulation_optimized_from_simulation')
path_05_forward_simulation_optimized_from_sim_reduced = os.path.join(output_path, '05_forward_simulation_optimized_from_simulation_reduced')

path_06_comparison = os.path.join(output_path, '06_comparison')
path_06_comparison_from_Image = os.path.join(output_path, '06_comparison_from_image')


#output_path = os.path.join(output_dir_testing, 'image_based_optimisation', "2017.2.0", "single-core", "steps-100")

path_to_2d_image = os.path.join(output_path, 'image_2d.mha')
path_to_2d_labels = os.path.join(output_path, 'labels_2d.mha')
path_to_2d_labelfunction = os.path.join(output_path, 'label_function_2d.h5')
path_to_2d_imagefunction = os.path.join(output_path, 'image_function_2d.h5')



params_sim = {
                "sim_time" : sim_time,
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
                "D_WM" :     0.01,
                "D_GM" :     0.01,
                "rho_WM" :   0.01,
                "rho_GM" :   0.01,
                "coupling" : 0.01
                }

seed_position = [148, -67]
image_slice   = 87

thresholds = [0.2, 0.8]


