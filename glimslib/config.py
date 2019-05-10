"""Provides path settings for GlimS project"""

import os

base_path = os.path.dirname(os.path.dirname(__file__))


output_dir                              = os.path.join(base_path,  'output')
output_dir_testing                      = os.path.join(output_dir, 'test_cases')
output_dir_simulation                   = os.path.join(output_dir, 'simulation')
output_dir_application                   = os.path.join(output_dir, 'application')

output_dir_temp                         = os.path.join(output_dir, 'temp')

test_dir = os.path.join(base_path, 'test_cases')
test_data_dir = os.path.join(test_dir, 'data')

# meshtool settings
path_to_meshtool = '/home/fenics/software/MESHTOOL_source'
path_to_meshtool_bin = os.path.join(path_to_meshtool, 'bin', 'MeshTool')
path_to_meshtool_xsd = os.path.join(path_to_meshtool, 'src', 'xml-io', 'imaging_meshing_schema.xsd')

# Switch for using adjoint; false by default.
USE_ADJOINT = False