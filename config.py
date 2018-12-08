"""Provides path settings for GlimS project"""

import os

base_path = os.path.dirname(__file__)


output_dir                              = os.path.join(base_path,  'output')
output_dir_testing                      = os.path.join(output_dir, 'testing')



# Switch for using adjoint; false by default.
USE_ADJOINT = False