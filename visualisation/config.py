"""Provides configuration settings for visualisation module.

It handles:
- Output option, either 'plt.plot()' for non-interactive backends or storage to local temp output dir.
"""

import os
import matplotlib as mpl

import config
import utils.file_utils as fu


#=== IDENTIFY TYPE OF PLOTTING BACKEND
#-- IMPORTANT if using PyCharm:
#   This only works if SciView is disabled. Settings, Tools, Python Scientific, disable -> Show Plots in Toolwindow
if mpl.get_backend() in mpl.rcsetup.non_interactive_bk:
    backend_interactive = False
else:
    backend_interactive = True

#=== PATH FOR TEMP PLOTS
path_tmp_fig = os.path.join(config.output_dir, 'tmp_fig')
fu.ensure_dir_exists(path_tmp_fig)


