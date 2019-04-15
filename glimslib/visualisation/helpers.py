"""Provides helper functions for visualisation module.

"""
import os
import time

import matplotlib.pylab as plt
import numpy as np
from matplotlib import pyplot as plt, colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from ufl import FiniteElement

from glimslib.fenics_local import FunctionSpace, Function

from glimslib.visualisation import config


def show_plot(save_as=None):
    """
    Wrapper around standard 'plt.show()' command to ensure that
    - command is only used when interactive backend is selected
    - otherwise, an image file is stored in a temp folder
    """
    print(config.backend_interactive)
    if config.backend_interactive:
        plt.show()
    else:
        fig = plt.gcf()
        tmp_name    = time.strftime("%Y-%m-%d_%H-%M-%S") + '.png'
        path_to_fig = os.path.join(config.path_tmp_fig, tmp_name)
        fig.savefig(path_to_fig)

    if save_as:
        fig = plt.gcf()
        fig.savefig(save_as)

    plt.close()

def convert_meshfunction_to_function(mesh, mesh_function):
    # from https://www.allanswered.com/post/avwnw/how-to-visualize-mesh-function-of-subdomains/
    subdIDs = np.asarray(mesh_function.array(),dtype=np.int32)

    IDvalues = np.zeros(max(subdIDs)+1)
    IDvalues[np.unique(subdIDs)]=np.unique(subdIDs)

    Element = FiniteElement("DP", mesh.ufl_cell(), 0)
    Space   = FunctionSpace(mesh, Element)
    markerFun = Function(Space)
    markerFun.vector()[:] = np.choose(subdIDs, IDvalues)
    return markerFun


def mesh2triang(mesh):
    import matplotlib.tri as tri
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())


def interpolate_over_grid(x, y, u, v, n=100, return_coords='linspace', method='cubic'):
    """
    Interpolates 2D vector field data with (values u, v, at positions x, y) over grid with n nodes.
    - return_coords='linspace' -> x_return, y_return each contain n values
    - return_coords='meshgrid' -> x_return, y_return each contain nxn values
    """
    # from:
    # https://stackoverflow.com/questions/33637693/how-to-use-streamplot-function-when-1d-data-of-x-coordinate-y-coordinate-x-vel
    nx = ny = n
    #-- (N, 2) arrays of input x,y coords and u,v values
    pts  = np.vstack((x, y)).T
    vals = np.vstack((u, v)).T
    #-- the new x and y coordinates for the grid, which will correspond to the
    #   columns and rows of u and v respectively
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    #-- an (nx * ny, 2) array of x,y coordinates to interpolate at
    ipts = np.vstack(a.ravel() for a in np.meshgrid(xi,yi)).T
    #-- an (nx * ny, 2) array of interpolated u, v values
    ivals = griddata(pts, vals, ipts, method=method)
    #-- reshape interpolated u,v values into (nx, ny) arrays
    ui, vi = ivals.T
    ui.shape = vi.shape = (ny, nx)
    if return_coords=='meshgrid':
        x_return = ipts[:,0]
        y_return = ipts[:,1]
    elif return_coords=='linspace':
        x_return = xi
        y_return = yi
    return x_return, y_return, ui, vi


def get_ranges_colormap(values, range=None, cmap='gist_earth', norm=None, norm_ref=None, n_cmap_levels=None,
                        **kwargs):
    """
    Parses commandline arguments to construct boundaries, colormap and norm.
    """
    values_flat = np.ndarray.flatten(values)
    if range is None:
        min_ = min(values_flat)
        max_ = max(values_flat)
    else:
        min_ = range[0]
        max_ = range[1]
        if min_ is None:
            min_ = min(values_flat)
        if max_ is None:
            max_ = max(values_flat)

    if type(cmap) == str:
        if n_cmap_levels:
            cmap_  = plt.cm.get_cmap(cmap, n_cmap_levels)
        else:
            cmap_ = plt.cm.get_cmap(cmap)
    else:
        cmap_ = cmap

    if norm_ref is None:
        norm_ref = (min_+max_)/2.

    if norm is None:
        norm = MidpointNormalize(midpoint=norm_ref, vmin=min_, vmax=max_)
    return min_, max_, cmap_, norm


def exclude_from_data(data, min_f, max_f,
                      exclude_below=None, exclude_above=None, exclude_min_max=False, exclude_around=None,
                      min_max_eps=0.00001, data_type='standard'):
    """
    Produces a boolean mask basked with True value when any or several selection criteria is met
    - data_type='standard'      ->  True/False value for each entry in 'data' array.
    - data_type='triangulation' ->  True/False value for each triangle (data triple) in 'data' array.
    If no criterium applies, the returned mask contains False values only
    """
    if not exclude_above:
        exclude_above = max_f # either from data value range, or from 'range' attibute -> get_ranges_colormap()
    if not exclude_below:
        exclude_below = min_f

    mask_list = []
    if exclude_min_max:
        if data_type=='standard':
            mask1 = np.ma.make_mask(np.where(data > exclude_above + min_max_eps, 1, 0))
            mask2 = np.ma.make_mask(np.where(data < exclude_below - min_max_eps, 1, 0))
        elif data_type=='triangulation':
            mask1 = np.logical_or.reduce((np.where(data > exclude_above+min_max_eps, 1, 0).T))
            mask2 = np.logical_or.reduce((np.where(data < exclude_below-min_max_eps, 1, 0).T))
        mask_min_max = np.logical_or(mask1, mask2)
        mask_list.append(mask_min_max)

    if type(exclude_around)==list:
        ref = exclude_around[0]
        eps = exclude_around[1]
        if data_type=='standard':
            mask1 = np.ma.make_mask(np.where(data > ref - eps, 1, 0))
            mask2 = np.ma.make_mask(np.where(data < ref - eps, 1, 0))
        elif data_type=='triangulation':
            mask_below = np.logical_or.reduce((np.where(data > ref - eps, 1, 0).T))
            mask_above = np.logical_or.reduce((np.where(data < ref - eps, 1, 0).T))
        mask_above_below = np.logical_or(mask_below, mask_above)
        mask_list.append(mask_above_below)

    if len(mask_list)>1:
        mask = mask_list[0]
        for i in range(1, len(mask_list)-1):
            mask = np.logical_or(mask,mask_list[i] )
    elif len(mask_list)==1:
        mask = mask_list[0]
    else:
        if data_type == 'standard':
            mask = np.ma.make_mask(np.zeros(data.shape))
        else:
            mask = np.logical_or.reduce(np.zeros(data.shape).T)
    return mask


def add_colorbar(fig, ax, img_handle, label=None, size='5%', pad=0.05, fontsize=None):
    divider = make_axes_locatable(ax)
    cbax = divider.append_axes("right", size=size, pad=pad)
    cbar = fig.colorbar(img_handle, cax=cbax)
    cbar.ax.get_yaxis().labelpad = 17
    if fontsize is not None:
        cbar.ax.tick_params(labelsize=fontsize)
    if not label == None:
        cbar.ax.set_ylabel(label, rotation=270)#, fontsize=fontsize)
    return cbax


class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	# from http://chris35wills.github.io/matplotlib_diverging_colorbar/
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))