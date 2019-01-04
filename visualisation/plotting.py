"""

Provides project-specific plotting functions.

The plotting library consists of one main plotting function `plot`, which makes use of several subfunctions
to combine/overlay plots of different data types:

- plot_fenics_function: for FENICS 2D data, with subfunctions

    - plot_fenics_function_scalar: for FENICS 2D scalar field data
    - plot_fenics_function_vector: for FENICS 2D vector field data

- plot_sitk_image: for 2D image data

Using this general plot function, multiple default layouts are defined for plotting the following data fields in 2D:

- plot_concentration
- plot_growth
- plot_proliferation
- plot_displacement

"""

import os

import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from visualisation.helpers import mesh2triang, interpolate_over_grid, get_ranges_colormap, exclude_from_data, \
    add_colorbar

import utils.file_utils as fu
import visualisation.helpers as vh
from fenics_local import Function, Point, cells, parameters


#==== PLOTTING SUBFUNCTIONS

#--- FENICS FUNCTIONS

def plot_fenics_function_vector(ax, f, mode='quiver', plot_nth=None,
                                interpolate=False, n_interpolate=100,
                                range_f=None, cmap='gist_earth', norm=None, norm_ref=None, n_cmap_levels=None,
                                exclude_below=None, exclude_above=None, exclude_min_max=False, exclude_around=None,
                                color=None, **kwargs):
    """
    Subfunction for plotting FENICS 2D vectorfields,
    returns plot axis.
    """
    mesh = f.function_space().mesh()
    gdim = mesh.geometry().dim()
    w0 = f.compute_vertex_values(mesh)
    nv = mesh.num_vertices()
    if len(w0) != gdim * nv:
        raise AttributeError('Vector length must match geometric dimension.')
    #-- rearrange coordinates
    X = mesh.coordinates()
    X_coords = [X[:, i] for i in range(gdim)]
    U = [w0[i * nv : (i + 1) * nv ] for i in range(gdim)]

    # -- quiver and streamplot assume that x and y axes are increasing.
    #    -> swap sign of Ux or Uy if one of the axes is ordered in different direction
    ax_x_lim = ax.get_xlim()
    ax_y_lim = ax.get_ylim()
    if ax_x_lim[0] > ax_x_lim[1]:
        print("-- fenics vector plot: x axis is decreasing; change sign of vector x value for plotting")
        U[0] = - U[0]
    if ax_y_lim[0] > ax_y_lim[1]:
        print("-- fenics vector plot: y axis is decreasing; change sign of vector y value for plotting")
        U[1] = - U[1]


    if mode=='quiver':
        # -- catch kwargs
        if 'density' in kwargs:
            kwargs.pop('density')

        if interpolate:
            X, Y, UX, UY = interpolate_over_grid(X_coords[0], X_coords[1], U[0], U[1],
                                                 n=n_interpolate,
                                                 return_coords='meshgrid', method='cubic')
            #plot = ax.quiver(xi, yi, ui, vi, [C], **kwargs)
        else:
            X, Y, UX, UY = [X_coords[0], X_coords[1], U[0], U[1]]

        if color: # ignore cmap
            if 'cbar_label' in kwargs:
                kwargs.pop('cbar_label')
            plot = ax.quiver(X, Y, UX, UY, color=color, **kwargs)
        else:
            # -- Compute magnitude
            C2 = UX ** 2 + UY ** 2
            C = np.sqrt(C2)

            # -- colormap and range settings
            min_f, max_f, colormap, norm = get_ranges_colormap(C,
                                                               range=range_f, cmap=cmap, norm=norm, norm_ref=norm_ref,
                                                               n_cmap_levels=n_cmap_levels)

            # -- exclude data from plotting
            mask = exclude_from_data(C, min_f, max_f,
                                     exclude_below=exclude_below, exclude_above=exclude_above,
                                     exclude_min_max=exclude_min_max, exclude_around=exclude_around,
                                     data_type='standard')
            C[mask] = np.nan
            plot = ax.quiver(X, Y, UX, UY, C, cmap=colormap, norm=norm, **kwargs)


    elif mode=='streamlines':
        xi, yi, ui, vi = interpolate_over_grid(X_coords[0], X_coords[1], U[0], U[1],
                                               n=n_interpolate,
                                               return_coords='linspace', method='cubic')
        plot = ax.streamplot(xi, yi, ui, vi, **kwargs)
    return plot



def plot_fenics_function_scalar(ax, f, showmesh=True,
                                range_f=None, cmap='gist_earth', norm=None, norm_ref=None, n_cmap_levels=None,
                                exclude_below=None, exclude_above=None, exclude_min_max=False, exclude_around=None,
                                plot_nth=None, shading='flat', alpha=1,
                                **kwargs):
    """
    Subfunction for plotting FENICS 2D scalar field,
    returns plot axis.
    """
    mesh    = f.function_space().mesh()
    nv      = mesh.num_vertices()
    gdim    = mesh.geometry().dim()
    mesh_coordinates = mesh.coordinates().reshape((nv, gdim))
    triangulation = mesh2triang(mesh)
    values = np.asarray([f(point) if mesh.bounding_box_tree().collides_entity(Point(point)) # check if point in domain
                                      else np.nan                                        # assign nan otherwise
                                for point in mesh_coordinates])
    #-- colormap and range settings
    min_f, max_f, colormap, norm = get_ranges_colormap(values,
                                                       range=range_f, cmap=cmap, norm=norm, norm_ref=norm_ref,
                                                       n_cmap_levels=n_cmap_levels)

    #-- exclude data from plotting
    data = values[triangulation.triangles]
    mask = exclude_from_data(data, min_f, max_f,
                             exclude_below=exclude_below, exclude_above=exclude_above,
                             exclude_min_max=exclude_min_max, exclude_around=exclude_around,
                             data_type='triangulation')
    triangulation.set_mask(mask)

    #-- plot
    if showmesh:
        plot = ax.tripcolor(triangulation, values, cmap=colormap, norm=norm, shading=shading, edgecolors='k', linewidth=0.1,
                           vmin=min_f, vmax=max_f, alpha=alpha, **kwargs)
    else:
        plot = ax.tripcolor(triangulation, values, cmap=colormap, norm=norm, shading=shading,
                           vmin=min_f, vmax=max_f, alpha=alpha, **kwargs)
    return plot



def plot_fenics_function(ax, f, mode='quiver', **kwargs):
    """
    Subfunction for plotting FENICS 2D vector or scalarfield,
    returns plot axis.
    """
    mesh = f.function_space().mesh()
    gdim = mesh.geometry().dim()
    tdim = mesh.topology().dim()
    w0 = f.compute_vertex_values(mesh)
    #nv = mesh.num_vertices()
    value_dim = f.function_space().element().value_dimension(0)
    #if len(w0) != gdim * nv:
    #    raise AttributeError('Vector length must match geometric dimension.')

    if gdim == 2:
        # -- scalar
        if value_dim == 1:
            if 'color' in kwargs:
                kwargs.pop('color')
            plot = plot_fenics_function_scalar(ax, f, **kwargs)
            return plot
        elif value_dim == 2:
            if 'showmesh' in kwargs:
                kwargs.pop('showmesh')
            if 'shading' in kwargs:
                kwargs.pop('shading')
            plot = plot_fenics_function_vector(ax, f, mode=mode, **kwargs)
            return plot
        else:
            print("Plot function not defined for value dimension > 2. Here: %i" % tdim)

    else:
        print("Plot function not defined for geometric dimension %i" % gdim)


#--- SITK IMAGE
def plot_sitk_image(ax, image, segmentation=None, contour=False,
                    range=None, cmap='gist_earth', norm=None, norm_ref=0, n_cmap_levels=None,
                    label_alpha=1,
                    origin='lower',
                    **kwargs):
    """
    Subfunction for plotting image data (SimpleITK data format),
    returns plot axis.
    """
    #-- Convert Image Types
    img_type = sitk.sitkUInt8
    image_rec = sitk.Cast(sitk.RescaleIntensity(image), img_type)
    #-- Segmentation
    if segmentation:
        image_label_rec = sitk.Cast(segmentation, img_type)
        if contour:
            img_ol = sitk.LabelOverlay(image_rec, sitk.LabelContour(image_label_rec), opacity=label_alpha)
        else:
            img_ol = sitk.LabelOverlay(image_rec, image_label_rec, opacity=label_alpha)
    else:
        img_ol = image_rec
    #-- Prepare Figure with image
    nda = sitk.GetArrayFromImage(img_ol)
    min_f, max_f, colormap, norm = get_ranges_colormap(nda,
                                                       range=range, cmap=cmap, norm=norm, norm_ref=norm_ref,
                                                       n_cmap_levels=n_cmap_levels)

    img_origin  = img_ol.GetOrigin()
    img_spacing = img_ol.GetSpacing()
    img_size    = img_ol.GetSize()
    # extent -> (left, right, bottom, top)
    extent = (img_origin[0], img_origin[0] + img_size[0] * img_spacing[0],
              img_origin[1] + img_size[1] * img_spacing[1], img_origin[1])
    plot = ax.imshow(nda, interpolation=None, extent=extent, origin=origin,
                     cmap=colormap, norm=norm, vmin=min_f, vmax=max_f)
    return plot




#--- MAIN PLOT FUNCTION


def plot(plot_object_list, dpi=100, plot_range=None, margin=0.02, cbarwidth=0.05,
         save_path=None, show=True, xlabel='x position [mm]', ylabel='y position [mm]', **kwargs):
    """
    Each element in `plot_object_list` is a dictionary of the form::

        { 'object' : the object to be plotted,
          'param1' : one plot specific parameter,
          'param2' : another plot specific parameter,
          ...
        }

    See :py:meth:`show_img_seg_f()` for examples.
    Unless a 'zorder' argument is specified, elements are plotted in the order of occurrence in `plot_object_list`.

    """

    # -- create Figure
    fig, ax = plt.subplots(dpi=dpi)
    ax.set_aspect('equal')

    # -- if an image is provided, the axes will be oriented as in the image,
    #   otherwise use xlim, ylim for fixing display orientation
    # -- for discussion about imshow/extent https://matplotlib.org/tutorials/intermediate/imshow_extent.html
    if plot_range:
        ax.set_xlim(*plot_range[0:2])
        ax.set_ylim(*plot_range[2:4])


    if 'title' in kwargs:
        fig.suptitle(kwargs.pop('title'))
    #-- Check if only a single plot_object has been provided, if so, transform to dict
    if not type(plot_object_list)==list:
        plot_object_dict = {}
        plot_object_dict['object'] = plot_object_list
        plot_object_dict.update(kwargs)
        plot_object_list = [plot_object_dict]
    # -- Iterate through plot objects
    cbar_ax_list = [ax]
    for plot_object_dict_orig in plot_object_list:
        plot_object_dict = plot_object_dict_orig.copy()
        plot_object = plot_object_dict.pop('object')
        # check for 'color'
        if 'color' in plot_object_dict:
            color = plot_object_dict['color']
        else:
            color=False
        cbar = False
        if ('cbar_label' in plot_object_dict) and not color:
            cbar_label = plot_object_dict.pop('cbar_label')
            cbar=True

        #-- use global kwargs as reference and overwrite with settings that are specific to this plot_object
        params = kwargs.copy()
        params.update(plot_object_dict)
        #-- plot
        if type(plot_object)==Function:
            plot = plot_fenics_function(ax, plot_object, **params)
        elif type(plot_object)==sitk.Image:
            plot = plot_sitk_image(ax, plot_object, **params)
        else:
            print("The plot_object is of type '%s' -- not supported."%(type(plot_object)))
            raise Exception

        if cbar:
            cbax = add_colorbar(fig, cbar_ax_list[0], plot, cbar_label)
            cbar_ax_list.append(cbax)
            #fig.subplots_adjust(left=margin, bottom=margin, right=1 - margin - cbarwidth, top=1 - 2 * margin)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if save_path:
        fu.ensure_dir_exists(os.path.dirname(save_path))
        fig.savefig(save_path, bbox_inches='tight',  dpi=dpi)
        print("  - Saved figure to '%s'"%save_path)

    if show:
        vh.show_plot()


def show_img_seg_f(image=None, segmentation=None, function=None, contour=False, alpha=1.0, dpi=100,
                   showmesh=True, shading='flat', colormap=None, n_cmap_levels=20, alpha_f=0.5, range_f=None,
                   exclude_below=None, exclude_above=None, exclude_min_max=False, exclude_around=None,
                   title=None, label=None, show=True, path=None, cmap_ref=None, norm=None, color=None,
                   plot_range=None):
    """
    Convenience function providing default settings for certain types of plots
    """
    plot_list = []

    if image:
        plot_obj_img = {'object': image,
                        'segmentation': segmentation,
                        'label_alpha': alpha,
                        'contour': True,
                        'cmap': 'Greys',
                        'origin': 'upper'
                        }

        plot_list.append(plot_obj_img)

    if function:
        plot_obj_function = {'object': function,
                           'cbar_label': label,
                           'exclude_below': exclude_below,
                           'exclude_above': exclude_above,
                           'exclude_min_max': exclude_min_max,
                           'exclude_around': exclude_around,
                           'cmap': colormap,
                           'n_cmap_levels': n_cmap_levels,
                           'range_f': range_f,
                           'showmesh': showmesh,
                           'shading': shading,
                           'alpha': alpha_f,
                           'norm': norm,
                           'norm_ref': cmap_ref,
                           'color': color
                           }

        plot_list.append(plot_obj_function)

    plot(plot_list, title=title, save_path=path, show=show, dpi=dpi, plot_range=plot_range)





def plot_concentration(image, label, fun, title, path=None, show=False, plot_range=None):
    """
    Convenience function for plotting concentration fields.
    """
    show_img_seg_f(image, label, fun, contour=True, showmesh=False, alpha_f=1,
                       range_f=[0.001, 1.01], exclude_min_max=True, colormap='viridis',
                       n_cmap_levels=20, title=title, label="concentration",
                       path=path, show=show, plot_range=plot_range)


def plot_growth(image, label, fun, title, path=None, show=False):
    """
    Convenience function for plotting growth fields.
    """
    show_img_seg_f(image, label, fun, contour=True, showmesh=False, alpha_f=1,
                       range_f=[0.0, 0.2], exclude_as_range=True, colormap='viridis',
                       n_cmap_levels=20,title=title, label="growth",
                       path=path, show=show)


def plot_proliferation(image, label, fun, title, path=None, show=False):
    """
    Convenience function for plotting proliferation fields.
    """
    show_img_seg_f(image, label, fun, contour=True, showmesh=False, alpha_f=1,
                       exclude_around=(0, 0.0001), range_f=[-0.02, 0.1],
                       title=title, label="proliferation term",
                       path=path, show=show, colormap='RdBu_r', n_cmap_levels=20, cmap_ref=0.0)


def plot_displacement(image, label, fun, title, path=None, show=False):
    """
    Convenience function for plotting displacement fields.
    """
    show_img_seg_f(image, label, fun, contour=True, showmesh=False, alpha_f=1,
                       exclude_around=None, range_f=[0.0, 20], exclude_min_max=True, exclude_below=0.5,
                       title=title, label="displacement", color=None,
                       path=path, show=show, colormap='viridis', n_cmap_levels=20)


def sitk_show(img, title=None, margin=0.05, dpi=40):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)

    vh.show_plot()


def show_img_seg(image, segmentation, contour=False, alpha=1.0, dpi=100):
        # == Convert image types to uint8
        img_type = sitk.sitkUInt8
        image_label_rec = sitk.Cast(segmentation, img_type)
        image_rec = sitk.Cast(image, img_type)
        if contour:
            sitk_show(sitk.LabelOverlay(image_rec, sitk.LabelContour(image_label_rec), alpha), dpi=dpi)
        else:
            sitk_show(sitk.LabelOverlay(image_rec, image_label_rec, alpha), dpi=dpi)



def plotmat(function, title=''):
    p = plot(function)
    p.set_cmap("viridis")
    p.set_clim(function.vector().min(), function.vector().max())
    plt.colorbar(p)
    plt.title(title)
    vh.show_plot()



def plot_plt(field_var, showmesh=True,  shading='flat', contours=[],
             colormap=plt.cm.jet, norm=None, cmap_ref=None, range_f=None,alpha_f=1,
             title=None, path=None, show=True, dpi=300):
    mesh = field_var.function_space().mesh()
    n = mesh.num_vertices()
    d = mesh.geometry().dim()
    value_dim = field_var.function_space().num_sub_spaces()
    #-- only 2D
    if d == 2:
        # Create triangulation
        mesh_coordinates = mesh.coordinates().reshape((n, d))
        triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
        triangulation = tri.Triangulation(mesh_coordinates[:, 0],
                                                  mesh_coordinates[:, 1],
                                                  triangles)
        plt.figure()
        parameters['allow_extrapolation'] = True
        z = np.asarray([field_var(point) if mesh.bounding_box_tree().collides_entity(Point(point)) # check if point in domain
                                          else np.nan                                        # assign nan otherwise
                                        for point in mesh_coordinates])
        #z = np.asarray([field_var(point) for point in mesh_coordinates])
        if range_f is None:
            min_f = min(z)
            max_f = max(z)
        else:
            min_f = range_f[0]
            max_f = range_f[1]
        if type(colormap)==str:
            cmap = plt.cm.get_cmap(colormap)
        elif colormap==None:
            cmap = plt.cm.get_cmap('gist_earth')
        else:
            cmap = colormap

        if (norm is None) and (not cmap_ref is None):
            norm = vh.MidpointNormalize(midpoint=cmap_ref, vmin=min_f, vmax=max_f)

        #-- scalar function
        if value_dim == 1:

            if showmesh:
                 plt.tripcolor(triangulation, z, cmap=cmap, norm=norm, shading=shading, edgecolors='k',linewidth=0.1,
                                vmin = min_f, vmax = max_f, alpha = alpha_f)
            else:
                 plt.tripcolor(triangulation, z, cmap=cmap, norm=norm, shading=shading,
                                vmin = min_f, vmax = max_f, alpha = alpha_f)

        elif value_dim == 2:
            plot()

        plt.colorbar()
        if type(contours)==int:
            plt.tricontour(triangulation, z, contours, colors = 'k')
        elif len(contours)>0:
            plt.tricontour(triangulation, z, levels=contours, colors='k')
    elif d == 3:
        plot(field_var)

    if title is not None:
        plt.suptitle(title, y=1.08)
    if path is not None:
        fu.ensure_dir_exists(os.path.dirname(path))
        plt.savefig(path, dpi=dpi,bbox_inches='tight')
        print("-- saved figure to '%s'"%path)
    if show:
        vh.show_plot()


