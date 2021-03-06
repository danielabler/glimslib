# Docker Image for GlimS

This image provides FEniCS 2017.2.0 (python 2.7 / python 3.5) and the now deprecated [dolfinadjoint/libadjoint](https://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html).
It extends [quay.io/dolfinadjoint/dolfin-adjoint] with additional python packages used by GlimS.
See *requirements.txt* for details.

It also adds a full installation of the [Advanced Normalization Tools (ANTs)](http://stnava.github.io/ANTs/) for image registration.

## Instructions

### Use Built Image

A built version of this image is available on [docker hub](https://hub.docker.com/) at:

You can download this image by:

```
docker pull danabl/glimslib_ants:2017-2
```  

### Building Image
Enter the directory where this readme file is located.
Build the docker image with name *glims_image* by

```
    docker build -t glims_image .
```

### Creating Container

Create container *glims* from *glims_image*, with working directory
(`-w`) */home/fenics/workdir* in the container and mapping (`-v`) local
*/path/on/host* to */home/fenics/workdir*.
```
docker run --name glims -w /home/fenics/workdir -v /path-on-host/:/home/fenics/workdir -d -p 127.0.0.1:8880:8888 glims_image:latest 'jupyter-notebook --ip=0.0.0.0'
```
Option `-d` detaches the process and `-p` maps ports for an ipython notebook.

To connect into the running container, use
```
docker exec -ti -u fenics glims /bin/bash -l
```

Restart a previously stopped container by
```
docker start container_name
```

### Adding Support for interactive plotting from Container

The following procedure does not provide all fenics plotting features
but allows basic interactive plotting via matplotlib.

* Install `python-tk` in an existing container, or build the image with
  with this library:

  ```
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python-tk
  ```

* Before connecting to the container, run socat:
  ```
  socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
  ```

* Connect by
  ```
  docker exec -ti -e DISPLAY=<your-ip>:0 -u fenics glims /bin/bash -l
  ```
  where *<your-ip>* is the current IP of your machine, and *glims* the
  name of the container.

* To plot from python, include the following at the top of your script:
  ```
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    parameters["plotting_backend"] = "matplotlib"
  ```

* You can plot now :-)
  ```
    mesh3d = UnitCubeMesh(16,16,16)
    plot(mesh3d)
    plt.show()
  ```

