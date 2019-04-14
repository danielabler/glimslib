# Dockerfile for GlimS based on dolfinadjoint/pyadjoint
FROM quay.io/dolfinadjoint/pyadjoint:2018.1.0

# Set the working directory
WORKDIR /shared

# Copy the current directory contents into the container at /app
COPY . /shared

# Install open-gl dependencies for vtk -- suboptimal solution ...
RUN sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 6B05F25D762E3157 &&\
    sudo apt-get update && \
    sudo apt-get -y install libgl1-mesa-glx

# Install any needed python packages specified in requirements.txt
#RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip3 install --upgrade pip &&\
    pip3 install --trusted-host pypi.python.org -r requirements.txt
# Set Environment variables
ENV MATPLOTLIBRC=/opt/project/matplotlibrc


# For TK support add
#RUN sudo apt-get update
#RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python-tk

# Create directory for custom software installation
ARG INST_DIR=/home/fenics/software
RUN mkdir ${INST_DIR} && echo "Created ${INST_DIR}"
# Replace outdated cmake by current version, following this guide:
# https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line/865294#865294
#- remove existing cmake and install libncurses-dev
RUN sudo apt-get update && \
    sudo apt -y remove --purge --auto-remove cmake && \
    sudo apt-get -y install libncurses-dev
#- install new cmake from sources
ARG version=3.13
ARG build=0
ARG CMAKE_NAME=cmake-$version.$build
RUN echo "Installing cmake..." && cd ${INST_DIR} && \
    wget https://cmake.org/files/v$version/${CMAKE_NAME}.tar.gz && \
    tar -xzvf ${CMAKE_NAME}.tar.gz && \
    cd ${INST_DIR}/${CMAKE_NAME} && \
    ./bootstrap && \
    make -j4 && \
    sudo make install
ENV PATH=$PATH:/usr/local/bin

# Install ANTs from sources
# ANTs seems to compile only with gcc 6, but this image ships with gcc 7
RUN sudo apt -y install g++-6 gcc-6
ENV CC=/usr/bin/gcc-6
ENV CXX=/usr/bin/g++-6

# https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS
ARG ANTS_FILES=${INST_DIR}/ants_git
ARG ANT_INST_PATH=${INST_DIR}/ants_build
RUN echo "Installing ANTs..." && cd ${INST_DIR} && \
    git clone https://github.com/stnava/ANTs.git ${ANTS_FILES} && \
    cd ${ANTS_FILES} && \
    git checkout tags/v2.3.0 && \
    mkdir ${ANT_INST_PATH} && \
    cd ${ANT_INST_PATH} && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr -DSuperBuild_ANTS_USE_GIT_PROTOCOL=OFF -DRUN_LONG_TESTS=OFF ${ANTS_FILES}&&\
    make -j8 && \
    cp ${ANTS_FILES}/Scripts/* ${ANT_INST_PATH}/bin

ENV ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
ENV ANTSPATH=${ANT_INST_PATH}/bin
ENV PATH=${ANTSPATH}:$PATH
ENV PYTHONPATH=/opt/project:$PYTHONPATH
