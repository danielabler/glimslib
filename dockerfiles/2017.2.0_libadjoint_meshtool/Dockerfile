# Dockerfile for GlimS based on dolfinadjoint/libadjoint
FROM danabl/glimslib_ants:2017-2

# Set the working directory
WORKDIR /shared

# Copy the current directory contents into the container at /app
COPY . /shared

# Install open-gl dependencies for vtk -- suboptimal solution ...
RUN sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 6B05F25D762E3157 &&\
    sudo apt-get update && \
    sudo apt-get -y install freeglut3-dev freeglut3

#-- Setup MESHING-TOOL
ARG INST_DIR=/home/fenics/software

# Install VTK
ARG VTK_SOURCE_DIR=${INST_DIR}/VTK_source
ARG VTK_BUILD_DIR=${INST_DIR}/VTK_build
ARG VTK_INSTALL_DIR=${INST_DIR}/VTK_install
RUN echo "Installing VTK..." && \
    git clone https://github.com/Kitware/VTK ${VTK_SOURCE_DIR} && \
    cd ${VTK_SOURCE_DIR} && \
    git checkout v8.1.0 && \
    mkdir ${VTK_BUILD_DIR} && \
    cd ${VTK_BUILD_DIR} && \
    cmake -DCMAKE_INSTALL_PREFIX=${VTK_INSTALL_DIR} -DBUILD_EXAMPLES:BOOL=OFF -DBUILD_TESTING:BOOL=OFF ${VTK_SOURCE_DIR} &&\
    make -j4 && \
    make install

# Install XERCESC
ARG XERCES_SOURCE_DIR=${INST_DIR}/XERCES_source
ARG XERCES_BUILD_DIR=${INST_DIR}/XERCES_build
ARG XERCES_INSTALL_DIR=${INST_DIR}/XERCES_install
RUN echo "Installing XERCESC..." && \
    git clone https://github.com/apache/xerces-c ${XERCES_SOURCE_DIR} && \
    cd ${XERCES_SOURCE_DIR}&& \
    git checkout Xerces-C_3_2_0 && \
    mkdir ${XERCES_BUILD_DIR} && \
    cd ${XERCES_BUILD_DIR} && \
    cmake -DCMAKE_INSTALL_PREFIX=${XERCES_INSTALL_DIR} ${XERCES_SOURCE_DIR} &&\
    make -j4 && \
    make install

# Install CGAL
ARG CGAL_SOURCE_DIR=${INST_DIR}/CGAL_source
ARG CGAL_BUILD_DIR=${INST_DIR}/CGAL_build
ARG CGAL_INSTALL_DIR=${INST_DIR}/CGAL_install
RUN echo "Installing CGAL..." && \
    git clone https://github.com/CGAL/cgal.git ${CGAL_SOURCE_DIR} && \
    cd ${CGAL_SOURCE_DIR}&& \
    git checkout releases/CGAL-4.8.1 && \
    mkdir ${CGAL_BUILD_DIR} && \
    cd ${CGAL_BUILD_DIR} && \
    cmake -DCMAKE_INSTALL_PREFIX=${CGAL_INSTALL_DIR} -DBUILD_SHARED_LIBS=OFF ${CGAL_SOURCE_DIR} &&\
    make -j4 && \
    make install

# Install XSD
ARG XSD_TAR_FILE=${INST_DIR}/XSD.tar.bz2
ARG XSD_INSTALL_DIR=${INST_DIR}/xsd-4.0.0-x86_64-linux-gnu
RUN echo "Installing XSD..." && cd ${INST_DIR} &&\
    wget -O ${XSD_TAR_FILE} https://www.codesynthesis.com/download/xsd/4.0/linux-gnu/x86_64/xsd-4.0.0-x86_64-linux-gnu.tar.bz2 && \
    tar -xjf ${XSD_TAR_FILE}

# Install MeshTool
ARG MESHTOOL_SOURCE_DIR=${INST_DIR}/MESHTOOL_source
ARG MESHTOOL_BUILD_DIR=${INST_DIR}/MESHTOOL_build
ARG MESHTOOL_INSTALL_DIR=${INST_DIR}/MESHTOOL_install
RUN echo "Installing MeshTool..." && \
    git clone https://github.com/danielabler/meshtool.git ${MESHTOOL_SOURCE_DIR} && \
    cd ${MESHTOOL_SOURCE_DIR} && \
    git checkout master && \
    mkdir ${MESHTOOL_BUILD_DIR} && \
    cd ${MESHTOOL_BUILD_DIR} && \
    cmake -DCMAKE_INSTALL_PREFIX=${MESHTOOL_INSTALL_DIR} -DCGAL_DIR=${CGAL_BUILD_DIR} -DVTK_DIR=${VTK_BUILD_DIR} -DXERCESC_ROOT_DIR=${XERCES_INSTALL_DIR} -DXSD_DIR=${XSD_INSTALL_DIR}/libxsd ${MESHTOOL_SOURCE_DIR} && \
    make -j4 && \
    make install

# cleanup
RUN echo "Removing source and build directories..." &&\
    rm -rf ${CGAL_SOURCE_DIR} && rm -rf ${CGAL_BUILD_DIR} && \
    rm -rf ${VTK_SOURCE_DIR} && rm -rf ${VTK_BUILD_DIR} && \
    rm -rf ${XERCES_SOURCE_DIR} && rm -rf ${XERCES_BUILD_DIR}