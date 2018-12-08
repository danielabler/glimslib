# Dockerfile for GlimS based on dolfinadjoint/pyadjoint
FROM quay.io/dolfinadjoint/pyadjoint

# Set the working directory
WORKDIR /shared

# Copy the current directory contents into the container at /app
COPY . /shared

# Install any needed python packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Set Environment variables
#-- path of matplotlibrc
ENV MATPLOTLIBRC=/opt/project/matplotlibrc

# For TK support add
#RUN sudo apt-get update
#RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python-tk
