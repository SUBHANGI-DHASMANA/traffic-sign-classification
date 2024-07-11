# Use the latest TensorFlow notebook from Jupyter as the base image
FROM jupyter/tensorflow-notebook:latest

# Switch to root user to install packages and fix permissions
USER root

# Upgrade pip, install necessary packages, and fix permissions
RUN pip install --upgrade pip && \
    pip install transformers opencv-python && \
    fix-permissions "/home/${NB_USER}"

# Switch back to the default non-root user
USER $NB_UID
