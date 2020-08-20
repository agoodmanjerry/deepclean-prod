#! /bin/bash -e

#======================================================
# DOWNLOAD AND INSTALL MINICONDA
# wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# ./miniconda.sh

#======================================================
# ADD CUDA TO PATHS AND ADD TO .bashrc
# echo 'PATH="/usr/local/cuda-10.2/bin:${PATH}"' >> ~/.bashrc
# echo 'LD_LIBRARY_PATH="/usr/local/cuda-10.2:${LD_LIBRARY_PATH}"' >> ~/.bashrc
# TODO: restart terminal somehow?
# source ~/.bashrc

#======================================================
# BUILD BASE DEEPCLEAN ENVIRONMENT (DO AS MUCH CONDA INSTALLING AS POSSIBLE)
conda env create -f environment.yaml
source activate deepclean

#======================================================
# INSTALL TENSORRT
# TODO: insert TRT installation here

#======================================================
# TODO: include as part of Deepclean setup
# INSTALL TRITON CLIENT LIBRARIES
TRITON_DIR=$PWD/triton
CLIENT_DIR=${TRITON_DIR}/client
if [[ ! -d $CLIENT_DIR ]]; then mkdir -p $CLIENT_DIR; fi

TAG=20.07
RELEASE=$(curl -s https://raw.githubusercontent.com/NVIDIA/triton-inference-server/r${TAG}/VERSION)
wget -O ${CLIENT_DIR}/clients.tar.gz https://github.com/NVIDIA/triton-inference-server/releases/download/v${RELEASE}/v${RELEASE}_ubuntu1804.clients.tar.gz

cd $CLIENT_DIR
tar xzf clients.tar.gz
pip install --upgrade python/triton*.whl
cd -

#======================================================
# INSTALL DEEPCLEAN
pip install git+https://git.ligo.org/alec.gunny/deepclean-prod.git@client-demo
