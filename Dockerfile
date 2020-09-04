ARG TAG=20.07
FROM nvcr.io/nvidia/pytorch:${TAG}-py3
ARG TAG

# necessary for condor
RUN mkdir -p /cvmfs /hdfs /gpfs /ceph /hadoop /etc/condor

# add dependencies
ADD . deepclean-prod/
RUN RELEASE=$(curl -s https://raw.githubusercontent.com/NVIDIA/triton-inference-server/r${TAG}/VERSION) && \
        mkdir /opt/clients && \
        wget -O /opt/clients/clients.tar.gz https://github.com/NVIDIA/triton-inference-server/releases/download/v${RELEASE}/v${RELEASE}_ubuntu1804.clients.tar.gz && \
        cd /opt/clients && \
        tar -xzf clients.tar.gz && \
        cd - && \
        pip install --upgrade \
            bokeh gwpy ./deepclean-prod /opt/clients/python/triton*.whl
