FROM mambaorg/micromamba:2.4-cuda12.9.1-ubuntu22.04
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_DOCKERFILE_ACTIVATE=1 \
    MAMBA_DOCKERFILE_SHELL=bash \
    PATH=/opt/conda/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    build-essential cmake \
    zlib1g-dev libcurl4-openssl-dev \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

COPY . /opt/DeepPBS
WORKDIR /opt/DeepPBS

RUN micromamba create -y -n deeppbs python=3.11 && \
    micromamba run -n deeppbs pip install torch==2.8 torchvision -i https://download.pytorch.org/whl/cu129 && \
    micromamba run -n deeppbs pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html && \
    micromamba run -n deeppbs pip install biopython==1.86 matplotlib==3.10.7 pandas==2.3.3 logomaker networkx pdb2pqr scikit-learn==1.7.2 scipy==1.16.3 seaborn freesasa gradio && \
    micromamba run -n deeppbs pip install -e . && \
    micromamba clean -afy

CMD [ "bash" ]
