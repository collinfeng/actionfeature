FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

# install python3-pip
RUN apt update && apt install python3-pip git vim sudo curl wget apt-transport-https ca-certificates gnupg libgl1 -y 
RUN pip install --upgrade pip
RUN pip install setuptools
# install dependencies via pip
# Only install jax/jaxlib to version 0.4.1 for st
# RUN pip3 install --upgrade "jax[cuda12_pip]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade "jax[cuda12_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install tensorflow-probability==0.16.0 chex celluloid brax==0.0.12 evosax flax gym==0.21.0 notebook matplotlib optax==0.1.4 wandb torch torchvision tqdm gymnax jupyter ipython orbax-export
RUN pip3 install -U dm-haiku

ARG UID
RUN useradd -u $UID --create-home duser && \
    echo "duser:duser" | chpasswd && \
    adduser duser sudo
USER duser
WORKDIR /home/duser/

RUN git config --global user.email "collinfeng2001@gmail.com"
RUN git config --global user.name "collinfeng"

