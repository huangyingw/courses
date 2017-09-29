#!/bin/bash - 
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

# This script is designed to work with ubuntu 16.04 LTS

# ensure system is updated and has basic build tools
apt-get update
apt-get --assume-yes upgrade
apt-get --assume-yes install tmux build-essential gcc g++ make binutils
apt-get --assume-yes install software-properties-common

# download and install GPU drivers
if [ ! -f "cuda-repo-ubuntu1604_8.0.44-1_amd64.deb" ]
then
    wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb" -O "cuda-repo-ubuntu1604_8.0.44-1_amd64.deb"
fi 

dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
apt-get update
apt-get -y install cuda
modprobe nvidia
nvidia-smi

# install Anaconda for current user
mkdir downloads
cd downloads
if [ ! -f "Anaconda2-4.2.0-Linux-x86_64.sh" ]
then
    wget "https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh" -O "Anaconda2-4.2.0-Linux-x86_64.sh"
fi 
bash "Anaconda2-4.2.0-Linux-x86_64.sh" -b -p /media/volgrp/anaconda2

echo "export PATH=\"$HOME/anaconda2/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/anaconda2/bin:$PATH"
conda install -y bcolz
conda upgrade -y --all

# install and configure theano
pip install theano
echo "[global]
device = gpu
floatX = float32

[cuda]
root = /usr/local/cuda" > ~/.theanorc

# install and configure keras
pip install keras==1.2.2
mkdir ~/.keras
echo '{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}' > ~/.keras/keras.json

# install cudnn libraries
if [ ! -f "cudnn.tgz" ]
then
    wget "http://files.fast.ai/files/cudnn.tgz" -O "cudnn.tgz"
fi 
tar -zxf cudnn.tgz
cd cuda
cp lib64/* /usr/local/cuda/lib64/
cp include/* /usr/local/cuda/include/

# configure jupyter and prompt for password
jupyter notebook --generate-config
jupass=`python -c "from notebook.auth import passwd; print(passwd())"`
echo "c.NotebookApp.password = u'"$jupass"'" >> $HOME/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py

# clone the fast.ai course repo and prompt to start notebook
cd ~
git clone https://github.com/fastai/courses.git
echo "\"jupyter notebook\" will start Jupyter on port 8888"
echo "If you get an error instead, try restarting your session so your $PATH is updated"
