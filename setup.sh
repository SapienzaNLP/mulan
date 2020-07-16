#!/bin/bash

# setup conda
source ~/miniconda3/etc/profile.d/conda.sh

# create conda env
read -p "Enter environment name: " env_name
conda create -yn $env_name python=3.7
conda activate $env_name

# install torch and faiss
read -p "Enter cuda version (check that faiss supports the number your enter. As of April 2020, for 10+, enter 10.0): " cuda_version
conda install -y pytorch torchvision faiss-gpu cudatoolkit=$cuda_version -c pytorch

# install python requirements
pip install -r requirements.txt

# download nltk wordnet
python -c "import nltk; nltk.download('wordnet')"
