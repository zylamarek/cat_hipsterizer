#!/bin/bash

conda create -n cat_hipsterizer pip python=3.7 -y
source activate cat_hipsterizer
conda install -c anaconda keras-gpu==2.2.4 cudnn=7.6.0=cuda9.0_0 -y
pip install -r requirements.txt
