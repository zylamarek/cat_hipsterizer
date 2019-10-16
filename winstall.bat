@echo off
PATH = %PATH%;%USERPROFILE%\Miniconda3\Scripts
conda create -n cat_hipsterizer pip python=3.7 -y
call activate cat_hipsterizer
conda install -c anaconda keras-gpu==2.2.4 cudnn=7.6.0=cuda9.0_0 -y
pip install -r requirements.txt
