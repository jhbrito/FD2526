# FD2526
MEEC FD 2025/2026

## Create conda environment
conda create -n FD2526Env310 -python=3.10

or

conda create -p PATH python=3.10

## Install packages
conda install jupyter

conda install matplotlib

conda install scikit-learn

## Export conda environment
conda env export > environment.yml

## Import conda environment
conda env create -n Project_Environment_Name --file environment.yml

## Import pip environment
pip install -r requirements.txt

## Export pip environment
pip freeze > requirements.txt

## Unofficial Windows Binaries for Python Extension Packages
<https://www.lfd.uci.edu/~gohlke/pythonlibs/>
