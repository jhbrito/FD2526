# FD2526
MEEC FD 2025/2026

## Create conda environment
```
conda create -n FD2526Env310 python=3.10
```

or
```
conda create -p PATH python=3.11
```

```
conda activate FD2526Env310
```

## Install packages
```
conda install jupyter

conda install matplotlib 

conda install pandas openpyxl

conda install tqdm numba

conda install scikit-learn
```
## Install TensorFlow

```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
### Anything above 2.10 is not supported on the GPU on Windows Native
```
pip install "tensorflow<2.11"
```
### Verify the installation:
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Export conda environment
```
conda env export > environment.yml
```
## Import conda environment
```
conda env create -n Project_Environment_Name --file environment.yml
```
## Import pip environment
```
pip install -r requirements.txt
```
## Export pip environment
```
pip freeze > requirements.txt
```
## Unofficial Windows Binaries for Python Extension Packages
<https://www.lfd.uci.edu/~gohlke/pythonlibs/>
