## Environment Setup

1. Install CUDA 10 and cuDNN 7.
2. Install miniconda.
3. Setup the conda environment: `conda env create -f env.yml`
4. Install TensorFlow: `pip install -U tensorflow-gpu==1.13.1`


## To reproduce figs:

All necessary scripts should be in code_figs, and all necessary data should be in data. We included data of the summary stats for plotting, since the full microdata cannot be released. 