## Using publicly available satellite imagery and deep learning to understand economic well-being in Africa

This repository includes the code and data necessary to reproduce the results and figures for the article "Using publicly available satellite imagery and deep learning to understand economic well-being in Africa" published in *Nature Communications* on May 22, 2020 ([link](https://www.nature.com/articles/s41467-020-16185-w)).

Please cite this article as follows, or use the BibTeX entry below.

> Yeh, C., Perez, A., Driscoll, A. *et al*. Using publicly available satellite imagery and deep learning to understand economic well-being in Africa. *Nat Commun* **11**, 2583 (2020). https://doi.org/10.1038/s41467-020-16185-w

```tex
@article{yeh2020using,
    author = {Yeh, Christopher and Perez, Anthony and Driscoll, Anne and Azzari, George and Tang, Zhongyi and Lobell, David and Ermon, Stefano and Burke, Marshall},
    day = {22},
    doi = {10.1038/s41467-020-16185-w},
    issn = {2041-1723},
    journal = {Nature Communications},
    month = {5},
    number = {1},
    title = {{Using publicly available satellite imagery and deep learning to understand economic well-being in Africa}},
    url = {https://www.nature.com/articles/s41467-020-16185-w},
    volume = {11},
    year = {2020}
}
```


## Hardware and Software Requirements

This code was tested on a system with the following specifications:

- operating system: Ubuntu 16.04.6 LTS
- CPU: Intel Xeon Silver 4110
- memory (RAM): 125GB
- disk storage: 500GB
- GPU: 1x NVIDIA Titan Xp

The main software requirements are Python 3.7 with TensorFlow r1.15, and R 3.6. The complete list of required packages and library are listed in the `env.yml` file, which is meant to be used with `conda` (version 4.8.3). See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for instructions on installing conda via Miniconda. Once conda is installed, run the following command to set up the conda environment:

```bash
conda env create -f env.yml
```

If you are using a GPU, you may need to also install CUDA 10 and cuDNN 7.


## Data Preparation Instructions

1. **Export satellite images from Google Earth Engine.** Follow the instructions in the `download/export_ee_images.ipynb` notebook.
2. **Process the satellite images.** Follow the instructions in the `preprocessing/process_tfrecords_dhs.ipynb` and `preprocessing/process_tfrecords_lsms.ipynb` notebooks. Then run the `preprocessing/analyze_tfrecords_dhs.ipynb` and `preprocessing/analyze_tfrecords_lsms.ipynb` notebooks.
3. **Prepare the data files.** Follow the instructions in the `data_analysis/dhs.ipynb` and `data_analysis/lsms.ipynb` notebooks.


## Model Training Instructions

1. **Run the baseline linear models.** Follow the instructions in `models/dhs_baselines.ipynb`, `models/lsms_baselines.ipynb`, , and `models/lsmsdelta_baselines.ipynb`.
2. **Train the convolutional neural network models.** If running this code on a SLURM-enabled computing cluster, run the scripts `train_directly_runner.py` and `train_directly_lsm_runner.py`. Otherwise, run `train_directly.py` and `train_delta.py` with the desired command-line arguments to set hyperparameters.
3. **Extract learned feature representations.** Run the scripts `extract_features_dhs.py` and `extract_features_lsmsdelta.py`.
4. **Run cross-validated ridge-regression.** Follow the instructions in `models/dhs_ridge_resnet.ipynb` and `model_analysis/lsmsdelta_resnet.ipynb`.


## To reproduce figs:

All necessary scripts should be in code_figs, and all necessary data should be in data. We included data of the summary stats for plotting, since the full microdata cannot be released. A few changes have been made to data and code to fix errors in plots. Code has been updated to fix an artificially inflated revisit rate for DigitalGlobe in Figure 1. Changes were made to the data on survey frequency used for Figure 1.

For the maximally-activating activation maps, see the `model_analysis/max_activating.ipynb` notebook.
