## Computing Environment

The code for this project has been tested to work on a system with the following hardware and software specifications.

### Hardware

- CPU: Intel Xeon Silver 4110
- Hard Drive: 256GB SSD
- Memory: 125GB
- GPU: 1x NVIDIA Titan Xp


### Software

- Ubuntu 16.04.5 LTS
- miniconda3
- python 3.7.4
- pillow 5.4.1
- scikit-learn 0.20.3
- numpy 1.17.2
- scipy 1.3.1
- seaborn 0.9.0
- cartopy 0.17.0
- earthengine-api 0.1.213
- gdal 2.3.3
- jupyter 1.0.0
- matplotlib 3.1.1
- pandas 0.25.1
- tensorflow-gpu 1.13.1
- tqdm 4.31.1


## Environment Setup

1. Install CUDA 10 and cuDNN 7.
2. Install miniconda.
3. Setup the conda environment: `conda env create -f env.yml`
4. Install TensorFlow: `pip install -U tensorflow-gpu==1.13.1`


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

All necessary scripts should be in code_figs, and all necessary data should be in data. We included data of the summary stats for plotting, since the full microdata cannot be released.

For the maximally-activating activation maps, see the `model_analysis/max_activating.ipynb` notebook.