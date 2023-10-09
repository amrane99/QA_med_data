# Quality Assurance for MRI Data

Using this repository, five artifact classifiers can be trained, evaluated and used for inference to assure the quality of Cardiac MRI scans. The cassifiers capture the five most commonly known artifacts: blurring, gaussian noise, ghosting, motion and spike artifacts.


## Installation
The simplest way to install all dependencies is by using [Anaconda](https://conda.io/projects/conda/en/latest/index.html):

1. Create a Python3.8 environment as follows: `conda create -n <your_anaconda_env> python=3.8` and activate the environment.
2. Install CUDA and PyTorch through conda with the command specified by [PyTorch](https://pytorch.org/). The command for Linux using the CUDA Toolkit 11.8 was at the time `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`. The code was developed and last tested with the PyTorch version 2.0.1.
3. Navigate to the project root (where setup.py lives).
4. Execute `pip install -r requirements.txt` to install all required packages.
5. Set your paths in mp.paths.py.
6. Execute `git update-index --assume-unchanged mp/paths.py` so that changes in the paths file are not tracked in the repository.
7. Optional: Execute `pytest` to ensure that everything is working. Note that one of the tests will test whether at least one GPU is present, if you do not wish to test this mark to ignore. The same holds for tests that used datasets that have to be previously downloaded.


## Training Evaluation or Inference
To train, evaluate or use the presented artifact classifiers from this repository, please refer to the corresponding [documentation](documentations/JIP.md).