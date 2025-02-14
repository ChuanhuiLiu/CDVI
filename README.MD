<!-- ABOUT THE PROJECT -->
# Official PyTorch Implementation for “Censor-dependent Variational Inference”

Welcome to the code repository of Censor-dependent Variational Inference! This project provides an comprehensive analysis of Variational Inference for Survival Analysis tasks in presence of censored data. In particular, CD-CVAE is a Conditional Variational Autoencoder leveraging a censor-dependent variational distribution. For more details, see full paper [here](https://arxiv.org/pdf/2502.09591).


This repository provides:
* Scripts to train CD-CVAEs and other state-of-the-art models (CoxPH, DeepSurv, Random Survival Forest, Deep Cox Mixture) provided by [Pycox](https://github.com/havakv/pycox) and [Auton-Survival](https://github.com/autonlab/auton-survival) package.
* Scripts to obtain validate and test results (time-dependent C-index and IBS) in the paper that is compatible with PyCox.evaluation and Scikit-survival.metrics 
* Useful utilities:
   1. a Pytorch.distribution class for Gumbel_minimum distribution; 
   2. a unified data loader function for loading and preprocessing functions;
   3. some overriding functions on [Auton-Survival](https://github.com/autonlab/auton-survival).



<!-- Setup -->
## Getting Started

### Installation
Download the repo manually (as a .zip file) or clone it using Git.
```sh
git clone https://github.com/ChuanhuiLiu/CDVI
```
### Prerequisites 
The following module is required on your machine:
* Ubuntu/Windows
* Python 3.9
* Packages in requirements.txt

(Recommended) To create a conda environment, use the following command in anaconda prompt:
```sh
conda env create --name envname --file Environment.yml
```

If you have installed Python 3.9 already or want to use an existing environment, use the following command:
```sh
pip install -r requirements.txt
```


<!-- USAGE EXAMPLES -->
## Usage

See `Demo.ipynb` for a quick start. 

Our implementation is highly integrated with [Auton-Survival](https://github.com/autonlab/auton-survival) package. If you are not familiar with this package, please refer to the examples [here](https://nbviewer.org/github/autonlab/auton-survival/tree/master/examples/). We have several modifications via utilities functions under `\utils` folder.  



<!--CITATION-->
## Bibtex:
If you find this repo helpful，please consider cite our paper:

```
@misc{liu2025censordependentvariationalinference,
      title={Censor Dependent Variational Inference}, 
      author={Chuanhui Liu and Xiao Wang},
      year={2025},
      eprint={2502.09591},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.09591}, 
}
```
<p align="right">(<a href="#readme-top">Back to Top</a>)</p>
