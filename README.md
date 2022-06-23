# Linear Neural Network as a Fast Solver for Dictionary Learning

This repository contains codes for reproducing the results presented in the paper titled "Linear Neural Network as a Fast Solver for Dictionary Learning".

Requirements: Anaconda with Python 3.8, MATLAB 2022

Follow these instructions to run the codes:
  1. Clone this respository.
  2. Download the MATLAB toolboxes for [K-SVD](https://www.cs.technion.ac.il/~ronrubin/Software/ksvdbox13.zip) and [OMP](https://www.cs.technion.ac.il/~ronrubin/Software/ompbox10.zip) within the directory containing the downloaded codes. Compile the MATLAB toolboxes following the instructions available within them. ([Source](https://www.cs.technion.ac.il/~ronrubin/software.html))
  3. Move the codes within the folder "matlab_codes" to the folder of ksvd toolbox (ksvdbox13).
  4. Create a conda environment using requirements.txt  

**Codes for three experiments in the paper are provided in three numbered directories. Run them in the following order from within the FastSolver_DictL directory:**

###### Experiment 1: uniqueness
```
python -m 1_uniqueness.1_1_create_data_triplets
```
```
python -m 1_uniqueness.1_2_perturbation
```
###### Experiment 2: convergence
Run generate_synthetic_dataset.m using MATLAB
```
python -m 2_convergence.2_1_odl
```
```
python -m 2_convergence.2_2_ksvd
```
```
python -m 2_convergence.2_3_fastsolver
```
###### Experiment 3: denoising
Run generate_noisy_data.m using MATLAB
```
python -m 3_denoising.3_1_dictl_denoising_ksvd
```
```
python -m 3_denoising.3_2_dictl_denoising_fastsolver
```
Run image_denoise.m using MATLAB



