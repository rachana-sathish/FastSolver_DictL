# Linear Neural Network as a Fast Solver for Dictionary Learning

This repository contains codes for reproducing the results presented in the paper titled "Linear Neural Network as a Fast Solver for Dictionary Learning".

Follow these instructions to run the codes:
  1. Clone this respository.
  2. Download the MATLAB toolboxes for [K-SVD](https://www.cs.technion.ac.il/~ronrubin/Software/ksvdbox13.zip) and [OMP](https://www.cs.technion.ac.il/~ronrubin/Software/ompbox10.zip) within the directory containing the downloaded codes. Compile the MATLAB toolboxes following the instructions available within them. ([Source](https://www.cs.technion.ac.il/~ronrubin/software.html))
  3. Move the codes within the folder "matlab_codes" to the folder of ksvd toolbox (ksvdbox13).

Codes for three experiments in the paper are provided in three numbered directories. Run them in the following order from within the FastSolver_DictL directory:
```
python -m 1_uniqueness.1_1_create_data_triplets
```

