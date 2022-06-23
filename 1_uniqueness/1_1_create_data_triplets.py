import os
import numpy as np
from sklearn.datasets import make_sparse_coded_signal

def main(path):
    '''
    path = directory for saving the synthetic dataset
    Dataset - X, Dictionary - D, Sparse codes - Z
    Parameters: k=15, m=20, s=10 ==> n=33033
    '''
    X, D, Z = make_sparse_coded_signal(n_samples=33033, n_components=15, n_features=20,
                                    n_nonzero_coefs=10, random_state=42)
    print('Data ->',X.shape,'Dictionary ->',D.shape,'Code ->',Z.shape)

    # Save data triplets
    if not os.path.isdir(path):
        os.makedirs(path)
    np.save(os.path.join(path,'dataset.npy'),X)
    np.save(os.path.join(path,'dictionary.npy'),D)
    np.save(os.path.join(path,'sparse_code.npy'),Z)

if __name__ == "__main__":
    main(path=os.path.join('1_uniqueness','dataset'))
