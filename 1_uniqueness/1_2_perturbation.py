import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit

def measure_error(z_orig, z_learnt):
    '''
    Calcuate error between original sparse code and the one 
    learnt using the perturbed dictioanary
    '''    
    return np.sum(abs(z_orig-z_learnt),axis=0)

def main():
    '''
    Add perturbation to generation dictionary and compute sparse codes
    '''
    # Load data triplet
    X_G = np.load('1_uniqueness/dataset/dataset.npy') # dataset
    D_G = np.load('1_uniqueness/dataset/dictionary.npy') # dictionary
    Z_G = np.load('1_uniqueness/dataset/sparse_code.npy') # sparse codes

    # Generate noise
    noise_level = 0.05
    save_path = '1_uniqueness/results/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    noise = np.random.uniform(-noise_level,noise_level,size=(20,15))
    np.save(save_path+'pertubation',noise)

    # Add noise to dictionary
    D_G_noisy = D_G + noise

    # Visualize dictionaries
    min_val = min([D_G.min(), D_G_noisy.min()])
    max_val = max([D_G.max(), D_G_noisy.max()])
    fig = plt.figure(figsize=[5,5])
    plt.imshow(D_G,cmap='jet',vmin=min_val,vmax=max_val)
    plt.xticks(list(np.arange(15)),list(np.arange(1,16)))
    plt.yticks([])
    plt.xlabel('Atoms',fontsize=20)
    plt.colorbar()
    plt.savefig(save_path+'D_G.png',transparent=True,bbox_inches='tight')

    # Visualize noisy data
    fig = plt.figure(figsize=[5,5])
    plt.imshow(D_G_noisy,cmap='jet',vmin=min_val,vmax=max_val)
    plt.xticks(list(np.arange(15)),list(np.arange(1,16)))
    plt.yticks([])
    plt.xlabel('Atoms',fontsize=20)
    plt.colorbar()
    plt.savefig(save_path+'D_G_noisy.png',transparent=True,bbox_inches='tight')

    # Visualize noise
    fig = plt.figure(figsize=[5,5])
    plt.imshow(noise,cmap='bwr')
    plt.xticks(list(np.arange(15)),list(np.arange(1,16)))
    plt.yticks([])
    plt.xlabel('Atoms',fontsize=20)
    plt.colorbar()
    plt.savefig(save_path+'noise.png',transparent=True,bbox_inches='tight')

    # Find sparse codes with perturbed dictionary
    Z = OrthogonalMatchingPursuit(n_nonzero_coefs=10,normalize=False).fit(D_G_noisy,X_G).coef_
    np.save(save_path+'detected_code',Z.T)

    # Compare detected sparse codes with original sparse codes
    plt.figure(figsize=[5,5])
    plt.bar(np.arange(15),Z_G[:,0])
    plt.xticks(list(np.arange(15)),list(np.arange(1,16)))
    plt.xlabel('Indices',fontsize=18)
    plt.ylabel('Coefficient',fontsize=18)
    plt.savefig(save_path+'original_sparse_code.png',transparent=True,bbox_inches='tight')

    # Visualize sparse codes
    plt.figure(figsize=[5,5])
    plt.bar(np.arange(15),Z[0])
    plt.xticks(list(np.arange(15)),list(np.arange(1,16)))
    plt.xlabel('Indices',fontsize=18)
    plt.ylabel('Coefficient',fontsize=18)
    plt.savefig(save_path+'detected_sparse_code.png',transparent=True,bbox_inches='tight')

    # Difference in codes
    code_diff = measure_error(Z_G,Z.T)
    print(f"Error in learnt sparse code | mean: {code_diff.mean():.3f} , std. dev.: {code_diff.std():.3f}")

    # Calculate reconstruction error
    X_hat = D_G @ Z.T
    mse_error = np.mean(np.sum((X_hat - X_G) ** 2, axis=0) / np.sum(X_G ** 2, axis=0))
    print(f"Reconstruction error: {mse_error}")

if __name__ == "__main__":
    main()
