import os
import argparse
import torch
import scipy.io
import numpy as np
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
from omp import run_omp
from utils import optimize_atom, clear_dict

def normalize_data(data):
    if not isinstance(data, torch.Tensor):
        data = torch.Tensor(data)
    data_norm_val = torch.linalg.norm(data,axis=0)
    return data/data_norm_val

def main(config):
    for sigma in config['sigmas']:
        for trial in config['trials']:
            print('Sigma:',sigma,'Trial:',trial)
            # Load initial dictionary
            load_path = '3_denoising/noisy_images/'+config['img_name']+'/sigma_'+str(sigma)+'_trial_'+str(trial)+'/'
            learnt_dict = torch.Tensor(scipy.io.loadmat(load_path+'init_dict.mat')['initdict']).float().to(config['device'])
            learnt_dict_norm = normalize_data(learnt_dict)
            # Load dataset
            gen_data = torch.Tensor(scipy.io.loadmat(load_path+'noisy_train_data.mat')['noisy_data']).float().to(config['device'])

            replaced_atoms = np.zeros(config['dict_size'])
            mu_thresh = 0.99

            save_path = '3_denoising/results/'+config['img_name']+'/sigma_'+str(sigma)+'_trial_'+str(trial)+'/ksvd/'
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            unused_data = np.arange(gen_data.shape[1])
            for round_num in tqdm(range(args.rounds)): 
                # Optimize code                    
                learnt_codes = run_omp(learnt_dict_norm, gen_data.T, n_nonzero_coefs=10, alg="v0").T
                
                # Update dictionary atoms
                replaced_atoms = torch.zeros(config['dict_size']).to(config['device'])  
                unused_data = torch.arange(gen_data.shape[1]).to(config['device'])  
                permuted_idx = torch.randperm(config['dict_size']).to(config['device'])
                for dict_idx in range(config['dict_size']):
                    (learnt_dict_norm[:,permuted_idx[dict_idx]],curr_code,data_indices,unused_data,replaced_atoms) = optimize_atom(gen_data, learnt_dict_norm,permuted_idx[dict_idx],learnt_codes,unused_data,replaced_atoms,device=device)                   
                    learnt_codes[permuted_idx[dict_idx],data_indices] = curr_code

                # Replace unused dictionary atoms
                learnt_dict_norm = normalize_data(learnt_dict) # Normalize dictionary
                for dict_idx in range(config['dict_size']):
                    data_indices = list(np.nonzero(learnt_codes.T[dict_idx,:])[0])
                    curr_code = learnt_codes.T[dict_idx,data_indices]
                    if (len(data_indices) < 1):
                        max_signals = 5000
                        perm = np.random.permutation(len(unused_data))
                        perm = list(perm[:min(max_signals,len(perm))])
                        error = sum((gen_data[:,unused_data[perm]] - learnt_dict_norm@learnt_codes.T[:,unused_data[perm]])**2)
                        max_err_idx = np.argmax(error)
                        atom = gen_data[:,unused_data[perm[max_err_idx]]]
                        atom = atom/np.linalg.norm(atom)
                        curr_code = np.zeros(curr_code.shape)
                        idx_list = list(np.arange(0,perm[max_err_idx])) + list(np.arange(perm[max_err_idx+1],len(perm)))
                        unused_data = unused_data[idx_list]
                        replaced_atoms[dict_idx] = 1 
                        learnt_dict_norm[:,dict_idx] = atom
                        learnt_codes.T[dict_idx,data_indices] = curr_code
                # clear dictionary
                if round_num < args.rounds-1:
                    learnt_dict_norm, unused_data = clear_dict(learnt_dict_norm,learnt_codes,gen_data,mu_thresh,unused_data,replaced_atoms)

            if config['device'] == 'cuda':
                learnt_dict_norm = learnt_dict_norm.cpu()
            learnt_dict_norm = learnt_dict_norm.numpy()
            np.save(save_path+'learnt_dict_norm',learnt_dict_norm)
            scipy.io.savemat(save_path+'learnt_dict_norm.mat',{'D': learnt_dict_norm})

            # Normalize dictionary
            learnt_dict_norm = normalize_data(learnt_dict_norm).numpy()
            # Visualize dictionary as a grid
            learnt_dict = np.expand_dims(learnt_dict_norm.T.reshape(-1,config['dict_grid'],config['dict_grid']),axis=1)
            learnt_dict_norm = (learnt_dict-learnt_dict.min())/(learnt_dict.max()-learnt_dict.min())
            grid_img = torchvision.utils.make_grid(torch.Tensor(learnt_dict_norm), nrow=16)

            # Visualize dictionary
            plt.figure(figsize=(10,10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.savefig(save_path+'learnt_dictionary.png',transparent=True,bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser()    
    parser.add_argument('--rounds',default=80,type=int)
    parser.add_argument('--img_name',default='barbara',type=str)
    args = parser.parse_args() 

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    config = {}
    config['rounds'] = args.rounds
    config['img_name'] = args.img_name
    config['dict_size'] = 256
    config['dict_grid'] = 8
    config['trials'] = [1,2,3,4,5]
    config['sigmas'] = [2,5,10,15,20,25,50,75,100]
    config['device'] = device
    main(config)

