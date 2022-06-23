'''
Based on the MATLAB toolbox ksvdbox13 by
Ron Rubinstein
Computer Science Department
Technion, Haifa 32000 Israel
ronrubin@cs
May 2009
http://www.cs.technion.ac.il/~ronrubin/software.html
'''
import os
import time
import scipy.io
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from omp import run_omp
from utils import clear_dict, optimize_atom, normalize_dictionary, count_matching_atoms, colnorms_squared, replace_atoms
from rearrange_dictionary import permute_scale_dictionary

torch.manual_seed(0)
np.random.seed(0)

def main():    
    success_rate = 0.0
    muthresh = 0.99
    time_consumed = np.zeros(NUM_TRIALS)
    num_detected_atoms = np.zeros(NUM_TRIALS)
    for trial in range(1,NUM_TRIALS+1):
        print(f"Trial {trial}/{NUM_TRIALS}")
        data_load_path = os.path.join('2_convergence','dataset','samples_'+str(NUM_SAMPLES)+'_trial_'+str(trial))
        save_path = os.path.join('2_convergence','results',DEVICE,'ksvd_'+'samples_'+str(NUM_SAMPLES)+'_trial_'+str(trial))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Load generation data dictionary and dataset
        gen_data = torch.Tensor(scipy.io.loadmat(os.path.join(data_load_path,'gen_data.mat'))['X']).float().to(DEVICE)
        gen_dictionary_norm = torch.Tensor(scipy.io.loadmat(os.path.join(data_load_path,'gen_dictionary.mat'))['D']).float()
        dict_size = gen_dictionary_norm.shape[1]

        # Initialize dictionary
        data_ids = (colnorms_squared(gen_data,device=DEVICE) > 1e-6).nonzero()   # ensure no zero data elements are chosen
        perm = torch.randperm(len(data_ids)).to(DEVICE)
        use_idx = list(data_ids[perm[:dict_size]])
        init_dictionary_norm = gen_data[:,use_idx]
        init_dictionary_norm /= torch.linalg.norm(init_dictionary_norm,axis=0)
        torch.save(init_dictionary_norm,os.path.join(save_path,'init_dict.pt'))

        total_rounds = 80 # No. of iterations of alternating minimization
        learnt_dictionary = init_dictionary_norm        

        start_time = time.time()
        for round_num in range(total_rounds):
            # Optimize code
            learnt_codes = run_omp(learnt_dictionary, gen_data.T, n_nonzero_coefs=3, alg="v0").T

            replaced_atoms = torch.zeros(dict_size).to(DEVICE)  
            unused_sigs = torch.arange(gen_data.shape[1]).to(DEVICE)            

            # Dictionary update
            p = torch.randperm(dict_size).to(DEVICE)
            for j in range(dict_size):
                (learnt_dictionary[:,p[j]],learnt_codes_j,data_indices,unused_sigs,replaced_atoms) = optimize_atom(gen_data,learnt_dictionary,p[j],learnt_codes,unused_sigs,replaced_atoms,device=DEVICE)
                learnt_codes[p[j],data_indices] = learnt_codes_j
            # replace atoms
            learnt_dictionary, learnt_codes, replaced_atoms, unused_sigs = replace_atoms(learnt_dictionary,learnt_codes,gen_data,replaced_atoms,unused_sigs)
            # clear dictionary
            (learnt_dictionary,_) = clear_dict(learnt_dictionary,learnt_codes,gen_data,muthresh,unused_sigs,replaced_atoms)
        end_time = time.time()
        total_time = end_time - start_time
        time_consumed[trial-1] = total_time
        print(f"Time consumed: {total_time:.3f}s")

        # Save learnt dictionary and codes
        if DEVICE == 'cuda':
            learnt_dictionary_np = learnt_dictionary.cpu().numpy()
            learnt_codes = learnt_codes.cpu()
        else:
            learnt_dictionary_np = learnt_dictionary.numpy()
        learnt_dictionary_norm = learnt_dictionary_np/np.linalg.norm(learnt_dictionary_np,axis=0)
        # Permute and re-scale dictionary atoms
        learnt_dictionary_norm = permute_scale_dictionary(learnt_dictionary_norm,gen_dictionary_norm.numpy())
        np.save(os.path.join(save_path,'D_KSVD.npy'),learnt_dictionary_norm)
        np.save(os.path.join(save_path,'Z_KSVD.npy'),learnt_codes.T)

        # Visualize dictionary
        gen_dictionary_norm = normalize_dictionary(gen_dictionary_norm).numpy()
        learnt_dictionary_norm = normalize_dictionary(torch.Tensor(learnt_dictionary_norm)).numpy()

        dict_min_val = min(gen_dictionary_norm.min(), learnt_dictionary_norm.min())
        dict_max_val = max(gen_dictionary_norm.max(), learnt_dictionary_norm.max())
        fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12,4))
        fig.suptitle('Dictionaries',fontsize=22, y=0.8)
        img = axes[0].imshow(gen_dictionary_norm, vmin=dict_min_val, vmax=dict_max_val, cmap='jet')
        axes[0].set_title('Reference',fontsize=20)
        img = axes[1].imshow(learnt_dictionary_norm, vmin=dict_min_val, vmax=dict_max_val, cmap='jet')
        axes[1].set_title('Learnt',fontsize=20)
        img = axes[2].imshow(abs(gen_dictionary_norm-learnt_dictionary_norm),cmap='bwr')
        axes[2].set_title('Difference',fontsize=20)
        im_ratio = gen_dictionary_norm.shape[0]/gen_dictionary_norm.shape[1]
        fig.colorbar(img,fraction=0.047*im_ratio)
        plt.savefig(os.path.join(save_path,'dictionary_comparison.png'),transparent=True,bbox_inches='tight')
        plt.close()
        
        # Compare learnt dictionary (D) with generation dictionary (gen_dictionary)
        # Compare each atom in generating dictionary with all atoms in learnt dictionary
        num_detected_atoms[trial-1] = count_matching_atoms(gen_dictionary_norm,
                                                        learnt_dictionary_norm)
        print(f"Trial: {trial} | Success rate: {100*num_detected_atoms[trial-1]/dict_size}")
        success_rate += 100*num_detected_atoms[trial-1]/dict_size
    np.save(os.path.join('2_convergence','results','ksvd_num_detected_atoms.npy'),num_detected_atoms)
    np.save(os.path.join('2_convergence','results','ksvd_time_consumed.npy'),time_consumed)
    print(f"Average success rate: {success_rate/NUM_TRIALS}")
    avg_time_consumed = np.mean(time_consumed)
    print(f"Avearge time: {avg_time_consumed//60} m {avg_time_consumed%60} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda',type=str)
    parser.add_argument('--num_trials',default=50,type=int)
    parser.add_argument('--num_samples',default=200000,type=int)
    args = parser.parse_args()
    DEVICE = args.device
    NUM_TRIALS = args.num_trials
    NUM_SAMPLES = args.num_samples
    main()
