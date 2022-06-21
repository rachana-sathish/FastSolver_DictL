import os
import time
import scipy.io
import numpy as np
import torch
import matplotlib.pyplot as plt
from omp import run_omp
from utils import clear_dict, optimize_atom, normalize_dictionary, count_matching_atoms, colnorms_squared
from rearrange_dictionary import permute_scale_dictionary

torch.manual_seed(0)
np.random.seed(0)

def main():
    num_trials = 50
    success_rate = 0.0
    muthresh = 0.99
    time_consumed = np.zeros(num_trials)
    num_detected_atoms = np.zeros(num_trials)
    for trial in range(1,num_trials+1):
        print(f"Trial {trial}/{num_trials}")
        data_load_path = '2_convergence/dataset/trial_'+str(trial)+'/'
        save_path = '2_convergence/results/ksvd_trial_'+str(trial)+'/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Load generation data dictionary and dataset
        gen_data = torch.Tensor(scipy.io.loadmat(data_load_path+'gen_data.mat')['X']).float().to(DEVICE)
        gen_dictionary_norm = torch.Tensor(scipy.io.loadmat(data_load_path+'gen_dictionary.mat')['D']).float()
        dict_size = gen_dictionary_norm.shape[1]

        # Initialize dictionary
        data_ids = (colnorms_squared(gen_data,device=DEVICE) > 1e-6).nonzero()   # ensure no zero data elements are chosen
        perm = torch.randperm(len(data_ids)).to(DEVICE)
        use_idx = list(data_ids[perm[:dict_size]])
        init_dictionary_norm = gen_data[:,use_idx]
        init_dictionary_norm /= torch.linalg.norm(init_dictionary_norm,axis=0)
        torch.save(init_dictionary_norm,save_path+'init_dict.pt')

        total_rounds = 80 # No. of iterations of alternating minimization
        learnt_dictionary = init_dictionary_norm        
        replaced_atoms = torch.zeros(dict_size).to(DEVICE)  
        all_idx = list(np.arange(gen_data.shape[1]))
        unused_idx = [idx for idx in all_idx if idx not in use_idx]
        unused_sigs = torch.Tensor(unused_idx).long().to(DEVICE)
        torch.save(unused_sigs,save_path+'unused_sigs.pt')
        start_time = time.time()
        for round_num in range(total_rounds):
            # Optimize code
            learnt_codes = run_omp(learnt_dictionary, gen_data.T, n_nonzero_coefs=3, alg="v0").T

            # Dictionary update
            p = torch.randperm(dict_size).to(DEVICE)
            for j in range(dict_size):
                (learnt_dictionary[:,p[j]],learnt_codes_j,data_indices,unused_sigs,replaced_atoms) = optimize_atom(gen_data,learnt_dictionary,p[j],learnt_codes,unused_sigs,replaced_atoms,device=DEVICE)
                learnt_codes[p[j],data_indices] = learnt_codes_j
            # clear dictionary
            (learnt_dictionary,_) = clear_dict(learnt_dictionary,learnt_codes,gen_data,muthresh,unused_sigs,replaced_atoms)
        end_time = time.time()
        total_time = end_time - start_time
        time_consumed[trial-1] = total_time

        # Save learnt dictionary and codes
        if DEVICE == 'cuda':
            learnt_dictionary_np = learnt_dictionary.cpu()
            learnt_codes = learnt_codes.cpu()
        learnt_dictionary_np = learnt_dictionary_np.numpy()
        learnt_dictionary_norm = learnt_dictionary_np/np.linalg.norm(learnt_dictionary_np,axis=0)
        # Permute and re-scale dictionary atoms
        learnt_dictionary_norm = permute_scale_dictionary(learnt_dictionary_norm,gen_dictionary_norm.numpy())
        np.save(save_path+'D_KSVD',learnt_dictionary_norm)
        np.save(save_path+'Z_KSVD',learnt_codes.T)

        # Visualize dictionary
        gen_dictionary_norm = normalize_dictionary(gen_dictionary_norm).numpy()
        learnt_dictionary_norm = normalize_dictionary(torch.Tensor(learnt_dictionary_norm)).numpy()

        dict_min_val = min(gen_dictionary_norm.min(), learnt_dictionary_norm.min())
        dict_max_val = max(gen_dictionary_norm.max(), learnt_dictionary_norm.max())
        fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12,4))
        fig.suptitle('Dictionaries',fontsize=22, y=1.02)
        img = axes[0].imshow(gen_dictionary_norm, vmin=dict_min_val, vmax=dict_max_val, cmap='jet')
        axes[0].set_title('Reference',fontsize=20)
        img = axes[1].imshow(learnt_dictionary_norm, vmin=dict_min_val, vmax=dict_max_val, cmap='jet')
        axes[1].set_title('Learnt',fontsize=20)
        img = axes[2].imshow(abs(gen_dictionary_norm-learnt_dictionary_norm),cmap='bwr')
        axes[2].set_title('Difference',fontsize=20)
        im_ratio = gen_dictionary_norm.shape[0]/gen_dictionary_norm.shape[1]
        fig.colorbar(img,fraction=0.047*im_ratio)
        plt.savefig(save_path+'dictionary_comparison.png',transparent=True,bbox_inches='tight')
        plt.close()
        
        # Compare learnt dictionary (D) with generation dictionary (gen_dictionary)
        # Compare each atom in generating dictionary with all atoms in learnt dictionary
        num_detected_atoms[trial-1] = count_matching_atoms(gen_dictionary_norm,
                                                        learnt_dictionary_norm)
        print(f"Trial: {trial} | Success rate: {100*num_detected_atoms[trial-1]/dict_size}")
        success_rate += 100*num_detected_atoms[trial-1]/dict_size
    np.save('2_convergence/results/ksvd_num_detected_atoms',num_detected_atoms)
    np.save('2_convergence/results/ksvd_time_consumed',time_consumed)
    print(f"Average success rate: {success_rate/num_trials}")
    avg_time_consumed = np.mean(time_consumed)
    print(f"Avearge time: {avg_time_consumed//60} m {avg_time_consumed%60} s")

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    main()
