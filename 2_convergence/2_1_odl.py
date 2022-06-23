import os
import time
import scipy.io
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rearrange_dictionary import permute_scale_dictionary
from sklearn.decomposition import DictionaryLearning as DictL
from utils import normalize_dictionary, count_matching_atoms
np.random.seed(0)

def main():
    data_load_path = os.path.join('2_convergence','dataset')
    success_rate = 0.0
    num_detected_atoms = np.zeros(NUM_TRIALS)
    time_consumed = np.zeros(NUM_TRIALS)
    detection_rate = 0.0
    for trial in range(1,NUM_TRIALS+1):
        print(f"Trial {trial}/{NUM_TRIALS}")
        save_path = os.path.join('2_convergence','results','odl_trial_'+str(trial))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Load generation dictionary and data
        gen_dictionary_norm = scipy.io.loadmat(os.path.join(data_load_path,'trial_'+str(trial),
                                            'gen_dictionary.mat'))['D']
        dict_size = gen_dictionary_norm.shape[1]
        gen_data = scipy.io.loadmat(os.path.join(data_load_path,'trial_'+str(trial),'gen_data.mat'))['X']        
        start_time = time.time()
        # Learn dictionary
        dict_learner = DictL(n_components=NUM_ATOMS,fit_algorithm='lars', transform_algorithm='omp',
                                        transform_n_nonzero_coefs=SPARSITY, max_iter=80, random_state=0)
        # Learn sparse codes
        learnt_codes = dict_learner.fit_transform(gen_data.T)
        end_time = time.time() - start_time
        time_consumed[trial-1] = end_time
        learnt_dictionary_norm = dict_learner.components_.T

        # Permute and re-scale dictionary atoms
        learnt_dictionary_norm = permute_scale_dictionary(learnt_dictionary_norm,gen_dictionary_norm)
        # Save learnt dictionary and codes
        np.save(os.path.join(save_path,'D_ODL.npy'),learnt_dictionary_norm)
        np.save(os.path.join(save_path,'Z_ODL.npy'),learnt_codes)

        # Visualize dictionary
        gen_dictionary_norm = normalize_dictionary(gen_dictionary_norm).numpy()
        learnt_dictionary_norm = normalize_dictionary(learnt_dictionary_norm).numpy()
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
        detection_rate += num_detected_atoms[trial-1]/NUM_ATOMS
        print(f"Trial: {trial} | success rate: {100*num_detected_atoms[trial-1]/NUM_ATOMS}")
        success_rate += 100*num_detected_atoms[trial-1]/dict_size   

    np.save(os.path.join('2_convergence','results','odl_num_detected_atoms.npy'),num_detected_atoms)
    np.save(os.path.join('2_convergence','results','odl_time_consumed.npy'),time_consumed)
    print(f"Average success rate: {success_rate/NUM_TRIALS}")
    avg_time_consumed = np.mean(time_consumed)
    print(f"Avearge time: {avg_time_consumed//60} m {avg_time_consumed%60} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trials',default=50,type=int)
    parser.add_argument('--num_atoms',default=50,type=int)
    parser.add_argument('--sparsity',default=3,type=int)
    args = parser.parse_args()
    NUM_TRIALS = args.num_trials
    NUM_ATOMS = args.num_atoms # k
    SPARSITY = args.sparsity # s
    main()
