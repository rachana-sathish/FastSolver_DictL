import os
import time
import scipy.io
import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from scipy.io import loadmat
from omp import run_omp
from utils import clear_dict, normalize_dictionary, count_matching_atoms
from rearrange_dictionary import permute_scale_dictionary

torch.manual_seed(0)
np.random.seed(0)

class NetD(nn.Module):
    '''
    Zero-bias Fully-connected Neural Network (ZBFCNN)
    k = no. of dictionary atoms
    m = dimension of an atom
    '''
    def __init__(self,k=50,m=20):
        super(NetD, self).__init__()
        self.fc1 = nn.Linear(k,m,bias=False)
    def forward(self,inputs):
        return self.fc1(inputs)

def train_net(net, optimizer, criterion, data, codes, total_epochs):
    '''
    Function for training ZBFCNN
    '''
    net.train()
    avg_loss = 0.0
    for epoch_num in range(total_epochs):
        sample_x_hat = net(codes)
        optimizer.zero_grad()
        recon_loss = criterion(sample_x_hat,data.t())
        avg_loss += recon_loss.item()
        recon_loss.backward()
        optimizer.step()
    avg_loss = avg_loss/total_epochs
    return net, optimizer, avg_loss

def normalize_dictionary(dictionary):
    dictionary_temp = dictionary @ torch.diag(torch.sign(dictionary[0,:]))
    return dictionary_temp/torch.linalg.norm(dictionary_temp,dim=0)

def update_net_weights(net, dictionary):
    '''
    Load normalized dictionary as  weights of ZBFCNN
    '''
    state_dict = net.state_dict()
    state_dict['fc1.weight'] = dictionary
    net.load_state_dict(state_dict)
    return net

def main():
    num_trials = 50
    success_rate = 0.0
    time_consumed = np.zeros(num_trials)
    num_detected_atoms = np.zeros(num_trials)
    for trial in range(1,num_trials+1):
        print(f"Trial {trial}/{num_trials}")
        data_load_path = '2_convergence/dataset/trial_'+str(trial)+'/'
        init_dict_load_path = '2_convergence/results/ksvd_trial_'+str(trial)+'/'
        save_path = '2_convergence/results/FastSolver_trial_'+str(trial)+'/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Load generation data dictionary and dataset
        gen_data = torch.Tensor(scipy.io.loadmat(data_load_path+'gen_data.mat')['X']).float().to(DEVICE)
        gen_dictionary_norm = torch.Tensor(scipy.io.loadmat(data_load_path+'gen_dictionary.mat')['D']).float()
        dict_size = gen_dictionary_norm.shape[1]
       
        # Initialize dictionary (same as K-SVD)
        init_dictionary_norm = torch.load(init_dict_load_path+'init_dict.pt')
        # init_dictionary_norm = torch.Tensor(loadmat(init_dict_load_path+'init_dict.mat')['D']).to(DEVICE)

        # Load initial dictionary as weights of ZBFCNN
        net = NetD().to(DEVICE)
        net = update_net_weights(net, init_dictionary_norm)

        net_lr = 1e-2 # Learning rate for NN
        total_epochs = 100 # Epochs for training NN
        total_rounds = 80 # No. of iterations of alternating minimization        
        optimizer = optim.Adam(net.parameters(),lr=net_lr)
        recon_criterion = nn.MSELoss()
        learnt_dictionary_norm = net.fc1.weight.data # Initial dictionary
        start_time = time.time()
        for round_num in range(total_rounds):
            # Optimize code
            learnt_codes = run_omp(learnt_dictionary_norm, gen_data.T, n_nonzero_coefs=3, alg="v0")
            net, optimizer, _ = train_net(net, optimizer, recon_criterion,
                                                 gen_data, learnt_codes, total_epochs)
            learnt_dictionary = net.fc1.weight.data
            # Normalize dictionary
            learnt_dictionary_norm = learnt_dictionary/torch.linalg.norm(learnt_dictionary,dim=0)
            
            # Load modified dictionary as model weights
            net = update_net_weights(net, learnt_dictionary_norm)
            # break
        
        end_time = time.time() - start_time
        time_consumed[trial-1] = end_time

        # Save learnt dictionary and codes
        learnt_dictionary_norm_np = net.fc1.weight.data
        if DEVICE == 'cuda':
            learnt_dictionary_norm_np = learnt_dictionary_norm_np.cpu()
            learnt_codes = learnt_codes.cpu()
        learnt_dictionary_norm = learnt_dictionary_norm_np.numpy()
        # Permute and re-scale dictionary atoms
        learnt_dictionary_norm = permute_scale_dictionary(learnt_dictionary_norm,gen_dictionary_norm.numpy())
        np.save(save_path+'D_FastSolver',learnt_dictionary_norm)
        np.save(save_path+'Z_FastSolver',learnt_codes.T)

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

        net_learnt_wts = net.fc1.weight.data
        net_init_wts = init_dictionary_norm
        if DEVICE == 'cuda':
            net_init_wts = net_init_wts.cpu()
            net_learnt_wts = net_learnt_wts.cpu()
        net_init_wts = net_init_wts.numpy()
        net_learnt_wts = net_learnt_wts.numpy()
        plt.figure()
        plt.imshow(net_init_wts-net_learnt_wts,cmap='bwr')
        plt.axis('off')
        im_ratio = net_init_wts.shape[0]/net_init_wts.shape[1]
        plt.colorbar(fraction=0.047*im_ratio)
        plt.title('Change in Net weights',fontsize=20)
        plt.savefig(save_path+'Net_wt_change.png',transparent=True,bbox_inches='tight')
        plt.close()

        # Compare learnt dictionary (D) with generation dictionary (gen_dictionary)
        # Compare each atom in generating dictionary with all atoms in learnt dictionary
        num_detected_atoms[trial-1] = count_matching_atoms(gen_dictionary_norm,
                                                        learnt_dictionary_norm)
        print(f"Trial: {trial} | Success rate: {100*num_detected_atoms[trial-1]/dict_size}")
        success_rate += 100*num_detected_atoms[trial-1]/dict_size
    np.save('2_convergence/results/fastsolver_num_detected_atoms',num_detected_atoms)
    np.save('2_convergence/results/fastsolver_time_consumed',time_consumed)
    print(f"Average success rate: {success_rate/num_trials}")
    avg_time_consumed = np.mean(time_consumed)
    print(f"Avearge time: {avg_time_consumed//60} m {avg_time_consumed%60} s")

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'
    main()
