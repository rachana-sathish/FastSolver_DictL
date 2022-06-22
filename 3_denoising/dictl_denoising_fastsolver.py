import os
import argparse
import scipy.io
import numpy as np
from tqdm import tqdm
import torch
from  torch import nn, optim
import torchvision
from omp import run_omp
import matplotlib.pyplot as plt

# Dictionary Network
class DictLearner(nn.Module):
    def __init__(self,dict_size):
        super(DictLearner,self).__init__()
        self.fc1 = nn.Linear(dict_size,64,bias=False)
    def forward(self,x):
        x = self.fc1(x)
        return x

def train_nn(net,learnt_codes,optimizer,criterion,train_data,config):
    '''
    Train ZBFCNN
    '''
    net.train()
    for _ in range(config['epochs']):
        sample_x_hat = net(learnt_codes)
        optimizer.zero_grad()
        recon_loss = criterion(sample_x_hat,train_data.T)
        recon_loss.backward()
        optimizer.step()
    return net


def main(config):    
    sigma_nums = config['sigmas']
    trial_nums = config['trials']
    for sigma in sigma_nums:
        for trial in trial_nums:
            print('Sigma:',sigma,'Trial:',trial)
            load_path = '3_denoising/noisy_images/'+config['img_name']+'/sigma_'+str(sigma)+'_trial_'+str(trial)+'/'
            # Create direcotry for saving results
            save_path = '3_denoising/results/'+config['img_name']+'/sigma_'+str(sigma)+'_trial_'+str(trial)+'/fastsolver/'
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            # Load dataset
            gen_data = torch.Tensor(scipy.io.loadmat(load_path+'noisy_train_data.mat')['noisy_data']).float().to(config['device'])
            
            # Load hyper-parameters
            lr_dict = config['lr'] # learning rate for training NN
            total_rounds = config['rounds'] # no. of rounds of alternating minimization
            dict_size = config['dict_size']
            # Initialize neural network
            net = DictLearner(dict_size).to(device)
            if config['init'] == 'odct':
                # Load ODCT based dictionary initialization
                load_path = '3_denoising/noisy_images/'+config['img_name']+'/sigma_'+str(sigma)+'_trial_'+str(trial)+'/'
                learnt_dict = torch.Tensor(scipy.io.loadmat(load_path+'init_dict.mat')['initdict']).float().to(config['device'])
                orig_sd = net.state_dict()
                orig_sd['fc1.weight'] = learnt_dict
                net.load_state_dict(orig_sd)
            net.to(device)
            dict_optimizer = optim.Adam(net.parameters(),lr=lr_dict)
            recon_criterion = nn.MSELoss()
            # Initial dictionary
            learnt_dict_norm = net.fc1.weight.data.clone()
           
            # Alternating minimization
            for _ in tqdm(range(total_rounds)):
                # Optimize code
                learnt_codes = run_omp(learnt_dict_norm, gen_data.T, n_nonzero_coefs=10, alg="v0")
                # Train neural network
                net = train_nn(net,learnt_codes,dict_optimizer,recon_criterion,
                                                gen_data,config)
                learnt_dict_norm = net.fc1.weight.data
                learnt_dict_norm = learnt_dict_norm/torch.linalg.norm(learnt_dict_norm,dim=0)
               
                # Load modified dictionary as model weights
                net_sd = net.state_dict()
                net_sd['fc1.weight'] = learnt_dict_norm
                net.load_state_dict(net_sd)

            learnt_dict = net.fc1.weight.data
            if device == 'cuda':
                learnt_dict = learnt_dict.cpu()
            learnt_dict = learnt_dict.numpy()
            np.save(save_path+'learnt_dict_orig',learnt_dict)
            scipy.io.savemat(save_path+'learnt_dict_orig.mat',{'D': learnt_dict})
            # Normalize dictionary
            learnt_dict_norm = learnt_dict/np.linalg.norm(learnt_dict,axis=0)
            np.save(save_path+'learnt_dictionary_norm',learnt_dict_norm)
            scipy.io.savemat(save_path+'learnt_dict_norm.mat',{'D': learnt_dict_norm})

            learnt_dict = np.expand_dims(learnt_dict_norm.T.reshape(-1,8,8),axis=1)
            learnt_dict_norm = (learnt_dict-learnt_dict.min())/(learnt_dict.max()-learnt_dict.min())
            grid_img = torchvision.utils.make_grid(torch.Tensor(learnt_dict_norm), nrow=16)
            # Visualize dictionary
            plt.figure(figsize=(10,10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.savefig(save_path+'learnt_dictionary.png',transparent=True,bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    # Initialize the Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',default=1e-2,type=float)
    parser.add_argument('--epochs',default=100,type=int)
    parser.add_argument('--rounds',default=80,type=int)
    parser.add_argument('--init',default='odct',type=str)
    parser.add_argument('--img_name',default='barbara',type=str)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    trial_nums = [1,2,3,4,5]
    sigma_vals = [2,5,10,15,20,25,50,75,100]
    config = {}
    config['lr'] = args.lr
    config['epochs'] = args.epochs
    config['rounds'] = args.rounds
    config['trials'] = trial_nums
    config['sigmas'] = sigma_vals
    config['init'] = args.init
    config['dict_size'] = 256
    config['img_name'] = args.img_name
    config['device'] = device
    main(config)
