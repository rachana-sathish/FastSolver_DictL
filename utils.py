import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

def normalize_dictionary(dictionary):
    if not isinstance(dictionary, torch.Tensor): 
        dictionary = torch.Tensor(dictionary)
    dictionary_temp = dictionary @ torch.diag(torch.sign(dictionary[0,:]))
    return dictionary_temp/torch.linalg.norm(dictionary_temp,dim=0)

def colnorms_squared(data,device):
    '''
    Normalise data across columns
    '''
    norm_data = torch.zeros(data.shape[1]).to(device)
    block_size = 2000
    # compute in blocks to conserve memory
    for i in range(0,data.shape[1],block_size):
        blockids = torch.arange(i,min(i+block_size,data.shape[1])).to(device)
        norm_data[blockids] = torch.sum(data[:,blockids]**2,axis=0)
    return norm_data

def  optimize_atom(data,dictionary,dict_idx,codes,unused_data,replaced_atoms,device):
    '''
    Update dictionary atoms
    '''
    data_indices = torch.nonzero(codes[dict_idx,:],as_tuple=True)[0]
    curr_code = codes[dict_idx,data_indices]
    if len(data_indices) < 1:
        max_signals = 5000
        perm = torch.randperm(len(unused_data),device=device)
        perm = list(perm[:min(max_signals,len(perm))])
        error = sum((data[:,unused_data[perm]] - dictionary@codes[:,unused_data[perm]])**2)
        max_err_idx = torch.argmax(error)
        atom = data[:,unused_data[perm[max_err_idx]]]
        atom = atom/torch.linalg.norm(atom)
        curr_code = torch.zeros(curr_code.shape).to(device)
        idx_list = list(torch.arange(0,perm[max_err_idx])) + list(torch.arange(perm[max_err_idx+1],len(unused_data)))
        unused_data = unused_data[idx_list]
        replaced_atoms[dict_idx] = 1
    else:
        small_codes = codes[:,data_indices]
        curr_atom = dictionary[:,dict_idx]
        residual = data[:,data_indices] - dictionary@small_codes + curr_atom.unsqueeze(1)@curr_code.unsqueeze(0)
        u_mat, s_mat, vt_mat = torch.linalg.svd(residual,full_matrices=False)
        atom = u_mat[:,0]
        curr_code = vt_mat.T[:,0]
        curr_code = s_mat[0]*curr_code
    return (atom,curr_code,data_indices,unused_data,replaced_atoms)

def clear_dict(dictionary,codes,data,mu_thresh,unused_data,replaced_atoms):
    '''
    Clear least used dictionary atoms
    '''
    use_thresh = 4  # at least this number of samples must use the atom to be kept
    dict_size = dictionary.shape[1]
    err = sum((data - dictionary@codes)**2)
    usecount = torch.sum(abs(codes)>1e-7, axis=1)
    for dict_idx in range(dict_size):
        # compute similarity between dictionary atoms
        dict_similarity = dictionary.T@dictionary[:,dict_idx]
        dict_similarity[dict_idx] = 0
        # replace atom
        if  (max(dict_similarity**2)>mu_thresh**2 or usecount[dict_idx]<use_thresh) and (replaced_atoms[dict_idx]==0):
            max_err_idx = torch.argmax(err[unused_data])
            dictionary[:,dict_idx] = data[:,unused_data[max_err_idx]] / torch.linalg.norm(data[:,unused_data[max_err_idx]])
            idx = list(torch.arange(max_err_idx)) + list(torch.arange(max_err_idx+1,unused_data.shape[0]))
            unused_data = unused_data[idx]
    return dictionary, unused_data

def replace_atoms(dictionary,codes,data,replaced_atoms,unused_data):
    '''
    Replace unused dictionary atoms
    '''
    dict_size = dictionary.shape[1]
    for dict_idx in range(dict_size):
        data_indices = list(np.nonzero(codes.T[dict_idx,:])[0])
        curr_learnt_code = codes.T[dict_idx,data_indices]
        if len(data_indices) < 1:
            maxsignals = 5000
            perm = np.random.permutation(len(unused_data))
            perm = list(perm[:min(maxsignals,len(perm))])
            error = sum((data[:,unused_data[perm]] -
                        dictionary@codes.T[:,unused_data[perm]])**2)
            max_err_idx = np.argmax(error)
            atom = data[:,unused_data[perm[max_err_idx]]]
            atom = atom/np.linalg.norm(atom)
            curr_learnt_code = np.zeros(curr_learnt_code.shape)
            idx_list = list(np.arange(0,perm[max_err_idx])) + \
                        list(np.arange(perm[max_err_idx+1],len(perm)))
            unused_data = unused_data[idx_list]
            replaced_atoms[dict_idx] = 1
            dictionary[:,dict_idx] = atom
            codes.T[dict_idx,data_indices] = curr_learnt_code
    return dictionary, codes, replaced_atoms, unused_data

def count_matching_atoms(gen_dictionary, learnt_dictionary):
    identical_atoms = 0
    # Find detection rate of dictionary atoms
    for i in range(gen_dictionary.shape[1]):
        atom = gen_dictionary[:,i]
        distances = 1-abs(atom@learnt_dictionary)
        min_dist = min(distances)
        identical_atoms += (min_dist < 0.01)
    return identical_atoms
    