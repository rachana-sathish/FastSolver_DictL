import numpy as np

def find_nearest(curr_idx,curr_errors,fixed_idxs):
    all_err = curr_errors[curr_idx,:]
    min_loc = int(np.argmin(all_err))
    # if candidate ref atom already selected
    # set error to high value to avoid getting selected again
    if min_loc in fixed_idxs:
        curr_errors[curr_idx, min_loc] = 1000
        # find next nearest atom with modified error matrix
        match_idx = find_nearest(curr_idx,curr_errors,fixed_idxs)
    else:
        match_idx = min_loc
    return match_idx # nearest reference atom for current index

def permute_scale_dictionary(learnt_dictionary,generation_dictionary):
    '''
    Permute and re-scale dictionary atoms
    '''
    dict_size = learnt_dictionary.shape[1]
    dict_elem = learnt_dictionary.shape[0]
    # Calculate error matrix and indices with minimum error for learnt atoms
    # Error matrix: row --> learnt dictionary | col --> reference dictionary
    error_mat = np.zeros((dict_size,dict_size))
    match_idx = np.zeros(dict_size,dtype=int)
    min_error = np.zeros(dict_size)
    for i in range(dict_size): # index of learnt dictionary
        for j in range(dict_size): # index of reference dictionary
            curr_atom = learnt_dictionary[:,i]
            ref_atom = generation_dictionary[:,j]
            error_mat[i,j] = np.mean((ref_atom-curr_atom)**2)
        match_idx[i] = np.argmin(error_mat[i,:]) # matching index in reference dictionary
        min_error[i] = np.min(error_mat[i,:])

    # Filter out repeated matches
    final_idx = np.ones(dict_size,dtype=int)*100 # array to store matches for reference atoms
    fixed_idxs = [] # List of indices matched
    for ref_idx in range(dict_size): # index in refernce dictionary
        loc_idx = np.where(match_idx==ref_idx)[0]
        if loc_idx.size == 1: # Single matching candidate
            final_idx[loc_idx] = ref_idx
            fixed_idxs.append(ref_idx)
            error_mat[:,ref_idx] = 1000
        if loc_idx.size > 1: # If more than one match, choose the one with min error
            error_vals = error_mat[loc_idx,ref_idx]
            min_error_idx = np.argmin(error_vals)
            if ref_idx not in fixed_idxs:
                final_idx[loc_idx[min_error_idx]] = ref_idx
                fixed_idxs.append(ref_idx)
                error_mat[:,ref_idx] = 1000

    remain_indices = list(np.where(final_idx==100)[0]) # unmatched atoms in learnt dictionary
    search_flag = False
    if len(remain_indices)>0:
        search_flag = True
        round_num = 1
    while search_flag:
        round_num += 1
        # List of selected candidates
        new_sel = [find_nearest(rem_idx,error_mat,fixed_idxs) for rem_idx in remain_indices]          
        # Unique candidates from all selected ones
        new_uniques = list(np.unique(np.array(new_sel)))
        for new_idx in new_uniques:
            if new_idx not in fixed_idxs:
                loc = np.where(np.array(new_sel)==new_idx)[0]
                # If matched to more than 1 atom find the atom with minimum error
                if loc.size > 1:
                    sel_idx = np.take(remain_indices,loc)
                    error_vals = np.take(error_mat[:,new_idx],sel_idx)
                    min_loc = np.argmin(error_vals)
                    fixed_idxs.append(new_idx)
                    final_idx[sel_idx[min_loc]] = new_idx
                    error_mat[:,new_idx] = 1000
                    loc_list = list(loc)
                    loc_list.remove(loc_list[min_loc])
                    for loc_idx in loc_list:  
                        final_idx[remain_indices[loc_idx]] = 100
                else: # If single match, add to main list
                    match_idx = new_idx
                    error_mat[:,match_idx] = 1000
                    fixed_idxs.append(match_idx)
                    final_idx[remain_indices[loc[0]]] = match_idx
            else:
                print('Index',new_idx,'already matched!')
        remain_indices = list(np.where(final_idx==100)[0]) # unmatched atoms - learnt dictionary
        if not remain_indices:
            search_flag = False
    # permute learnt dictionary
    permuted_idx = np.array(final_idx)
    permuted_learnt_dict = np.zeros(learnt_dictionary.shape)
    for i in range(dict_size):
        permuted_learnt_dict[:,permuted_idx[i]] = learnt_dictionary[:,i]
    # rescale learnt dictionary
    scale_diff = np.mean(generation_dictionary/permuted_learnt_dict,axis=0)
    scale_diff_matrix = np.repeat(np.expand_dims(scale_diff,axis=0),dict_elem,axis=0)
    scaled_permut_learnt_dict = permuted_learnt_dict*scale_diff_matrix
    return scaled_permut_learnt_dict
    