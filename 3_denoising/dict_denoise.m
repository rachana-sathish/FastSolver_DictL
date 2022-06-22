function [y,D,nz] = dict_denoise(params)
blocksize = params.blocksize;
sigma = params.sigma;
gain = 1.15;
params.gain = gain;
params.Edata = sqrt(prod(blocksize)) * sigma * gain;   % target error for omp
params.codemode = 'error';
params.noisemode = 'sigma';
load_path = params.load_path;
data_load_path = params.data_load_path;

%%%% loading training data %%%
params.data = load(strcat(data_load_path,"noisy_train_data.mat")).noisy_data;

% load normalized learnt dictionary
D = double(load(strcat(load_path,'learnt_dict_norm.mat')).D);

%%%%%  denoise the signal  %%%%%
maxval = params.maxval;

if (~isfield(params,'lambda'))
  params.lambda = maxval/(10*sigma);
end

params.dict = D;

disp('OMP denoising...');

[y,nz] = ompdenoise2(params);

end