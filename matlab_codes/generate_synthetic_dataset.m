%  Based on the code by
%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  May 2009

clear;clc;
% dictionary dimensions
m = 20; % dimension of an atom
k = 50; % no. of atoms
% number of sampels
n = 200000;
% sparsity of each example
s = 3;
%  num of trials
total_trials = 50;
%% generate random dictionary and data %%
for n_trial = 1:total_trials
    D = normcols(randn(m,k));
    Z = zeros(k,n);
    for i = 1:n
      p = randperm(k);
      Z(p(1:s),i) = randn(s,1);
    end
    
    X = D*Z;
    path = fullfile('2_convergence','dataset',strcat('samples_',int2str(150000),'_trial_',int2str(n_trial)));
    status = mkdir(path); 

    save(fullfile(path,'gen_dictionary.mat'),'D')
    save(fullfile(path,'gen_code.mat'),'Z')
    save(fullfile(path,'gen_data.mat'),'X')
end