%  Based on codes by
%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  August 2009
clear;clc;
sigma_vals = [2,10,15,20,25,50,75,100]; % noise levels
total_trials = 5;
blocksize = 8; 
dictsize = 256; % no. of dictionary atoms
image_path = "ksvd_toolbox\ksvdbox13\images\";
dirname = fullfile(image_path,'*.png');
imglist = dir(dirname);
for image_num = 1:length(imglist)
    for trial_num = 1:total_trials
        for s_num = 1:numel(sigma_vals)
            sigma = sigma_vals(s_num);
            sprintf('Sigma: %.0f', sigma);
            imgname = imglist(image_num).name;
            save_path = strcat("noisy_images\",imgname(1:end-4),"\sigma_",int2str(sigma),"_trial_",int2str(trial_num),"\");            
            status = mkdir(save_path);
            im = imread(strcat(image_path,imgname));
            im = double(im);
            % add noise
            n = randn(size(im)) * sigma;
            imnoise = im + n;
            save(strcat(save_path,"noisy_image"),"imnoise"); 
            p = ndims(imnoise);
            if (p==2 && any(size(imnoise)==1) && length(blocksize)==1)
              p = 1;
            end
            initdict = odctndict(blocksize,dictsize,p);
            initdict = initdict(:,1:dictsize);
            save(strcat(save_path,"init_dict"),"initdict");
        end
    end
end

