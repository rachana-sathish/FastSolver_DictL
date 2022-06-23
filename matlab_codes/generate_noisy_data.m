%  Based on codes by
%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  August 2009
clear;clc;
sigma_vals = [2,5,10,15,20,25,50,75,100]; % noise levels
total_trials = 5;
image_path = fullfile('ksvdbox13','images');
dirname = fullfile(image_path,'*.png');
imglist = dir(dirname);
for image_num = 1:length(imglist)
    for trial_num = 1:total_trials
        for s_num = 1:numel(sigma_vals)
            sigma = sigma_vals(s_num);
            sprintf('Sigma: %.0f', sigma);
            imgname = imglist(image_num).name;
            save_path = fullfile('3_denoising','noisy_images',imgname(1:end-4),strcat('sigma_',int2str(sigma),'_trial_',int2str(trial_num)));           
            status = mkdir(save_path);
            im = imread(fullfile(image_path,imgname));
            im = double(im);
            save(fullfile('3_denoising','noisy_images',imgname(1:end-4),'orig_image'),'im'); 

            % params
            blocksize = 8; 
            trainnum = 40000; % no. of patches from image for training
            dictsize = 256; % no. of dictionary atoms

            % add noise
            n = randn(size(im)) * sigma;
            imnoise = im + n;
            save(fullfile(save_path,'noisy_image'),'imnoise'); 
            imwrite(imnoise/255,fullfile(save_path,'noisy_image.png'))
            p = ndims(imnoise);
            % blocksize %
            if (numel(blocksize)==1)
              blocksize = ones(1,p)*blocksize;
            end
            if (p==2 && any(size(imnoise)==1) && length(blocksize)==1)
              p = 1;
            end
            initdict = odctndict(blocksize,dictsize,p);
            initdict = initdict(:,1:dictsize);
            save(fullfile(save_path,'init_dict'),'initdict');

            % create training data
            ids = cell(p,1);
            if (p==1)
              ids{1} = reggrid(length(imnoise)-blocksize+1, trainnum, 'eqdist');
            else
            %   disp(size(x));
              [ids{:}] = reggrid(size(imnoise)-blocksize+1, trainnum, 'eqdist');
            end

            noisy_data = sampgrid(imnoise,blocksize,ids{:});
            % remove dc in blocks to conserve memory %
            blocksize = 2000;
            for i = 1:blocksize:size(noisy_data,2)
              blockids = i : min(i+blocksize-1,size(noisy_data,2));
              noisy_data(:,blockids) = remove_dc(noisy_data(:,blockids),'columns');
            end
            save(fullfile(save_path,'noisy_train_data'),'noisy_data');
        end
    end
end

