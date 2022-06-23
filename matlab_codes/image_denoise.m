%  Based on the code by
%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  May 2009
clear;clc;
image_name = 'barbara'; % change image name here
sigma_vals = [2,5,10,15,20,25,50,75,100];
total_trials = 5;
result_path = fullfile('3_denoising','results');
algorithm = 'ksvd'; % ksvd or fastsolver
psnr_vals = zeros(numel(sigma_vals),total_trials);

for trial = 1:total_trials
    for s_num = 1:numel(sigma_vals)
        curr_sigma = sigma_vals(s_num);
        curr_sample = strcat('Image:',image_name,' sigma: ',int2str(curr_sigma),' trial: ',int2str(trial));
        disp(curr_sample);
        load_path = fullfile(result_path,image_name,strcat('sigma_',int2str(curr_sigma),'_trial_',int2str(trial)),algorithm);
        data_load_path = fullfile('3_denoising','noisy_images',image_name,strcat('sigma_',int2str(curr_sigma),'_trial_',int2str(trial)));
        im = load(fullfile('3_denoising','noisy_images',image_name,'orig_image.mat')).im;
        imnoise = load(fullfile(data_load_path,'noisy_image.mat')).imnoise;

        % set parameters            
        params.x = imnoise;
        params.blocksize = 8;
        params.dictsize = 256;
        params.sigma = curr_sigma;
        params.maxval = 255;
        params.trainnum = 40000;
        params.iternum = 20;
        params.memusage = 'high';
        params.load_path = load_path;
        params.data_load_path = data_load_path;

        % denoise image
        disp('Performing denoising...');
        [imout, dict] = dict_denoise(params);
        save(fullfile(params.load_path,'denoised_image'),'imout');
        imwrite(imout/params.maxval,fullfile(params.load_path,'denoised_image.png'))

        % Find PSNR
        psnr_vals(s_num,trial) = 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:)));
        noisy_psnr = sprintf('Noisy image, PSNR = %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imnoise(:))));
        disp(noisy_psnr);
        denoised_psnr = sprintf('Denoised image, PSNR: %.2fdB', psnr_vals(s_num,trial));
        disp(denoised_psnr);
    end
end
save(fullfile('3_denoising','results',strcat(image_name,'_',algorithm,'_psnr.mat')),'psnr_vals')