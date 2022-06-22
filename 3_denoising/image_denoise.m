image_name = 'barbara';
sigma_vals = [2,5,10,15,20,25,50,75,100];
total_trials = 5;
result_path = '..\3_denoising\results\';
algorithm = 'fastsolver'; % ksvd or fastsolver
psnr_vals = zeros(10,numel(sigma_vals),total_trials);

for trial = 1:total_trials
    for s_num = 1:numel(sigma_vals)
        curr_sigma = sigma_vals(s_num);
        curr_sample = strcat("Image:",image_name," sigma: ",int2str(curr_sigma)," trial: ",int2str(trial));
        disp(curr_sample);
        load_path = strcat(result_path,image_name,"\sigma_",int2str(curr_sigma),"_trial_",int2str(trial),"\",algorithm,"\");
        data_load_path = strcat("noisy_images\",image_name,"\sigma_",int2str(curr_sigma),"_trial_",int2str(trial),"\");
        im = load(strcat("noisy_images\",image_name,"\","orig_image.mat")).im;
        imnoise = load(strcat(data_load_path,"noisy_image.mat")).imnoise;

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
        save(strcat(params.load_path,"denoised_image"),"imout");

        % Find PSNR
        psnr_vals(s_num,trial) = 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:)));
    end
end
save(strcat("..\3_denoising\results\",image_name,"_",algorithm,"_psnr.mat"),"psnr_vals")