close all;
clear all;
clc;
%Download Test Data provided by authors Ziyi et al. from,
%https://sites.google.com/site/ziyishenmi/cvpr18_face_deblur
%listinfo = dir('./Test_data_Helen/final_Helen_gt');


gt_path = 'gt_img_path';
deblur_path = 'deblur_img_path';

gtlist = dir(gt_path);
blurlist = dir(deblur_path);

gt_num = length(gtlist);
blur_num = length(blurlist);

file = fopen('PSNR_SSIM.txt','w');
avg_ssim = 0.0;
avg_psnr = 0.0;
n = 0;
psnrcell = {};
ssimcell = {};
nblr = 1;
for i = 3:gt_num
    gtname = strcat(gt_path,'/',gtlist(i).name);
    gtimg = imread(gtname);
    tmp1 = split(gtlist(i).name,'.');
    gt_name = tmp1{1};
    for j = 1:10
        num = 1;
        for k=13:2:27
            // % cvpr2018
            // blrname = strcat(gt_name, '_ker',num2str(j,'%02d'),'_blur_k',num2str(k),'_random.png');            
            % ours
            % blrname = strcat(gt_name, '_ker',num2str(j,'%02d'),'_blur_k',num2str(k),'_out3.png');
            tmp2 = split(blrname,'.');
            blur_name = tmp2{1};
            blrpath = strcat(deblur_path,'/',blrname);
            blimg = imread(blrpath);
            disp(blrname); disp(gtname), 
            ssimval = ssim(gtimg,blimg);
            psnrval = psnr(gtimg,blimg);
            fprintf(file, ' [GT: ');
            fprintf(file, gt_name);
            fprintf(file, ' ] ');
            fprintf(file, ' [BLUR: ');
            fprintf(file, blur_name);
            fprintf(file, ' ] ');
            fprintf(file, ' [PSNR: ');
            psnrvalstr = num2str(psnrval); 
            fprintf(file, psnrvalstr);
            fprintf(file, ' ] ');
            fprintf(file, ' [SSIM: ');
            ssimvalstr = num2str(ssimval); 
            fprintf(file, ssimvalstr);
            fprintf(file, ' ] ');
            fprintf(file, '\n');
            psnrcell{num, nblr} = psnrval;
            ssimcell{num, nblr} = ssimval;
            disp(psnrval); disp(ssimval);
            avg_ssim = avg_ssim + ssimval;
            avg_psnr = avg_psnr + psnrval;
            disp(avg_psnr); disp(ssimval);
            % ssimlist(n+1) = ssimval;
            num = num + 1;
            n = n+1;
        end
        nblr = nblr + 1;
    end 
end
blrange = 1;
for k=13:2:27
    psnrval_blrange = sum([psnrcell{blrange,:}])/length([psnrcell{blrange,:}]);
    ssimval_blrange = sum([ssimcell{blrange,:}])/length([ssimcell{blrange,:}]);
    disp(psnrval_blrange); disp(ssimval_blrange);
    fprintf(file,' [BLUR RANGE: k');
    kstr = num2str(k);
    fprintf(file, kstr);
    fprintf(file, ' ] ');
    psnrvalstr_blrange = num2str(psnrval_blrange);
    fprintf(file, 'Aveage -- [PSNR: ');
    fprintf(file, psnrvalstr_blrange);
    fprintf(file, ' ] ');
    ssimvalstr_blrange = num2str(ssimval_blrange);
    fprintf(file, ' [SSIM: ');
    fprintf(file, ssimvalstr_blrange);
    fprintf(file, ' ] ');
    fprintf(file, '\n');
    blrange = blrange + 1;
end
disp(avg_psnr/(n)); disp(avg_ssim/(n));
psnr = avg_psnr/(n);
ssim = avg_ssim/(n);
fprintf(file, 'Total Average -- [PSNR: ');
psnrstr = num2str(psnr);
fprintf(file, psnrstr);
fprintf(file, ' ] ');
fprintf(file, '[SSIM: ');
ssimstr = num2str(ssim);
fprintf(file, ssimstr);
fprintf(file, ' ] ');
fclose(file);