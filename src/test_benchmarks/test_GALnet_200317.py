import os
import numpy as np
import random
import cv2
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms

from generator import GALNet_v2
from utils import make_dir, load_network, prepare, write_log, denorm, tensor2img
from metrics import calculate_psnr, calculate_ssim
from prepare_dataset import Testset


TEST_BLUR_RANGE = ['k13', 'k15', 'k17', 'k19', 'k21', 'k23', 'k25', 'k27']


def main(model_dir, data_dir, save_dir, dataset_type='MSPL', dataset_name='CelebA', is_visual=True, use_gpu=True):
    
    # 1. Set Paths
    make_dir(save_dir)
    save_result_txt = '{}_{}_PSNR_SSIM'.format(dataset_type, dataset_name)
    if is_visual:
        save_img_dir = os.path.join(save_dir, 'img')
        make_dir(save_img_dir)
            
        
    # 2. Set Testset
    test_set= Testset(data_dir, dataset_type, dataset_name)
    teste_loader = DataLoader(test_set, batch_size=1, shuffle=False)    
    
    # 2. Model Load
    # Set GPU or CPU 
    device = torch.device('cuda') if use_gpu else device = torch.device('cpu')
    netG = generator.GALNet_v2(3, 3, 128, [4, 4, 4, 4]).to(device)
    netG = nn.DataParallel(netG)
    load_network(netG, model_dir)
    netG.eval()

    PSNR = dict(all=[])
    SSIM = dict(all=[])
    
    for b_range in TEST_BLUR_RANGE:
        PSNR['{}'.format(b_range)] = []
        SSIM['{}'.format(b_range)] = []
        
    for _, data in enumerate(valid_loader):
        gt_img, gt_filename, blur_img, blur_filename = data
        
        batch_size = gt_img.size(0)
        gt_img, blur_img = prepare([gt_img, blur_img], device)
        
        with torch.no_grad(): 
            output_list = netG(blur_img)
        
        gt_filename = gt_filename[0] 
        blur_filename = blur_filename[0]
        
        output_list= tensor2img_list(output_list)

        im_gt = denorm(gt_img)
        im_gt = tensor2img(im_gt)
        im_blur = output_list[-1]
        
        psnr_val = calculate_psnr(im_gt, im_blur)
        ssim_val = calculate_ssim(im_gt, im_blur)
        
        PSNR['all'].append(psnr_val)
        SSIM['all'].append(ssim_val)

        for b_range in TEST_BLUR_RANGE:
            if b_range in blur_filename:
                PSNR['{}'.format(b_range)].append(psnr_val)
                SSIM['{}'.format(b_range)].append(ssim_val)
        
        log = '[GT: {}] [BLUR: {}] [PSNR: {:.6f}] [SSIM: {:.6f}]'.format(
                    gt_filename, blur_filename, psnr_val, ssim_val)
        print(log)
        write_log(log, save_path, save_result_txt)
                                
        if is_visual:
            # save Images        
            for i in range(len(output_list)):
                save_name = os.path.join(save_img_dir, '{}_out{}.png'.format(blur_filename, i))
                cv2.imwrite(save_name, output_list[i])
    
    assert len(PSNR['all']) == len(SSIM['all']) 

    log = '='*60
    print(log)
    write_log(log, save_path, save_result_txt)
        
    for b_range in TEST_BLUR_RANGE:
        assert len(PSNR['{}'.format(b_range)]) == len(SSIM['{}'.format(b_range)])
        log = '[Blur Range: {}] Average -- [PSNR: {:.6f}] [SSIM: {:.6f}]'.format(
            b_range, 
            sum(PSNR['{}'.format(b_range)]) / len(PSNR['{}'.format(b_range)]),
            sum(SSIM['{}'.format(b_range)]) / len(SSIM['{}'.format(b_range)]),            
        )
        print(log)
        write_log(log, save_path, save_result_txt)

    log = '='*60
    print(log)
    write_log(log, save_path, save_result_txt)

    log = 'Total Average -- [PSNR: {:.6f}] [SSIM: {:.6f}]'.format(
            sum(PSNR['all']) / len(PSNR['all']),
            sum(SSIM['all']) / len(SSIM['all'])
    )
    print(log)
    write_log(log, save_path, save_result_txt)

   
if __name__ == '__main__':

    MODLE_DIR = '/home/ltb/Projects/FocusFace/results/COMPONENT/200313_ComponentNet_MSGFeatureStyle/checkpoints/netG_best.pth'
    DATA_DIR = './'
    SAVE_DIR = './'
    # DATASET_TYPE must be => 'MSPL' or 'CVPR18'
    DATASET_TYPE = 'MSPL'
    DATASET_NAME = 'CelebA'
    
    # if is_visual is true => save result images. else just save psnr result txtfile
    IS_VISUAL = True
    USE_GPU = True

    main(MODLE_DIR, DATA_DIR, save_dir, DATASET_TYPE, DATASET_NAME, IS_VISUAL, USE_GPU)

