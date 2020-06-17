import os
import shutil
import numpy as np
import random
import cv2
from PIL import Image
import time
from collections import OrderedDict
from scipy import signal
import scipy.io
import logging
from math import log10

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from configs import train_configs
from data import celeba_hq
from model import generator, discriminator, initializers
from loss import ganloss, vgg16faceloss 
from utils.misc import AverageMeter, make_dir, set_random_seed, setup_logger
from utils.visualize import denorm, tensor2img, vis_parsing_maps


def main():
    cfg = train_configs.get_configs()

    #======================================================
    # 1. Make dirs
    #======================================================
    save_dir, model_dir, valid_dir, log_dir = set_dirs(cfg)

    #======================================================
    # 2. Set loggers & tensorboard
    #======================================================
    setup_logger('configs', save_dir, 'configs', level=logging.INFO, screen=True) 
    setup_logger('valid', save_dir, 'valid', level=logging.INFO, is_formatter=True, screen=False) 
    config_logger = logging.getLogger('configs')  # training logger
    valid_logger = logging.getLogger('valid')  # validation logger
    # Tensorboard
    board_writer = SummaryWriter(log_dir)

    # Save Configurations
    for k, v in cfg.items():
        log = '{} : {}'.format(k, v)
        config_logger.info(log)
          
    #======================================================
    # 3. Set GPU
    #======================================================
    is_gpu = False
    gpu_ids = cfg.GPU_IDS
    device = None    
    if gpu_ids is not None and torch.cuda.is_available():       
        is_gpu = True
        torch.cuda.set_device(gpu_ids[0])
        torch.backends.cudnn.benckmark = True           
        device = torch.device('cuda')
    else: 
        device = torch.device('cpu')
    cfg.is_gpu = is_gpu
    config_logger.info('Training on: {:s} \t GPUs: {}'.format(str(device), gpu_ids))

    #======================================================
    # 4. Random Seed, Define Models
    #======================================================
    set_random_seed(is_gpu)    
    
    # generators    
    GENERATORS = dict(netG_0 = generator.simpleUshapeNet(3, 3, 64, 8).to(device),
                      netG_1 = generator.simpleUshapeNet(3, 3, 64, 8).to(device),
                      netG_2 = generator.simpleUshapeNet(3, 3, 64, 8).to(device),
                      netG_3 = generator.simpleUshapeNet(3, 3, 64, 8).to(device)
                      )
    for _, netG in GENERATORS.items():
        initializers.init_weights(netG, init_type=cfg.INIT_TYPE, scale=0.1)
        if is_gpu:
            netG = nn.DataParallel(netG, device_ids=gpu_ids)
            
    # discriminators    
    if cfg.LOSS_ADV_WEIGHT:
        DISCRIMINATORS = dict(netD_0 = None,
                              netD_1 = discriminator.SimpleDiscriminator(3, True, True).to(device),
                              netD_2 = discriminator.SimpleDiscriminator(3, True, True).to(device),
                              netD_3 = discriminator.SimpleDiscriminator(3, True, True).to(device)
                              )      
        for _, netD in DISCRIMINATORS.items():
            if netD is not None:
                initializers.init_weights(netD, init_type=cfg.INIT_TYPE, scale=1)
                if is_gpu:
                    netD = nn.DataParallel(netD, device_ids=gpu_ids)
    else: DISCRIMINATORS = None
    
    #======================================================
    # 5. Loss Fucntion, Optimizers, Schedulers
    #======================================================        
    OPTIMIZERS = dict()
    for name, model in GENERATORS.items():
        if model is not None:
            opt = torch.optim.Adam(model.parameters(), lr=cfg.INIT_LR_G, weight_decay=cfg.WEIGHT_DECAY_D, betas=(0.9, 0.999))
            OPTIMIZERS['{}'.format(name)] = opt
    
    if DISCRIMINATORS is not None: 
        for name, model in DISCRIMINATORS.items():
            if model is not None:
                opt = torch.optim.Adam(model.parameters(), lr=cfg.INIT_LR_G, weight_decay=cfg.WEIGHT_DECAY_D, betas=(0.9, 0.999))
                OPTIMIZERS['{}'.format(name)] = opt
    
        
    SCHEDULERS = []  
    for _, opt in OPTIMIZERS.items():
        SCHEDULERS.append(lr_scheduler.ExponentialLR(opt, gamma=0.99))
        # SCHEDULERS.append(lr_scheduler.MultiStepLR(opt, cfg.LR_STEPS, cfg.LR_GAMMA))
                
    if cfg.LOSS_L1_WEIGHT:
        cri_l1 = nn.L1Loss().to(device)         
    else: cri_l1 = None

    if cfg.LOSS_ADV_WEIGHT:   
        cri_gan = ganloss.GAN_Loss(gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0).to(device)    
    else: cri_gan=None

    # vggface Loss Network
    if cfg.LOSS_VGG_WEIGHT:
        vggface = vgg16faceloss.VGG16FeatureExtractor(model_path=cfg.LOAD_PATH['VGGFace16']).to(device)
        if is_gpu:
            vggface = nn.DataParallel(vggface, device_ids=gpu_ids)
            vggface.eval()
    else: vggface=None

    LOSS_MODEL = dict(vggface=vggface)
    CRITERIONS = dict(cri_l1=cri_l1, cri_gan=cri_gan)    
    cri_mse = nn.MSELoss().to(device)

    #======================================================
    # 6. Data loader    
    #======================================================
    train_set =  celeba_hq.CelebA_HQ(**cfg.TRAINSET)
    train_loader = DataLoader(
        train_set, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, drop_last=True, pin_memory=True)
    
    valid_set = celeba_hq.validface(**cfg.VALIDSET)
    valid_loader = DataLoader(
        valid_set, batch_size=cfg.VALID_BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, drop_last=True, pin_memory=True)    
    
    # Blur kernel files
    train_kernel_dict = get_blurkernels(cfg.TRAIN_KERNEL_PATH)
    # valid_kernel_dict = get_blurkernels(cfg.TEST_KERNEL_PATH)

    #======================================================
    # 7. Resume Training
    #======================================================
    start_epoch = 0
    total_epoch = cfg.TOTAL_EPOCH   

    if cfg.LOAD_PATH['RESUME_STATE']:
        start_epoch = load_resume(cfg.LOAD_PATH['RESUME_STATE'], OPTIMIZERS, SCHEDULERS)
        start_epoch = start_epoch + 1

    best_psnr = 0    
    for epoch in range(start_epoch, total_epoch):

        train_loss = train(cfg, train_loader, GENERATORS, DISCRIMINATORS, LOSS_MODEL, OPTIMIZERS, SCHEDULERS, CRITERIONS, epoch, train_kernel_dict, device)
                
        for loss_type, loss_val in train_loss.items():
            board_writer.add_scalar('{}'.format(loss_type), loss_val.avg, epoch+1)       
        
        is_visual = False
        if cfg.VISUAL_N_EPOCH is not None and epoch % cfg.VISUAL_N_EPOCH == 0:
            is_visual = True        
        psnr_result = validate(cfg, valid_loader, GENERATORS, cri_mse, device, epoch, valid_dir, is_visual)
        avg_psnr = psnr_result['netG_3'].avg
        
        log = '[Epoch:{}|{}] [PSNR  Average] \
            [netG_0: {:.4f}] [netG_1: {:.4f}] [netG_2: {:.4f}] [netG_3: {:.4f}]'.format(
            epoch+1, total_epoch,
            psnr_result['netG_0'].avg,
            psnr_result['netG_1'].avg,
            psnr_result['netG_2'].avg,
            psnr_result['netG_3'].avg
        )        
        print(log)
        valid_logger.info(log)
        for psnr_type, psnr_val in psnr_result.items():            
            board_writer.add_scalar('PSNR_{}'.format(psnr_type), psnr_val.avg, epoch+1)
       
        is_best = avg_psnr > best_psnr
        best_psnr = max(avg_psnr, best_psnr)        
        save_network(GENERATORS, DISCRIMINATORS, model_dir, is_best)
        save_state(SCHEDULERS, OPTIMIZERS, model_dir, epoch)
        
        adjust_learning_rate(SCHEDULERS) 


def train(cfg, 
          train_loader, 
          GENERATORS, 
          DISCRIMINATORS, 
          LOSS_MODEL, 
          OPTIMIZERS, 
          SCHEDULERS, 
          CRITERIONS, 
          epoch, 
          train_kernel_dict, 
          device):

    batch_time = AverageMeter() 

    loss_dict = OrderedDict() 
    # loss_dict['netD_total_loss'] = AverageMeter()
    if cfg.LOSS_ADV_WEIGHT:
        for name, models in DISCRIMINATORS.items():
            if models is not None:
                loss_dict['dis_loss_{}'.format(name)] = AverageMeter()
                loss_dict['adv_loss_{}'.format(name)] = AverageMeter()                
                models.train() # switch train mode
    # generators 
    # loss_dict['netG_total_loss'] = AverageMeter()
    for name, models in GENERATORS.items():
        if models is not None:
            models.train()
            loss_dict['gen_loss_{}'.format(name)] = AverageMeter()
            if cfg.LOSS_L1_WEIGHT:
                loss_dict['l1_loss_{}'.format(name)] = AverageMeter()
                           
    if cfg.LOSS_VGG_WEIGHT:     
        loss_dict['vgg_loss'] = AverageMeter()
          
    cri_l1 = CRITERIONS['cri_l1']
    cri_gan = CRITERIONS['cri_gan']
              
    end = time.time()
    for cur_iter, data in enumerate(train_loader):
        
        gt_img, gt_onehot_segmap, _ = data
                       
        batch_size = gt_img.size(0)

        blur_range, kernels = random.choice(list(train_kernel_dict.items()))        
        blur_img = get_blurtensor(gt_img, blur_range, kernels)

        blur_img, gt_img, gt_onehot_segmap = prepare([blur_img, gt_img, gt_onehot_segmap], device)
              
        
        # out_1 = GENERATORS['netG_1'](out_0.detach())
        # out_2 = GENERATORS['netG_2'](out_1.detach())
        # out_3 = GENERATORS['netG_3'](out_2.detach())
        #======================================
        # 1. Training Model netG_0
        #======================================          
        out_0 = GENERATORS['netG_0'](blur_img)
        g_loss = 0
        l1_loss = cri_l1(gt_img, out_0)
        loss_dict['l1_loss_netG_0'].update(l1_loss.item(), batch_size)
        g_loss += l1_loss
        
        loss_dict['gen_loss_netG_0'].update(g_loss.item(), batch_size)
        OPTIMIZERS['netG_0'].zero_grad()
        g_loss.backward(retain_graph=True)
        OPTIMIZERS['netG_0'].step()

        #======================================
        # 2. Training Model netD_1, netG_1
        #======================================
        out_1 = GENERATORS['netG_1'](out_0)
        gt_part_1 = gt_img * gt_onehot_segmap[:,3:4,:,:]
        out_part_1 = out_1 * gt_onehot_segmap[:,3:4,:,:]
        # 1.1. netD_1
        if cfg.LOSS_ADV_WEIGHT:
            d_loss = 0                                
            dis_input_real = gt_part_1
            dis_input_fake = out_part_1.detach()

            dis_real = DISCRIMINATORS['netD_1'](dis_input_real) 
            dis_fake = DISCRIMINATORS['netD_1'](dis_input_fake)
            dis_real_loss = cri_gan(dis_real, True)        
            dis_fake_loss = cri_gan(dis_fake, False)         
            d_loss += (dis_real_loss + dis_fake_loss) / 2
            loss_dict['dis_loss_netD_1'].update(d_loss.item(), batch_size)

            OPTIMIZERS['netD_1'].zero_grad()
            d_loss.backward(retain_graph=True)
            OPTIMIZERS['netD_1'].step()

        # 1.2. netG_1
        g_loss = 0 
        if cfg.LOSS_ADV_WEIGHT:      
            gen_input_fake = out_part_1.detach()      
            gen_fake = DISCRIMINATORS['netD_1'](gen_input_fake)
            adv_loss = cri_gan(gen_fake, True) * cfg.LOSS_ADV_WEIGHT        
            loss_dict['adv_loss_netD_1'].update(adv_loss.item(), batch_size)
            g_loss += adv_loss
                        
        if cfg.LOSS_L1_WEIGHT:             
            l1_loss = cri_l1(gt_part_1, out_part_1) * cfg.LOSS_L1_WEIGHT
            loss_dict['l1_loss_netG_1'].update(l1_loss.item(), batch_size)
            g_loss += l1_loss
        
        loss_dict['gen_loss_netG_1'].update(g_loss.item(), batch_size)
        OPTIMIZERS['netG_1'].zero_grad()
        g_loss.backward(retain_graph=True)
        OPTIMIZERS['netG_1'].step()    

        #======================================
        # 3. Training Model netD_2, netG_2
        #======================================
        out_2 = GENERATORS['netG_2'](out_1)
        gt_part = gt_img * gt_onehot_segmap[:,1:2,:,:]
        out_part = out_2 * gt_onehot_segmap[:,1:2,:,:]
        # 1.1. netD_2
        if cfg.LOSS_ADV_WEIGHT:
            d_loss = 0                                
            dis_input_real = gt_part
            dis_input_fake = out_part.detach()

            dis_real = DISCRIMINATORS['netD_2'](dis_input_real) 
            dis_fake = DISCRIMINATORS['netD_2'](dis_input_fake)
            dis_real_loss = cri_gan(dis_real, True)        
            dis_fake_loss = cri_gan(dis_fake, False)         
            d_loss += (dis_real_loss + dis_fake_loss) / 2
            loss_dict['dis_loss_netD_2'].update(d_loss.item(), batch_size)

            OPTIMIZERS['netD_2'].zero_grad()
            d_loss.backward(retain_graph=True)
            OPTIMIZERS['netD_2'].step()

        # 1.2. netG_2
        g_loss = 0 
        if cfg.LOSS_ADV_WEIGHT:      
            gen_input_fake = out_part.detach()      
            gen_fake = DISCRIMINATORS['netD_2'](gen_input_fake)
            adv_loss = cri_gan(gen_fake, True) * cfg.LOSS_ADV_WEIGHT        
            loss_dict['adv_loss_netD_2'].update(adv_loss.item(), batch_size)
            g_loss += adv_loss
                        
        if cfg.LOSS_L1_WEIGHT:             
            l1_loss = cri_l1(gt_part, out_part) * cfg.LOSS_L1_WEIGHT
            loss_dict['l1_loss_netG_2'].update(l1_loss.item(), batch_size)
            g_loss += l1_loss
        
        loss_dict['gen_loss_netG_2'].update(g_loss.item(), batch_size)
        OPTIMIZERS['netG_2'].zero_grad()
        g_loss.backward(retain_graph=True)
        OPTIMIZERS['netG_2'].step() 

        #======================================
        # 4. Training Model netD_3, netG_3
        #======================================
        out_3 = GENERATORS['netG_3'](out_2)
        gt_part = gt_img * gt_onehot_segmap[:,2:3,:,:]
        out_part = out_3 * gt_onehot_segmap[:,2:3,:,:]
        # 1.1. netD_3
        if cfg.LOSS_ADV_WEIGHT:
            d_loss = 0                                
            dis_input_real = gt_part
            dis_input_fake = out_part.detach()

            dis_real = DISCRIMINATORS['netD_3'](dis_input_real) 
            dis_fake = DISCRIMINATORS['netD_3'](dis_input_fake)
            dis_real_loss = cri_gan(dis_real, True)        
            dis_fake_loss = cri_gan(dis_fake, False)         
            d_loss += (dis_real_loss + dis_fake_loss) / 2
            loss_dict['dis_loss_netD_3'].update(d_loss.item(), batch_size)

            OPTIMIZERS['netD_3'].zero_grad()
            d_loss.backward(retain_graph=True)
            OPTIMIZERS['netD_3'].step()

        # 1.2. netG_3
        g_loss = 0 
        if cfg.LOSS_ADV_WEIGHT:      
            gen_input_fake = out_part.detach()      
            gen_fake = DISCRIMINATORS['netD_3'](gen_input_fake)
            adv_loss = cri_gan(gen_fake, True) * cfg.LOSS_ADV_WEIGHT        
            loss_dict['adv_loss_netD_3'].update(adv_loss.item(), batch_size)
            g_loss += adv_loss
                        
        if cfg.LOSS_L1_WEIGHT:             
            l1_loss = cri_l1(gt_part, out_part) * cfg.LOSS_L1_WEIGHT
            loss_dict['l1_loss_netG_3'].update(l1_loss.item(), batch_size)
            g_loss += l1_loss
               
        if cfg.LOSS_VGG_WEIGHT:
            # vgg face feature loss
            gen_input_real = gt_img
            gen_input_fake = out_3.detach()             
            fake_feat = LOSS_MODEL['vggface'](gen_input_fake)
            real_feat = LOSS_MODEL['vggface'](gen_input_real)
            vgg_loss = cri_l1(fake_feat, real_feat) * cfg.LOSS_VGG_WEIGHT
            loss_dict['vgg_loss'].update(vgg_loss.item(), batch_size)
            g_loss += vgg_loss
        
        loss_dict['gen_loss_netG_3'].update(g_loss.item(), batch_size)
        OPTIMIZERS['netG_3'].zero_grad()
        g_loss.backward(retain_graph=True)
        OPTIMIZERS['netG_3'].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        print("="*30)        
        log  = '[Epoch: {}|{}] [Iter: {}|{}({:.3f}s)]\
            \n[GPU:{}] [KEY: {}]\
            \n[Blur:{}]'.format(
                epoch, cfg.TOTAL_EPOCH,
                cur_iter+1, len(train_loader),
                batch_time.avg,
                cfg.GPU_IDS, cfg.KEYPOINT,
                blur_range,                             
            )
        print(log)
        for loss_name, value in loss_dict.items():
            log = '\t[{} : {:.6f}]'.format(loss_name, value.avg)
            print(log)
   
    return loss_dict


def validate(cfg, 
             valid_loader, 
             GENERATORS, 
             cri_mse, 
             device, 
             epoch, 
             save_path, 
             is_visual):
    print("="*30)
    print("START VALIDATON")   
    psnr_result = OrderedDict()
    # switch to evaluate mode    
    for name, models in GENERATORS.items():
        if models is not None:
            models.eval()
            psnr_result['{}'.format(name)] = AverageMeter()    
               
    for _, data in enumerate(valid_loader):
        gt_img, gt_filename, blur_img, blur_filename = data
        
        batch_size = gt_img.size(0)
        gt_img, blur_img = prepare([gt_img, blur_img], device)
                
        with torch.no_grad(): 
            out_0 = GENERATORS['netG_0'](blur_img)
            out_1 = GENERATORS['netG_1'](out_0)
            out_2 = GENERATORS['netG_2'](out_1)
            out_3 = GENERATORS['netG_3'](out_2)
        
        output = dict(netG_0 = out_0, netG_1 = out_1, netG_2 = out_2, netG_3 = out_3)
        
        for name, outs in output.items():
            mse = cri_mse(outs, gt_img)
            psnr = 10 * log10(1 / mse.item())
            psnr_result['{}'.format(name)].update(psnr, batch_size) 

        if is_visual:
            gt_filename = gt_filename[0] 
            blur_filename = blur_filename[0]

            save_out_path = os.path.join(save_path, 'output')
            make_dir(save_out_path)
            save_ep_path = os.path.join(save_out_path, 'ep_{}'.format(epoch))
            make_dir(save_ep_path)
                                    
            output_list = []
            for _, outs in output.items():                 
                output_list.append(outs[0,:,:,:])

            output_list= tensor2img_list(output_list)
            for i in range(len(output_list)):
                save_name = os.path.join(save_ep_path, '{}_out{}.png'.format(blur_filename, i))
                cv2.imwrite(save_name, output_list[i])
    
    return psnr_result


def set_dirs(cfg):
    # 1. Make Dirs
    save_dir = cfg.SAVE_DIR
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    make_dir(save_dir)

    model_dir = os.path.join(save_dir, 'checkpoints')
    make_dir(model_dir)

    valid_dir = os.path.join(save_dir, 'valid')
    make_dir(valid_dir)

    log_dir = os.path.join(save_dir, 'logs')
    make_dir(log_dir)

    return save_dir, model_dir, valid_dir, log_dir


def get_multiscaleimgs(img):

    orgin_size = img.size()
    w, h = orgin_size[2:]
    
    new_w, new_h = int(w//4), int(h//4)
    img1 = F.interpolate(img, size=(new_w, new_h), mode='bilinear', align_corners=True)
    
    new_w, new_h = int(w//2), int(h//2)
    img2 = F.interpolate(img, size=(new_w, new_h), mode='bilinear', align_corners=True)
    
    ouputs = [img1, img2, img]

    return ouputs


def get_testblurtensor(image_tensor, b_range, kernels, index):
    batch_size = image_tensor.size(0)
    image_clone = image_tensor.clone() 
    image_np = image_clone.numpy()

    blur_img_np = np.copy(image_np)
    copy_img_np = np.copy(image_np)

    PADDING = dict(k13=6, k15=7, k17=8, k19=9, k21=10, k23=11, k25=12, k27=13)
    pad_size = PADDING['{}'.format(b_range)]
    n_pad = ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size))
    copy_img_np = np.pad(copy_img_np, n_pad, 'reflect')

    if kernels is not None:
        for j in range (batch_size):
            blur_img_np[j,0,:,:]= signal.convolve(copy_img_np[j,0,:,:],kernels[index,:,:],mode='valid')
            blur_img_np[j,1,:,:]= signal.convolve(copy_img_np[j,1,:,:],kernels[index,:,:],mode='valid')
            blur_img_np[j,2,:,:]= signal.convolve(copy_img_np[j,2,:,:],kernels[index,:,:],mode='valid')
    
    # blur_img_np = blur_img_np + (1.0/255.0)* np.random.normal(0,4,blur_img_np.shape) 
    
    blur_tensor = torch.from_numpy(blur_img_np)  
    blur_tensor = blur_tensor.float()

    assert image_tensor.size() == blur_tensor.size()
    return blur_tensor 


def get_blurtensor(image_tensor, b_range, kernels):
    batch_size = image_tensor.size(0)
    image_clone = image_tensor.clone() 
    image_np = image_clone.numpy()

    blur_img_np = np.copy(image_np)
    
    if b_range != 'k0':
        copy_img_np = np.copy(image_np)
        PADDING = dict(k13=6, k15=7, k17=8, k19=9, k21=10, k23=11, k25=12, k27=13)
        pad_size = PADDING['{}'.format(b_range)]
        n_pad = ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size))
        copy_img_np = np.pad(copy_img_np, n_pad, 'reflect')

        if kernels is not None:
            for j in range (batch_size):
                index = random.randint(0,2249)
                blur_img_np[j,0,:,:]= signal.convolve(copy_img_np[j,0,:,:],kernels[index,:,:],mode='valid')
                blur_img_np[j,1,:,:]= signal.convolve(copy_img_np[j,1,:,:],kernels[index,:,:],mode='valid')
                blur_img_np[j,2,:,:]= signal.convolve(copy_img_np[j,2,:,:],kernels[index,:,:],mode='valid')
    blur_img_np = blur_img_np + (1.0/255.0)* np.random.normal(0,4,blur_img_np.shape) 
    
    blur_tensor = torch.from_numpy(blur_img_np)  
    blur_tensor = blur_tensor.float()

    assert image_tensor.size() == blur_tensor.size()
    return blur_tensor 


def get_blurkernels(kernel_path):
    blur_range = ['k0', 'k13', 'k15', 'k17', 'k19', 'k21', 'k23', 'k25', 'k27']
    kernel_dict = OrderedDict()
    for blur_size in blur_range: 
        if blur_size == 'k0':
            kernel_dict['{}'.format(blur_size)] = None
        else:        
            k_filename_kernel =os.path.join(kernel_path, 'blur_{}.mat'.format(blur_size))
            kernel_file = scipy.io.loadmat(k_filename_kernel)
            kernels = np.array(kernel_file['blurs_{}'.format(blur_size)])
            kernels = kernels.transpose([2,0,1])
            kernel_dict['{}'.format(blur_size)] = kernels
    return kernel_dict


def adjust_learning_rate(schedulers):
    for scheduler in schedulers:
        scheduler.step()   


def load_resume(resume_state_path, optimizers, schedulers):
    resume_state = torch.load(resume_state_path)
    resume_optimizers = resume_state['optimizers']
    resume_schedulers = resume_state['schedulers']
    # assert len(resume_optimizers) == len(optimizers), 'Wrong lengths of optimizers'
    assert len(resume_schedulers) == len(schedulers), 'Wrong lengths of schedulers'
    for i, (_, opt) in enumerate(optimizers.items()):
        opt.load_state_dict(resume_optimizers[i])

    # for i, o in enumerate(resume_optimizers):
    #     optimizers[i].load_state_dict(o)
    for i, s in enumerate(resume_schedulers):
        schedulers[i].load_state_dict(s)
    start_epoch = resume_state['epoch']
    # current_step = resume_state['iter']    
    return start_epoch


def load_network(model, load_path, strict=True):
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.load_state_dict(torch.load(load_path), strict=strict)


def print_network(model):    
    def _get_network_description(network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
    s, n = _get_network_description(model)
    if isinstance(model, nn.DataParallel):
        net_struc_str = '{} - {}'.format(model.__class__.__name__,
                                            model.module.__class__.__name__)
    else: net_struc_str = '{}'.format(model.__class__.__name__)
    log = 'Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n)
    return log, s


def save_network(GENERATORS, DISCRIMINATORS, save_path, is_best):
    
    def _save(model, model_path):
        if isinstance(model, nn.DataParallel):
            network = model.module
            state_dict = network.state_dict()
        else: 
            state_dict = model.state_dict()
        
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, model_path)

    for name, model in GENERATORS.items():
        if model is not None:
            save_filename = '{}_last.pth'.format(name)
            model_path = os.path.join(save_path, save_filename)
            _save(model, model_path)  

    for name, model in DISCRIMINATORS.items():
        if model is not None:
            save_filename = '{}_last.pth'.format(name)
            model_path = os.path.join(save_path, save_filename)
            _save(model, model_path)   
    
    if is_best:
        for name, model in GENERATORS.items():
            if model is not None:
                save_filename = '{}_best.pth'.format(name)
                model_path = os.path.join(save_path, save_filename)
                _save(model, model_path)
   

def save_state(schedulers, optimizers, save_path, epoch):
    state = {'epoch': epoch, 'schedulers': [], 'optimizers': []}    
    for s in schedulers:
        state['schedulers'].append(s.state_dict())
    for _, o in optimizers.items():
        if o is not None:
            state['optimizers'].append(o.state_dict())
    save_filename = 'train_state.state'
    save_path = os.path.join(save_path, save_filename)
    torch.save(state, save_path) 


def prepare(l, device, volatile=False):
    def _prepare(tensor): return tensor.to(device)           
    return [_prepare(_l) for _l in l]


def tensor2img_list(tensor_list):    
    def _tensor2img_list(img_tensor):
        img_denorm = denorm(img_tensor)
        img_np = tensor2img(img_denorm)
        return img_np
    return [_tensor2img_list(_l) for _l in tensor_list]


def create_grid(samples, img_files):
    """
    utility function to create a grid of GAN samples
    :param samples: generated samples for storing list[Tensors]
    :param img_files: list of names of files to write
    :return: None (saves multiple files)
    """
    from torchvision.utils import save_image
    from numpy import sqrt
    
    # save the images:
    for sample, img_file in zip(samples, img_files):
        sample = torch.clamp((sample.detach() / 2) + 0.5, min=0, max=1)
        save_image(sample, img_file, nrow=int(sqrt(sample.shape[0])))


if __name__ == '__main__':
    main()



