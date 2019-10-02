import os
import time
from collections import OrderedDict
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from data.simple_dataset import simpleDataset
from model.Generatros import FocusNet
from model.Discriminators import Discriminator_112, NLayerDiscriminator
from loss.sphereface_extractor import SphereFace_Extractor
from loss.gan_loss import GAN_Loss
from loss.vgg_extractor import VGG19_FeatureExtractor
from utils.weight_initializer import init_weights
from utils.utils import tensor2img, bgr2ycbcr, make_dirs, write_log
from utils.eval_metric import calculate_psnr, calculate_ssim

class Trainer():
    def __init__(self, args):
        self.args = args
        # 1. DIRS setup
        self._set_filesystems(args)
        # 2. check GPu Setting
        self._check_CUDA(args)
        self.schedulers = []
        self.optimizers = []
        # ===================================================================
        # Datasets
        self.train_set = simpleDataset(**args.Dataset['trainset'])
        self.val_set = simpleDataset(**args.Dataset['validset'])        
        self.loader_train = torch.utils.data.DataLoader(
            self.train_set, batch_size=args.Dataset['batch_size'],
            shuffle=True, num_workers=args.Dataset['num_workers'],
            drop_last=True, pin_memory=True)        
        self.loader_val =  torch.utils.data.DataLoader(
            self.val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        print('Set Dataset')
        # ===================================================================
        # 3. build Generator
        self.netG = FocusNet(**args.Model['Generator'])           

        # 3.1. geneartor initialize weights
        init_weights(self.netG, init_type='kaiming', scale=0.1)
        self.netG = self.netG.to(self.device)
        if self.is_gpu:
            self.netG = nn.DataParallel(self.netG, device_ids=self.gpu_ids)
        # 3.2. load pretrained model
        if args.Paths['load_path_G']:
            self.load_network(self.netG, args.Paths['load_path_G'])
        self.optim_G = torch.optim.Adam(
            self.netG.parameters(), lr=args.Training['lr_G'], 
            weight_decay=0, betas=(0.9, 0.999))
        self.optimizers.append(self.optim_G)
        self.print_network(self.netG, args.Model['Generator']['block_type'])        

        # Define Lossnetworks, Losses
        # self.cri_l1 = None 
        self.cri_l1 = None
        if args.Losses['l_img_weight'] != 0:
            if self.cri_l1 is None: 
                self.cri_l1 = nn.L1Loss().to(self.device)
            self.l_img_w = args.Losses['l_img_weight']
        else: self.l_img_w = None
                
        # VGG
        if args.Losses['l_cont_weight'] != 0:
            self.l_cont_w = args.Losses['l_cont_weight']           
            if self.cri_l1 is None: 
                self.cri_l1 = nn.L1Loss().to(self.device)            
            self.netVGG = VGG19_FeatureExtractor(
                use_bn=False, use_input_norm=True, device=self.device).to(self.device)
            if self.is_gpu:
                self.netVGG = nn.DataParallel(self.netVGG, device_ids=self.gpu_ids)
            self.netVGG.eval()
        else: 
            self.netVGG = None
            self.l_cont_w = None
        
        # Discriminators
        if args.Losses['l_adv_weight'] != 0:
            self.l_adv_w = args.Losses['l_adv_weight']
            self.cri_gan = GAN_Loss(
                gan_type='vanilla', 
                real_label_val=1.0, fake_label_val=0.0).to(self.device)            
            self.netD_type = args.Losses['netD_type']
            self.netD = NLayerDiscriminator(input_nc=6, ndf=64, n_layers=5)
            # self.netD = Discriminator_112(**args.Model['Discriminator']).to(self.device)            
            if self.is_gpu:
                self.netD = nn.DataParallel(self.netD, device_ids=self.gpu_ids)
            self.print_network(self.netD, 'Discriminator')
            if args.Paths['load_path_D']:
                self.load_network(self.netD, args.Paths['load_path_D'])
            self.optim_netD = torch.optim.Adam(self.netD.parameters(), 
                lr=args.Training['lr_D'], weight_decay=0, betas=(0.9, 0.999))
            self.optimizers.append(self.optim_netD)
        else: 
            self.netD = None
            self.l_adv_w = None
        
        # ArcFace
        if args.Losses['l_face_weight'] != 0:
            self.l_face_w = args.Losses['l_face_weight']
            if self.cri_l1 is None: 
                self.cri_l1 = nn.L1Loss().to(self.device) 
            self.netArc = SphereFace_Extractor(load_path=args.Paths['load_FR_path']).to(self.device)
            if self.is_gpu:
                self.netArc = nn.DataParallel(self.netArc, device_ids=self.gpu_ids)
            self.netArc.eval()
        else: 
            self.netArc = None
            self.l_face_w = None
        
        # schedulers
        for optimizer in self.optimizers:
            self.schedulers.append(lr_scheduler.MultiStepLR(optimizer,
                    args.Training['lr_steps'], args.Training['lr_gamma'])) 
    
        self.total_epochs = args.Training['total_epochs']
        # 1.1 Resume Training
        self.resume_state = None
        if  args.Paths['resume_state']:
            self.resume_state = torch.load(args.Paths['resume_state'])
            self.resume_training(self.resume_state)
            self.start_epoch = self.resume_state['epoch']
            self.current_step = self.resume_state['iter']
        else:
            self.current_step = 0
            self.start_epoch = 0   

        #self.keywoard = args.keywoard      
        self.val_n_epoch = args.Training['val_n_epoch']       
        self.save_n_epoch = args.Training['save_n_epoch']
        #log_dict = OrderedDict()
    
    def train(self):
        total_iters = len(self.loader_train)
        total_epochs = self.total_epochs
        current_step = self.current_step
        start_epoch = self.start_epoch
        print('Start Training')
        self.netG.train()
        if self.netD : 
            self.netD.train()       

        for epoch in range(start_epoch, total_epochs): 
            start_iter_time = 0
            log_dict = OrderedDict()
            for iters, (gt_data, in_data, _) in enumerate(self.loader_train):
                start_iter_time = time.time()
                current_step += 1
                # update learning rate
                self.update_learning_rate()
                # training Generator
                if self.netD : 
                    for p in self.netD.parameters():
                        p.requires_grad = False
                self.optim_G.zero_grad()
                loss_G_total = 0
                in_data, gt_data = self.prepare([in_data, gt_data])
                out_data = self.netG(in_data)

                # Image Loss
                if self.l_img_w:
                    loss_img = self.l_img_w * self.cri_l1(gt_data, out_data)
                    log_dict['loss_img'] = loss_img.item()
                    loss_G_total += loss_img    

                # VGG Loss
                if self.l_cont_w:
                    fake_feat = self.netVGG(out_data)
                    real_feat = self.netVGG(gt_data).detach()                    
                    loss_content = self.l_cont_w * \
                            self.cri_l1(fake_feat, real_feat)
                    log_dict['loss_content'] = loss_content.item()
                    loss_G_total += loss_content
                                
                # Adversarial Loss
                if self.l_adv_w:                        
                    if self.netD_type == 'cRaGAN':
                        cat_out = torch.cat((out_data, in_data), 1).detach()
                        cat_gt = torch.cat((gt_data, in_data), 1).detach()
                        pred_g_fake = self.netD(cat_out)
                        pred_d_real = self.netD(cat_gt).detach()
                    else: 
                        pred_g_fake = self.netD(out_data)
                        pred_d_real = self.netD(gt_data).detach()
                    loss_adv_i = self.l_adv_w *  \
                        (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) + \
                        self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
                    log_dict['loss_adv_img'] = loss_adv_i.item()
                    loss_G_total += loss_adv_i
                
                # Arc face feature Loss
                if self.l_face_w:                                        
                    fake_feat = self.netArc(out_data)
                    real_feat = self.netArc(gt_data).detach() 
                    #print(len(fake_feat))                  
                    #print(real_feat.shape) 
                    # loss_face = 0
                    # for i in range(len(fake_feat)):
                    #     target = torch.empty_like(real_feat[0].view(-1, 1)).fill_(1.0)
                    #     loss_face += self.l_face_w * \
                    #             self.cri_cos(fake_feat[i].view(-1, 1), real_feat[i].view(-1, 1), target=target)
                    loss_face = self.l_face_w * \
                        self.cri_l1(fake_feat, real_feat)
                    log_dict['loss_face'] = loss_face.item()
                    loss_G_total += loss_face                    

                log_dict['loss_G_total'] = loss_G_total.item()
                loss_G_total.backward()
                self.optim_G.step()

                # training Discriminator
                if self.l_adv_w:                    
                    for p in self.netD.parameters():
                        p.requires_grad = True
                        self.optim_netD.zero_grad()
                        loss_netD_total = 0
                        if self.netD_type == 'cRaGAN':
                            cat_out = torch.cat((out_data, in_data), 1).detach()
                            cat_gt = torch.cat((gt_data, in_data), 1).detach()
                            pred_d_real = self.netD(cat_gt)
                            pred_d_fake = self.netD(cat_out.detach())
                        else:
                            pred_d_real = self.netD(gt_data)
                            pred_d_fake = self.netD(out_data.detach())  # detach to avoid BP to G
                    
                        l_netD_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                        log_dict['loss_netD_real'] = l_netD_real.item()
                    
                        l_netD_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                        log_dict['loss_netD_fake'] = l_netD_fake.item()

                        loss_netD_total = (l_netD_real + l_netD_fake) / 2
                        log_dict['loss_netD_total'] = loss_netD_total.item()
                        loss_netD_total.backward()
                        self.optim_netD.step()
                t_per_iter = time.time() - start_iter_time
                print('============================================')
                print('[GPUID: {}]'.format(self.gpu_ids))
                print(self.args.KEYWORD)
                self.print_loss(
                    t_per_iter, log_dict, epoch, iters, total_epochs, total_iters)

            # validation
            if epoch % self.val_n_epoch == 0 or epoch+1==total_epochs:
                self.netG.eval()   
                if self.netD: 
                    self.netD.eval()
                              
                with torch.no_grad():
                    test_results = OrderedDict()
                    test_results['psnr'] = []
                    test_results['ssim'] = []
                    test_results['psnr_y'] = []
                    test_results['ssim_y'] = []
                    for _, (gt_data, in_data, filename) in enumerate(self.loader_val, 0):                            
                        # tensor to device
                        gt_data, in_data= self.prepare([gt_data, in_data])
                        # inference
                        out_data = self.netG(in_data)
                        # tensor2np
                        out_img = tensor2img(out_data)                                            
                        gt_img = tensor2img(gt_data)                        
                        # save images
                        filename = filename[0] 
                        val_dir = self.path_dict['valid_dir']
                        in_path = self.path_dict['valid_dir_in']
                        gt_path = self.path_dict['valid_dir_gt']
                        out_path = self.path_dict['valid_dir_out']
                        if epoch == 0:                            
                            in_img = tensor2img(in_data)
                            # img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(os.path.join(in_path, 
                                '{}.png'.format(filename)), in_img)
                            # img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                            cv2.imwrite(os.path.join(gt_path, 
                                '{}.png'.format(filename)), gt_img)
                        out_path = os.path.join(out_path, 'ep{}'.format(epoch))
                        if not os.path.exists(out_path): os.makedirs(out_path)
                        # img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(os.path.join(out_path, 
                            '{}.png'.format(filename)), out_img)
                        # calculate PSNR and SSIM
                        gt_img = gt_img / 255.
                        out_img = out_img / 255.
                        if epoch == 0:
                            in_img = in_img / 255.
                            input_metric = OrderedDict()
                            input_metric['psnr'] = []
                            input_metric['ssim'] = []
                            input_metric['psnr_y'] = []
                            input_metric['ssim_y'] = []                            
                            psnr = calculate_psnr(in_img * 255, gt_img * 255)
                            ssim = calculate_ssim(in_img * 255, gt_img * 255)
                            input_metric['psnr'].append(psnr)
                            input_metric['ssim'].append(ssim)                        
                            out_img_y = bgr2ycbcr(in_img, only_y=True)
                            gt_img_y = bgr2ycbcr(gt_img, only_y=True)                        
                            psnr_y = calculate_psnr(out_img_y * 255, gt_img_y * 255)
                            ssim_y = calculate_ssim(out_img_y * 255, gt_img_y * 255)
                            input_metric['psnr_y'].append(psnr_y)
                            input_metric['ssim_y'].append(ssim_y)   
                            log = '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'\
                                .format(filename, psnr, ssim, psnr_y, ssim_y)
                            write_log(
                                log, val_dir, save_name='input_PSNR_SSIM')                                
                        # HR
                        psnr = calculate_psnr(out_img * 255, gt_img * 255)
                        ssim = calculate_ssim(out_img * 255, gt_img * 255)
                        test_results['psnr'].append(psnr)
                        test_results['ssim'].append(ssim)                        
                        out_img_y = bgr2ycbcr(out_img, only_y=True)
                        gt_img_y = bgr2ycbcr(gt_img, only_y=True)                        
                        psnr_y = calculate_psnr(out_img_y * 255, gt_img_y * 255)
                        ssim_y = calculate_ssim(out_img_y * 255, gt_img_y * 255)
                        test_results['psnr_y'].append(psnr_y)
                        test_results['ssim_y'].append(ssim_y)   
                        log = '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'\
                            .format(filename, psnr, ssim, psnr_y, ssim_y)
                        write_log(
                            log, val_dir, save_name='val_PSNR_SSIM_ep{}'.format(epoch))
                    if epoch == 0:
                        # Average PSNR/SSIM results
                        ave_psnr = sum(input_metric['psnr']) / len(input_metric['psnr'])
                        ave_ssim = sum(input_metric['ssim']) / len(input_metric['ssim'])                    
                        ave_psnr_y = sum(input_metric['psnr_y']) / len(input_metric['psnr_y'])
                        ave_ssim_y = sum(input_metric['ssim_y']) / len(input_metric['ssim_y'])                    
                        log = 'Avearge PSNR: {:.6f} dB; SSIM: {:.6f} PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}'\
                            .format(ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y)
                        write_log(
                            log, val_dir, save_name='input_PSNR_SSIM')
                    # Average PSNR/SSIM results
                    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])                    
                    ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                    ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])                    
                    log = '[Epoch:  {}] Avearge PSNR: {:.6f} dB; SSIM: {:.6f} PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}'\
                        .format(epoch, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y)
                    print(log)
                    write_log(
                        log, val_dir, save_name='val_PSNR_SSIM')
                    write_log(
                        log, val_dir, save_name='val_PSNR_SSIM_ep{}'.format(epoch))
        
            # save models and training states
            if epoch % self.save_n_epoch == 0 or epoch+1==total_epochs:
                self.save_network(self.netG, 'netG', epoch)
                if self.netD:
                    self.save_network(self.netD, 'netD', epoch)
                self.save_training_state(epoch, current_step)

    def print_loss(self,t_per_iter, log_dict, epoch, iters, total_epochs, total_iters):
        log = '[Epochs:{}|{}][Iters:{}|{}] [{:.2f}sec]'.format(
            epoch, total_epochs, iters, total_iters, t_per_iter)
        save_path = self.path_dict['save_dir']
        save_name = 'loss_log'
        print(log)  
        write_log(log, save_path, save_name)    
        for k, v in log_dict.items():
            log = '\t{} : {:.6f}'.format(k, v)
            print(log)
            write_log(log, save_path, save_name)
   
    def prepare(self, l, volatile=False):
        def _prepare(tensor): return tensor.to(self.device)           
        return [_prepare(_l) for _l in l]
    
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def load_network(self, model, load_path, strict=True):
        if isinstance(model, nn.DataParallel):
            model = model.module
        model.load_state_dict(torch.load(load_path), strict=strict)
        print('Load Model')
    
    def save_network(self, network, network_label, epoch):
        save_filename = '{}_ep{}.pth'.format(network_label, epoch)
        save_path = os.path.join(self.path_dict['model_dir'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = 'ep{}_iter{}.state'.format(epoch, iter_step)
        save_path = os.path.join(self.path_dict['state_dir'], save_filename)
        torch.save(state, save_path)

    def print_network(self, model, save_name):    
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
        write_log(log, self.path_dict['save_dir'], save_name)
        write_log(s, self.path_dict['save_dir'], save_name)

    def _set_filesystems(self, args):
        self.path_dict = OrderedDict()
        self.path_dict['save_dir'] = make_dirs(args.Paths['save_dir'])
        self.path_dict['model_dir'] = make_dirs(os.path.join(
                args.Paths['save_dir'], 'models'))
        self.path_dict['state_dir'] = make_dirs(os.path.join(
                args.Paths['save_dir'], 'states'))
        val_dir = make_dirs(os.path.join(args.Paths['save_dir'], 'valid'))
        self.path_dict['valid_dir'] = val_dir
        self.path_dict['valid_dir_gt'] = make_dirs(os.path.join(val_dir, 'gt'))
        self.path_dict['valid_dir_in'] = make_dirs(os.path.join(val_dir, 'in'))
        self.path_dict['valid_dir_out'] = make_dirs(os.path.join(val_dir, 'out')) 
        
    def _check_CUDA(self, args):        
        if args.is_gpu:
            assert torch.cuda.is_available(), 'GPU is not avalialble!'
            self.is_gpu = True
            self.gpu_ids = args.gpu_ids
            torch.cuda.set_device(self.gpu_ids[0])
            torch.backends.cudnn.benckmark = True           
            self.device = torch.device('cuda') 
        else :
            self.is_gpu = False
            self.device = torch.device('cpu') 