CONFIG_FILE = './config/train_config.py'
KEYWORD = '190820_rrdb5_imgl1'
is_gpu = True
gpu_ids = [0]
#===========================================================================
Paths = {
    'save_dir': '/home/ltb/FocusFace/Resutls/test004_190820/{}'.format(KEYWORD),
    'load_path_G' : None,#'/home/ltb/FocusFace/Resutls/test002_190818/190818_resnet20_FaceFocus_featL1/models/netG_ep30.pth',
    'load_path_D' : None,
    'resume_state' : None,#'/home/ltb/FocusFace/Resutls/test002_190818/190818_resnet20_FaceFocus_featL1/states/ep30_iter96875.state',
    'load_FR_path': '/home/ltb/FocusFace/sphere/sphere20a_20171020.pth',
}
#===========================================================================
Training = {
    'total_epochs': 50,
    'val_n_epoch': 5,
    'save_n_epoch': 5,
    'lr_G' : 1e-4, 'lr_D' : 1e-4,
    'lr_steps' : [50000, 100000, 200000, 300000],
    'lr_gamma' : 0.5
}
#===========================================================================
Losses = {
    'l_img_weight': 100,#100,
    'l_face_weight': 0,#20,#20,#30, # ArcFace    
    'l_cont_weight': 0, # VGG19
    'l_adv_weight': 0, # Discriminator   
    'netD_type': 'cRaGAN', # 'cRaGAN' | 'RaGAN'
}
#===========================================================================
Dataset = {
    'batch_size': 16, 'num_workers': 8,
    'trainset' : {            
        'image_dir' : '/home/ltb/Datasets/FaceDatasets/sampling_ms1m_align_112/trainset/imgs',
        'train' : True, 'n_data': None,
        'image_size': (112, 96), 'n_channels' : 3, 'interp' : 'cubic', 'rgb_range' : 1,
        'ran_crop': False, 'crop_size': None,
        'gen_lr': False, 'scale': None,
        'add_blur' : True, 'blur_type' : 'G', 'blur_ksize' : 25, 'blur_val' : 100, 
        'add_noise' : True, 'noise_type' : 'G', 'noise_val' : 0.1,
        'is_augment' : True, 'hflip' : True, 'rot' : False, 
    },
    'validset' : {           
        'image_dir' : '/home/ltb/Datasets/FaceDatasets/sampling_ms1m_align_112/validset/imgs',
        'train' : False, 'n_data': 100,
        'image_size': None, 'n_channels' : 3, 'interp' : 'cubic', 'rgb_range' : 1,
        'ran_crop': False, 'crop_size': None,
        'gen_lr': False, 'scale': None,
        'add_blur' : True, 'blur_type' : 'G', 'blur_ksize' : 25, 'blur_val' : 100, 
        'add_noise' : True, 'noise_type' : 'G', 'noise_val' : 0.01,
        'is_augment' : False, 'hflip' : False, 'rot' : False
    },
}
#===========================================================================
Model = {
    # 'Generator' : {
    #     'model_type' : 'RRDBNet',  # 'RRDBNet' | 'ResNet'
    #     'in_nc': 3, 'out_nc': 3, 'n_feat': 64, 'n_blocks': 10, 'gc': 32, 
    #     'norm_type': None, 'act_type' : 'leakyrelu', 'mode' : 'CNA'
    # },
    'Discriminator': {
        'model_type' : 'cRaGAN', 'base_nf': 3, 
        'norm_type': None, 'act_type' : 'leakyrelu', 'mode' : 'CNA'
    },       
    'Generator' : {
        'block_type' : 'rrdb',  # 'rrdb' | 'resnet'
        'in_nc': 3, 'out_nc': 3, 'n_feat': 64, 'n_blocks': 5, 
        'norm_type': None, 'act_type' : 'relu', 'mode' : 'CNA', 'res_scale': 0.2
    }
}
#===========================================================================





