CONFIG_FILE = './config/train_config.py'
KEYWORD = '190819_resnet20_FaceFocus_contL1featL1'
is_gpu = True
gpu_ids = [3]
#===========================================================================
Paths = {
    'save_dir': '/home/ltb/FocusFace/Resutls/test002_190818/{}'.format(KEYWORD),
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
    'l_img_weight': 0,#100,
    'l_face_weight': 10,#30, # ArcFace    
    'l_cont_weight': 10, # VGG19
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
        'add_blur' : True, 'blur_type' : 'G', 'blur_ksize' : 15, 'blur_val' : 100, 
        'add_noise' : False, 'noise_type' : 'G', 'noise_val' : 10,
        'is_augment' : True, 'hflip' : True, 'rot' : False, 
        'ran_crop': False, 'crop_size': 128
    },
    'validset' : {           
        'image_dir' : '/home/ltb/Datasets/FaceDatasets/sampling_ms1m_align_112/validset/imgs',
        'train' : False, 'n_data': None,
        'image_size': (112, 96), 'n_channels' : 3, 'interp' : 'cubic', 'rgb_range' : 1,
        'add_blur' : True, 'blur_type' : 'G', 'blur_ksize' : 15, 'blur_val' : 100, 
        'add_noise' : False, 'noise_type' : 'G', 'noise_val' : 10,
        'is_augment' : False, 'hflip' : True, 'rot' : False, 
        'ran_crop': False, 'crop_size': 128
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
        'in_nc' : 6, 'base_nf': 3, 
        'norm_type': None, 'act_type' : 'leakyrelu', 'mode' : 'CNA'
    },       
    'Generator' : {
        'model_type' : 'ResNet',  # 'RRDBNet' | 'ResNet'
        'in_nc': 3, 'out_nc': 3, 'n_feat': 64, 'n_blocks': 20, 
        'norm_type': None, 'act_type' : 'relu', 'mode' : 'CNA', 'res_scale': 0.2
    }
}
#===========================================================================





