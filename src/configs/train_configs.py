from easydict import EasyDict


def get_configs():
    cfg = EasyDict()    
    #===============================================================================
    # Training configurations
    #===============================================================================
    # GPU IDs to Use
    cfg.GPU_IDS = [0]
    # cfg.DEBUG_MODE = False    
    cfg.KEYPOINT = 'TRAINING MGPL-GAN'
    # Save Results
    cfg.SAVE_DIR = '' 

    cfg.TOTAL_EPOCH = 200
    cfg.VISUAL_N_EPOCH = 10
    
    #===============================================================================
    # Loss, Optimizers
    #===============================================================================        
    cfg.LOSS_L1_WEIGHT = 1 
    cfg.LOSS_ADV_WEIGHT = 0.05 
    cfg.LOSS_VGG_WEIGHT = 0.05
   
    cfg.INIT_TYPE = 'kaiming'
    cfg.INIT_LR_G = 1e-4
    cfg.WEIGHT_DECAY_G = 0
    cfg.INIT_LR_D = 1e-4
    cfg.WEIGHT_DECAY_D = 0
    cfg.LR_GAMMA = 0.5

    #================================================================================
    # Train & Valid Datset
    #================================================================================    
    cfg.TRAINSET = {
        'data_dir': '',
        'n_classes': 4, 'mode': 'train'
    }
    cfg.VALIDSET = {
        'data_dir': '',
    }
    
    cfg.TRAIN_BATCH_SIZE = 16
    cfg.VALID_BATCH_SIZE = 16
    cfg.NUM_WORKERS = 8
    
    cfg.TRAIN_KERNEL_PATH = ''
    cfg.TRAIN_BLUR_KERNEL = ['k13', 'k15', 'k17', 'k19', 'k21', 'k23', 'k25', 'k27']
      
    #================================================================================
    # Models 
    #================================================================================
        
    cfg.LOAD_PATH = {  
        'load_path': None, 
        'netG': None,
        'netD': None,
        'VGGFace16': '', # VGGFace16 Path
        'RESUME_STATE': None,
    }    

    return cfg
