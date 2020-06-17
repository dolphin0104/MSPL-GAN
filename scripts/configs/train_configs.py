from easydict import EasyDict


def get_configs():
    cfg = EasyDict()    
    #===============================================================================
    # Training configurations
    #===============================================================================
    cfg.GPU_IDS = [0]
    # cfg.DEBUG_MODE = False    
    cfg.KEYPOINT = '200531_COMP_RCAN_GAN'
    cfg.SAVE_DIR = '/home/ltb/Projects/FocusFace/results/COMPONENT/200531_COMP_RCAN_GAN' 

    cfg.TOTAL_EPOCH = 230
    cfg.VISUAL_N_EPOCH = None
    
    #===============================================================================
    # Loss, Optimizers
    #===============================================================================        
    cfg.LOSS_L1_WEIGHT = 1 #None#None#1 # 5 #0.2 #1 
    cfg.LOSS_ADV_WEIGHT = 0.05 #0.5#0.05#None#0.05#0.05#0.05#0.5 #None#1#None#0.05#0.05#0.05#0.1#0.1 #None #None #0.1 #0.1#None#0.1
    cfg.LOSS_VGG_WEIGHT = None #0.05#0.05#None#0.05#0.05#0.05#0.05#0.05#0.05#0.05
   
    cfg.INIT_TYPE = 'kaiming'
    cfg.INIT_LR_G = 1e-4
    cfg.WEIGHT_DECAY_G = 0
    cfg.INIT_LR_D = 1e-4
    cfg.WEIGHT_DECAY_D = 0
    cfg.LR_STEPS = [100000, 200000, 250000, 300000]
    cfg.LR_GAMMA = 0.5

    #================================================================================
    # Train & Valid Datset
    #================================================================================    
    cfg.TRAINSET = {
        'data_dir': '/home/ltb/storage/ltb/dataset/CelebAMask-HQ/partion',
        'n_classes': 4, 'mode': 'train'
    }
    cfg.VALIDSET = {
        'data_dir': '/home/ltb/storage/ltb/dataset/valid_200311',
    }
    
    cfg.TRAIN_BATCH_SIZE = 16
    cfg.VALID_BATCH_SIZE = 16
    cfg.NUM_WORKERS = 8
    
    cfg.TRAIN_KERNEL_PATH = '/home/ltb/storage/ltb/dataset/face_deblur/Kernel_val_train_nips19/train'
    cfg.TRAIN_BLUR_KERNEL = ['k13', 'k15', 'k17', 'k19', 'k21', 'k23', 'k25', 'k27']
    
    cfg.TEST_KERNEL_PATH = '/home/ltb/storage/ltb/dataset/face_deblur/Kernel_val_train_nips19/val'
    cfg.TEST_BLUR_KERNEL = ['k13', 'k15', 'k17', 'k19', 'k21', 'k23', 'k25', 'k27']
      
    #================================================================================
    # Models 
    #================================================================================
        
    cfg.LOAD_PATH = {  
        'load_path': '/home/ltb/Projects/FocusFace/results/COMPONENT/200531_COMP_RCAN_GAN/checkpoints', #'/home/ltb/Projects/FocusFace/results/COMPONENT/200330_COMPv03_GAN_Dense_6block/checkpoints', 
        'netG': '/home/ltb/Projects/FocusFace/results/COMPONENT/200531_COMP_RCAN_GAN/checkpoints/netG_last.pth',#'/home/ltb/Projects/FocusFace/results/COMPONENT/200313_ComponentNet_MSGFeatureStyle/checkpoints/netG_last.pth',
        'netD': '/home/ltb/Projects/FocusFace/results/COMPONENT/200531_COMP_RCAN_GAN/checkpoints/netD_last.pth',#'/home/ltb/Projects/FocusFace/results/COMPONENT/200313_ComponentNet_MSGFeatureStyle/checkpoints/netD_last.pth',        
        'VGGFace16': '/home/ltb/storage/ltb/trained_weight/VGGFace16/VGGFace16.pth',
        'RESUME_STATE': '/home/ltb/Projects/FocusFace/results/COMPONENT/200531_COMP_RCAN_GAN/checkpoints/train_state.state',#'/home/ltb/Projects/FocusFace/results/COMPONENT/200313_ComponentNet_MSGFeatureStyle/checkpoints/train_state.state'
    }    

    return cfg
